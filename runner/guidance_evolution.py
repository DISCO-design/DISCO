#!/usr/bin/env python3
"""Evolutionary outer loop for DISCO motif-guidance experiments.

This is the DISCO equivalent of ProteinGenerator's `evolutionary_motif_search.py`.
It is deliberately an outer runner: it builds Hydra commands, launches optional
DISCO inference runs, scores hard output sequences, and evolves parameter sets.

Use `--dry-run --run-unit-checks` for local CPU validation without starting
model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.evolution import (
    DiscoEvaluation,
    DiscoGuidanceGenome,
    build_inference_command,
    evaluation_row,
    fitness_score,
    initial_population,
    materialize_template_input_json,
    mutate_genome,
    next_generation,
    parse_motifs,
    sequence_from_disco_sequence_text,
)
from disco.model.cogen_inference.guidance.disco_analysis import analyze_disco_dump_rows


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluation_fieldnames(motifs: list[str]) -> list[str]:
    genome_fields = [
        "n_step",
        "n_cycle",
        "gradient_scale",
        "motif_weight_all",
        "motif_weight_many",
        "motif_top_k",
        "motif_temperature",
        "local_rescue_enabled",
        "local_rescue_top_k",
        "local_rescue_bonus",
        "local_rescue_min_probability",
        "save_best_motif_weight",
        "save_best_confidence_weight",
    ]
    return [
        "generation",
        "individual",
        "fitness",
        "confidence",
        "unique_motifs",
        "total_motifs",
        "sequence",
        "run_dir",
        *genome_fields,
        *[f"{motif}_count" for motif in motifs],
    ]


def analyze_disco_dump(
    *,
    dump_dir: Path,
    motifs: list[str],
    fitness_mode: str,
    target_total_motifs: int,
) -> dict[str, object]:
    """Return the best hard-motif row from a DISCO dump directory."""

    rows = analyze_disco_dump_rows(
        dump_dir=dump_dir,
        motifs=motifs,
        motif_all_weight=1.0,
        motif_many_weight=0.0,
    )
    best_row = None
    best_fitness = float("-inf")
    for row in rows:
        confidence = float(row["confidence"])
        row_fitness = fitness_score(
            confidence=confidence,
            unique_motifs=int(row["unique_motifs"]),
            total_motifs=int(row["total_motifs"]),
            required_motifs=len(motifs),
            target_total_motifs=target_total_motifs,
            mode=fitness_mode,
        )
        if row_fitness > best_fitness:
            best_fitness = row_fitness
            best_row = {
                "sequence_path": str(row["sequence_path"]),
                "sequence": str(row["sequence"]),
                "confidence": confidence,
                "fitness": row_fitness,
                "unique_motifs": int(row["unique_motifs"]),
                "total_motifs": int(row["total_motifs"]),
                "counts": {motif: int(row.get(f"{motif}_count", 0)) for motif in motifs},
                "confidence_match_status": row["confidence_match_status"],
                "confidence_path": row["confidence_path"],
            }

    assert best_row is not None
    return best_row


def evaluate_individual(
    *,
    generation: int,
    individual: int,
    genome: DiscoGuidanceGenome,
    args: argparse.Namespace,
    motifs: list[str],
    disco_root: Path,
) -> DiscoEvaluation:
    generation_dir = args.out / f"generation_{generation:03d}"
    run_dir = generation_dir / f"individual_{individual:03d}_{genome.label()}"
    dump_dir = run_dir / "disco_output"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_json_path = materialize_template_input_json(
        base_input_json_path=args.input_json_path,
        output_json_path=run_dir / "input_template.json",
        template_sequence=genome.template_sequence,
    )

    seed = args.seeds[individual % len(args.seeds)]
    command = build_inference_command(
        python_executable=sys.executable,
        disco_root=disco_root,
        genome=genome,
        motifs=motifs,
        input_json_path=input_json_path,
        dump_dir=dump_dir,
        seed=seed,
        experiment=args.experiment,
    )
    (run_dir / "command.txt").write_text(" ".join(command) + "\n")

    manifest_path = run_dir / "batch_manifest.json"
    manifest = {
        "status": "planned",
        "script_name": "guidance_evolution.py",
        "run_type": "disco_guidance_evolution",
        "experiment_id": args.experiment_id,
        "generation": generation,
        "individual": individual,
        "seed": seed,
        "run_dir": str(run_dir),
        "dump_dir": str(dump_dir),
        "input_json_path": str(input_json_path),
        "motifs": motifs,
        "fitness_mode": args.fitness_mode,
        "target_total_motifs": args.target_total_motifs,
        "parameters": asdict(genome),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.dry_run:
        manifest["status"] = "dry_run"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return DiscoEvaluation(
            generation=generation,
            individual=individual,
            genome=genome,
            run_dir=run_dir,
            fitness=0.0,
            confidence=0.0,
            unique_motifs=0,
            total_motifs=0,
            sequence="",
            counts={motif: 0 for motif in motifs},
            command=command,
        )

    log_path = run_dir / "run.log"
    try:
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with log_path.open("w") as log_handle:
            subprocess.run(
                command,
                cwd=disco_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

        best = analyze_disco_dump(
            dump_dir=dump_dir,
            motifs=motifs,
            fitness_mode=args.fitness_mode,
            target_total_motifs=args.target_total_motifs,
        )
        manifest.update(
            {
                "status": "finished",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "best_fitness": best["fitness"],
                "best_confidence": best["confidence"],
                "best_unique_motifs": best["unique_motifs"],
                "best_total_motifs": best["total_motifs"],
                "best_sequence": best["sequence"],
                "best_sequence_path": best["sequence_path"],
                "best_confidence_path": best["confidence_path"],
                "best_confidence_match_status": best["confidence_match_status"],
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return DiscoEvaluation(
            generation=generation,
            individual=individual,
            genome=genome,
            run_dir=run_dir,
            fitness=float(best["fitness"]),
            confidence=float(best["confidence"]),
            unique_motifs=int(best["unique_motifs"]),
            total_motifs=int(best["total_motifs"]),
            sequence=str(best["sequence"]),
            counts=dict(best["counts"]),
            command=command,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


def write_generation_csv(
    path: Path,
    evaluations: list[DiscoEvaluation],
    motifs: list[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluation_fieldnames(motifs))
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(evaluation_row(evaluation, motifs))


def write_best_outputs(out_dir: Path, best: DiscoEvaluation) -> None:
    (out_dir / "best_config.json").write_text(
        json.dumps(
            {
                "fitness": best.fitness,
                "confidence": best.confidence,
                "unique_motifs": best.unique_motifs,
                "total_motifs": best.total_motifs,
                "counts": best.counts,
                "sequence": best.sequence,
                "run_dir": str(best.run_dir),
                "genome": asdict(best.genome),
                "command": best.command,
            },
            indent=2,
        )
    )
    (out_dir / "best_sequence.fasta").write_text(
        f">best_disco_guidance_evolution fitness={best.fitness:.6f} "
        f"confidence={best.confidence:.3f} total={best.total_motifs}\n"
        f"{best.sequence}\n"
    )


def run_unit_checks() -> None:
    motifs = parse_motifs("GHK,KTTKS")
    balance_a = fitness_score(
        confidence=0.8,
        unique_motifs=2,
        total_motifs=2,
        required_motifs=2,
        target_total_motifs=6,
        mode="balance",
    )
    balance_b = fitness_score(
        confidence=0.5,
        unique_motifs=2,
        total_motifs=8,
        required_motifs=2,
        target_total_motifs=6,
        mode="balance",
    )
    assert balance_a > balance_b

    max_a = fitness_score(
        confidence=0.8,
        unique_motifs=2,
        total_motifs=3,
        required_motifs=2,
        target_total_motifs=6,
        mode="max_motifs",
    )
    max_b = fitness_score(
        confidence=0.5,
        unique_motifs=2,
        total_motifs=8,
        required_motifs=2,
        target_total_motifs=6,
        mode="max_motifs",
    )
    assert max_b > max_a

    genome = DiscoGuidanceGenome()
    command = build_inference_command(
        python_executable=sys.executable,
        disco_root=Path.cwd(),
        genome=genome,
        motifs=motifs,
        input_json_path=Path("input_jsons/unconditional_config.json"),
        dump_dir=Path("/tmp/disco_evolution_test"),
        seed=0,
    )
    assert "cogen_guidance.enabled=true" in command
    assert "cogen_guidance.motifs=[GHK,KTTKS]" in command

    for _ in range(20):
        mutated = mutate_genome(
            genome,
            mutation_rate=1.0,
            include_local_rescue=True,
            fixed_n_cycle=4,
        )
        assert mutated.n_cycle == 4

    text = ">cogen_seq 0\nCHAIN A type=protein mode=design seq=GHKAA\n"
    assert sequence_from_disco_sequence_text(text) == "GHKAA"


def write_manifest(
    *,
    path: Path,
    args: argparse.Namespace,
    motifs: list[str],
    status: str,
    started_at: str,
    finished_at: str | None = None,
    successful_evaluations: int = 0,
    failed_evaluations: int = 0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "motifs": motifs,
                "run_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motifs", required=True)
    parser.add_argument("--input-json-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--experiment", default="designable")
    parser.add_argument("--experiment-id", default="disco_evolution")
    parser.add_argument("--generations", default=4, type=int)
    parser.add_argument("--population-size", default=8, type=int)
    parser.add_argument("--seeds", default="0", help="comma-separated DISCO seeds")
    parser.add_argument(
        "--fitness-mode",
        default="balance",
        choices=["balance", "max_motifs", "high_confidence"],
    )
    parser.add_argument("--target-total-motifs", default=8, type=int)
    parser.add_argument("--elite-fraction", default=0.2, type=float)
    parser.add_argument("--mutation-rate", default=0.35, type=float)
    parser.add_argument("--max-parallel", default=1, type=int)
    parser.add_argument("--random-seed", default=0, type=int)
    parser.add_argument("--include-local-rescue", action="store_true")
    parser.add_argument("--fixed-n-step", default=None, type=int)
    parser.add_argument("--fixed-n-cycle", default=4, type=int)
    parser.add_argument("--template-from-elites", action="store_true", default=True)
    parser.add_argument("--no-template-from-elites", dest="template_from_elites", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-unit-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    args.out = args.out.resolve()
    args.input_json_path = args.input_json_path.resolve()
    args.seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    started_at = datetime.now(timezone.utc).isoformat()

    if args.run_unit_checks:
        run_unit_checks()
        print("Unit checks passed")
        if args.dry_run and args.generations <= 0:
            return

    motifs = parse_motifs(args.motifs)
    disco_root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    live_path = args.out / "live_evaluations.csv"
    failure_path = args.out / "failed_individuals.csv"
    write_manifest(
        path=manifest_path,
        args=args,
        motifs=motifs,
        status="running",
        started_at=started_at,
    )

    population = initial_population(
        args.population_size,
        include_local_rescue=args.include_local_rescue,
    )
    if args.fixed_n_step is not None:
        population = [
            DiscoGuidanceGenome(**{**asdict(genome), "n_step": args.fixed_n_step})
            for genome in population
        ]
    if args.fixed_n_cycle is not None:
        population = [
            DiscoGuidanceGenome(**{**asdict(genome), "n_cycle": args.fixed_n_cycle})
            for genome in population
        ]

    all_evaluations = []
    failed_evaluations = 0
    for generation in range(args.generations):
        generation_start = time.monotonic()
        print(f"Generation {generation + 1}/{args.generations}")
        evaluations = []

        if args.max_parallel <= 1:
            iterable = enumerate(population)
            for index, genome in iterable:
                try:
                    evaluation = evaluate_individual(
                        generation=generation,
                        individual=index,
                        genome=genome,
                        args=args,
                        motifs=motifs,
                        disco_root=disco_root,
                    )
                except Exception as exc:
                    failed_evaluations += 1
                    append_csv_row(
                        failure_path,
                        ["generation", "individual", "run_dir", "error_type", "error"],
                        {
                            "generation": generation,
                            "individual": index,
                            "run_dir": str(args.out / f"generation_{generation:03d}"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    print(f"  individual {index:03d} failed: {type(exc).__name__}: {exc}")
                    continue
                evaluations.append(evaluation)
                append_csv_row(
                    live_path,
                    evaluation_fieldnames(motifs),
                    evaluation_row(evaluation, motifs),
                )
        else:
            with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
                futures = {
                    executor.submit(
                        evaluate_individual,
                        generation=generation,
                        individual=index,
                        genome=genome,
                        args=args,
                        motifs=motifs,
                        disco_root=disco_root,
                    ): index
                    for index, genome in enumerate(population)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        evaluation = future.result()
                    except Exception as exc:
                        failed_evaluations += 1
                        print(f"  individual {index:03d} failed: {type(exc).__name__}: {exc}")
                        continue
                    evaluations.append(evaluation)
                    append_csv_row(
                        live_path,
                        evaluation_fieldnames(motifs),
                        evaluation_row(evaluation, motifs),
                    )
            evaluations.sort(key=lambda item: item.individual)

        write_generation_csv(args.out / f"generation_{generation:03d}.csv", evaluations, motifs)
        all_evaluations.extend(evaluations)

        if args.dry_run:
            commands = [" ".join(evaluation.command) for evaluation in evaluations]
            (args.out / f"generation_{generation:03d}_commands.txt").write_text(
                "\n".join(commands) + "\n"
            )
            break

        if not evaluations:
            print("  no successful individuals in this generation; stopping early")
            break

        best = max(evaluations, key=lambda item: item.fitness)
        elapsed_min = (time.monotonic() - generation_start) / 60
        print(
            f"  best fitness={best.fitness:.4f} confidence={best.confidence:.3f} "
            f"unique={best.unique_motifs} total={best.total_motifs} "
            f"elapsed={elapsed_min:.1f} min"
        )
        if generation < args.generations - 1:
            population = next_generation(
                evaluations=evaluations,
                population_size=args.population_size,
                elite_fraction=args.elite_fraction,
                mutation_rate=args.mutation_rate,
                motifs=motifs,
                template_from_elites=args.template_from_elites,
                include_local_rescue=args.include_local_rescue,
                fixed_n_step=args.fixed_n_step,
                fixed_n_cycle=args.fixed_n_cycle,
            )

    write_generation_csv(args.out / "evolution_summary.csv", all_evaluations, motifs)
    if all_evaluations and not args.dry_run:
        write_best_outputs(args.out, max(all_evaluations, key=lambda item: item.fitness))
    write_manifest(
        path=manifest_path,
        args=args,
        motifs=motifs,
        status="finished",
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        successful_evaluations=len(all_evaluations),
        failed_evaluations=failed_evaluations,
    )


if __name__ == "__main__":
    main()
