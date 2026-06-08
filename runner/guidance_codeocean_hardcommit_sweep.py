#!/usr/bin/env python3
"""Run a CodeOcean-ready DISCO hard-commit 15-motif sweep.

The script is intentionally an outer runner. It creates an immutable plan first,
then optionally executes DISCO inference, scores the exact final saved
sequences with ESMFold, and writes validation/report artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.disco_analysis import analyze_disco_dump_rows
from disco.model.cogen_inference.guidance.scoring import DEFAULT_MOTIFS, HardMotifScorer, parse_motifs
from disco.model.cogen_inference.guidance.trajectory import write_trajectory_outputs


PRIORITY_MOTIFS = [
    "KTTKS",
    "DRLDR",
    "DRLER",
    "GSPLD",
    "PTSLA",
    "STTLE",
    "AHK",
    "GQP",
    "GHK",
    "KVK",
    "VGP",
    "VW",
    "GP",
    "DR",
    "DK",
]

PG_REFERENCE = {
    "pg_evo_gen1": {
        "n": 20,
        "unique_motifs_mean": 8.15,
        "unique_motifs_median": 8.00,
        "unique_motifs_max": 10,
        "total_motifs_mean": 13.60,
        "total_motifs_median": 14.00,
        "total_motifs_max": 16,
        "pg_plddt_mean": 0.911,
        "pg_plddt_median": 0.948,
        "mean_external_esmfold_plddt": 0.730,
        "median_external_esmfold_plddt": 0.748,
        "min_external_esmfold_plddt": 0.515,
        "max_external_esmfold_plddt": 0.866,
        "note": "Template-overnahme/evolution; nicht komplett unconditional de novo.",
    },
    "pg_non_evo": {
        "n": 20,
        "unique_motifs_mean": 4.15,
        "unique_motifs_median": 4.00,
        "unique_motifs_max": 5,
        "total_motifs_mean": 5.25,
        "total_motifs_median": 5.00,
        "total_motifs_max": 8,
        "pg_plddt_mean": 0.943,
        "pg_plddt_median": 0.948,
        "mean_external_esmfold_plddt": 0.775,
        "median_external_esmfold_plddt": 0.775,
        "min_external_esmfold_plddt": 0.602,
        "max_external_esmfold_plddt": 0.915,
        "note": "Bester fairer de-novo-naeherer PG-Kontrastpunkt.",
    },
    "pg_evo_gen0": {
        "n": 10,
        "unique_motifs_mean": 5.50,
        "unique_motifs_median": 5.00,
        "unique_motifs_max": 7,
        "total_motifs_mean": 9.30,
        "total_motifs_median": 9.50,
        "total_motifs_max": 12,
        "pg_plddt_mean": 0.971,
        "pg_plddt_median": 0.972,
        "mean_external_esmfold_plddt": 0.803,
        "median_external_esmfold_plddt": 0.794,
        "min_external_esmfold_plddt": 0.720,
        "max_external_esmfold_plddt": 0.883,
        "note": "Evo-Ausgangskontrolle vor Template-Weitergabe.",
    },
}


@dataclass(frozen=True)
class PlannedRun:
    variant: str
    guidance_mode: str
    target_count: int
    seed: int
    seed_index: int
    placement_mode: str
    placement_strategy: str
    position_offset: int
    motifs: tuple[str, ...]
    starts_1based: tuple[int, ...]
    run_dir: Path
    dump_dir: Path
    input_json_path: Path
    command: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(-?\d+)-(-?\d+)", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(part))
    if not result:
        raise ValueError("Expected at least one integer.")
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def json_dumps(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_length_input_json(path: Path, length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "name": f"length_{length}",
            "sequences": [{"proteinChain": {"sequence": "-" * length, "count": 1}}],
        }
    ]
    path.write_text(json_dumps(records))


def motif_priority_order(motifs: Iterable[str]) -> list[str]:
    motif_set = list(dict.fromkeys(parse_motifs(motifs)))
    ordered = [motif for motif in PRIORITY_MOTIFS if motif in motif_set]
    ordered.extend(motif for motif in motif_set if motif not in ordered)
    return ordered


def choose_motif_subset(motifs: list[str], target_count: int, seed_index: int) -> tuple[str, ...]:
    if target_count < 1 or target_count > len(motifs):
        raise ValueError(f"target_count must be in 1..{len(motifs)}")
    ordered = motif_priority_order(motifs)
    if target_count == len(ordered):
        return tuple(ordered)
    offset = seed_index % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    return tuple(rotated[:target_count])


def ideal_hard_commit_starts_1based(
    motifs: Iterable[str],
    length: int,
    position_offset: int = 0,
) -> tuple[int, ...]:
    """Mirror the controller's spread placement for an empty L-length sequence."""

    motifs = list(motifs)
    claimed: set[int] = set()
    starts: list[int] = []
    for motif_index, motif in enumerate(motifs):
        motif_length = len(motif)
        target_center = (motif_index + 1) * length / (len(motifs) + 1)
        target_start = int(round(target_center - motif_length / 2))
        target_start += position_offset
        target_start = max(0, min(target_start, length - motif_length))
        candidates = sorted(
            range(0, length - motif_length + 1),
            key=lambda start: (abs(start - target_start), start),
        )
        chosen = None
        for start in candidates:
            positions = set(range(start, start + motif_length))
            if positions.isdisjoint(claimed):
                chosen = start
                claimed.update(positions)
                break
        if chosen is None:
            raise ValueError(f"Could not place motif {motif!r} without overlap.")
        starts.append(chosen + 1)
    return tuple(starts)


def hydra_list(values: Iterable[object]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def trajectory_overrides(*, save_trajectories: bool, trajectory_dtype: str) -> list[str]:
    if not save_trajectories:
        return []
    return [
        "cogen_guidance.trajectory_dump.enabled=true",
        "cogen_guidance.trajectory_dump.save_x1_denoised=true",
        "cogen_guidance.trajectory_dump.save_xt_trajectory=true",
        "cogen_guidance.trajectory_dump.save_final_coordinate=true",
        f"cogen_guidance.trajectory_dump.dtype={trajectory_dtype}",
        "cogen_guidance.trajectory_dump.max_steps=0",
    ]


def build_inference_command(
    *,
    python_executable: str,
    disco_root: Path,
    experiment: str,
    input_json_path: Path,
    dump_dir: Path,
    seed: int,
    motifs: tuple[str, ...],
    length: int,
    n_step: int,
    n_cycle: int,
    position_offset: int = 0,
    save_trajectories: bool = True,
    trajectory_dtype: str = "float32",
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(disco_root / "runner" / "inference.py"),
        f"experiment={experiment}",
        f"input_json_path={input_json_path}",
        f"dump_dir={dump_dir}",
        f"seeds=[{seed}]",
        f"sample_diffusion.N_step={n_step}",
        f"model.N_cycle={n_cycle}",
        "cogen_guidance.enabled=true",
        f"cogen_guidance.motifs={hydra_list(motifs)}",
        "cogen_guidance.logit_guidance.enabled=false",
        "cogen_guidance.logit_guidance.gradient_enabled=false",
        "cogen_guidance.logit_guidance.local_rescue_enabled=false",
        "cogen_guidance.logit_guidance.hard_commit_enabled=true",
        "cogen_guidance.logit_guidance.hard_commit_spread=true",
        f"cogen_guidance.logit_guidance.hard_commit_position_offset={position_offset}",
        "cogen_guidance.logit_guidance.hard_commit_all_motifs=true",
        f"cogen_guidance.logit_guidance.hard_commit_max_motifs_per_step={len(motifs)}",
        f"cogen_guidance.logit_guidance.hard_commit_max_changes_per_step={length}",
        "cogen_guidance.scoring.motif_all_weight=1.0",
        "cogen_guidance.scoring.motif_many_weight=0.0",
        "cogen_guidance.save_best.enabled=true",
        "cogen_guidance.save_best.select_best_output=true",
        "cogen_guidance.save_best.motif_score_weight=1.0",
        "cogen_guidance.save_best.confidence_weight=0.0",
    ]
    command.extend(
        trajectory_overrides(
            save_trajectories=save_trajectories,
            trajectory_dtype=trajectory_dtype,
        )
    )
    return tuple(command)


def build_soft_aggressive_command(
    *,
    python_executable: str,
    disco_root: Path,
    experiment: str,
    input_json_path: Path,
    dump_dir: Path,
    seed: int,
    motifs: tuple[str, ...],
    length: int,
    n_step: int,
    n_cycle: int,
    gradient_scale: float,
    local_rescue_bonus: float,
    motif_temperature: float,
    save_trajectories: bool = True,
    trajectory_dtype: str = "float32",
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(disco_root / "runner" / "inference.py"),
        f"experiment={experiment}",
        f"input_json_path={input_json_path}",
        f"dump_dir={dump_dir}",
        f"seeds=[{seed}]",
        f"sample_diffusion.N_step={n_step}",
        f"model.N_cycle={n_cycle}",
        "cogen_guidance.enabled=true",
        f"cogen_guidance.motifs={hydra_list(motifs)}",
        "cogen_guidance.logit_guidance.enabled=true",
        "cogen_guidance.logit_guidance.gradient_enabled=true",
        f"cogen_guidance.logit_guidance.gradient_scale={gradient_scale}",
        "cogen_guidance.logit_guidance.motif_weight_all=1.0",
        "cogen_guidance.logit_guidance.motif_weight_many=0.0",
        f"cogen_guidance.logit_guidance.motif_top_k={len(motifs)}",
        f"cogen_guidance.logit_guidance.motif_temperature={motif_temperature}",
        "cogen_guidance.logit_guidance.local_rescue_enabled=true",
        f"cogen_guidance.logit_guidance.local_rescue_bonus={local_rescue_bonus}",
        f"cogen_guidance.logit_guidance.local_rescue_top_k={len(motifs)}",
        "cogen_guidance.logit_guidance.local_rescue_min_probability=0.0",
        f"cogen_guidance.logit_guidance.local_rescue_max_motifs_per_step={len(motifs)}",
        f"cogen_guidance.logit_guidance.local_rescue_max_changes_per_step={length}",
        "cogen_guidance.logit_guidance.local_rescue_only_missing=true",
        "cogen_guidance.logit_guidance.hard_commit_enabled=false",
        "cogen_guidance.logit_guidance.hard_commit_all_motifs=false",
        "cogen_guidance.scoring.motif_all_weight=1.0",
        "cogen_guidance.scoring.motif_many_weight=0.0",
        "cogen_guidance.save_best.enabled=true",
        "cogen_guidance.save_best.select_best_output=true",
        "cogen_guidance.save_best.motif_score_weight=1.0",
        "cogen_guidance.save_best.confidence_weight=0.0",
    ]
    command.extend(
        trajectory_overrides(
            save_trajectories=save_trajectories,
            trajectory_dtype=trajectory_dtype,
        )
    )
    return tuple(command)


def parse_shift_offsets(value: str) -> list[int]:
    offsets = parse_int_list(value)
    if 0 not in offsets:
        offsets.insert(0, 0)
    return offsets


def placement_offset_for_seed(args: argparse.Namespace, seed_index: int) -> tuple[str, int]:
    if args.placement_mode == "even":
        return "even", 0
    if args.placement_mode == "even_shifted":
        offsets = parse_shift_offsets(args.shift_offsets)
        offset = offsets[seed_index % len(offsets)]
        return ("even" if offset == 0 else "shifted", offset)
    raise ValueError(f"Unsupported placement mode: {args.placement_mode}")


def plan_runs(args: argparse.Namespace) -> list[PlannedRun]:
    motifs = parse_motifs(args.motifs)
    target_counts = parse_int_list(args.target_counts)
    seeds = parse_int_list(args.seeds)
    input_json_path = args.out / "inputs" / f"unconditional_l{args.length}.json"
    write_length_input_json(input_json_path, args.length)
    runs: list[PlannedRun] = []
    for target_count in target_counts:
        if target_count > len(motifs):
            raise ValueError(f"target_count={target_count} exceeds {len(motifs)} motifs")
        variant = f"hardcommit_target{target_count:02d}"
        for seed_index, seed in enumerate(seeds):
            subset = choose_motif_subset(motifs, target_count, seed_index)
            placement_strategy, position_offset = placement_offset_for_seed(args, seed_index)
            starts = ideal_hard_commit_starts_1based(
                subset,
                args.length,
                position_offset=position_offset,
            )
            run_dir = args.out / "runs" / variant / f"seed_{seed:03d}"
            dump_dir = run_dir / "disco_output"
            command = build_inference_command(
                python_executable=args.python_executable,
                disco_root=args.disco_root,
                experiment=args.experiment,
                input_json_path=input_json_path.resolve(),
                dump_dir=dump_dir.resolve(),
                seed=seed,
                motifs=subset,
                length=args.length,
                n_step=args.n_step,
                n_cycle=args.n_cycle,
                position_offset=position_offset,
                save_trajectories=not args.no_trajectory_dump,
                trajectory_dtype=args.trajectory_dtype,
            )
            runs.append(
                PlannedRun(
                    variant=variant,
                    guidance_mode="hard_commit",
                    target_count=target_count,
                    seed=seed,
                    seed_index=seed_index,
                    placement_mode=args.placement_mode,
                    placement_strategy=placement_strategy,
                    position_offset=position_offset,
                    motifs=subset,
                    starts_1based=starts,
                    run_dir=run_dir,
                    dump_dir=dump_dir,
                    input_json_path=input_json_path,
                    command=command,
                )
            )
    if args.include_soft_aggressive:
        soft_target_count = int(args.soft_target_count)
        if soft_target_count > len(motifs):
            raise ValueError(
                f"soft_target_count={soft_target_count} exceeds {len(motifs)} motifs"
            )
        variant = f"soft_aggressive_target{soft_target_count:02d}"
        for seed_index, seed in enumerate(parse_int_list(args.soft_seeds)):
            subset = choose_motif_subset(motifs, soft_target_count, seed_index)
            run_dir = args.out / "runs" / variant / f"seed_{seed:03d}"
            dump_dir = run_dir / "disco_output"
            command = build_soft_aggressive_command(
                python_executable=args.python_executable,
                disco_root=args.disco_root,
                experiment=args.experiment,
                input_json_path=input_json_path.resolve(),
                dump_dir=dump_dir.resolve(),
                seed=seed,
                motifs=subset,
                length=args.length,
                n_step=args.n_step,
                n_cycle=args.n_cycle,
                gradient_scale=args.soft_gradient_scale,
                local_rescue_bonus=args.soft_local_rescue_bonus,
                motif_temperature=args.soft_motif_temperature,
                save_trajectories=not args.no_trajectory_dump,
                trajectory_dtype=args.trajectory_dtype,
            )
            runs.append(
                PlannedRun(
                    variant=variant,
                    guidance_mode="soft_aggressive",
                    target_count=soft_target_count,
                    seed=seed,
                    seed_index=seed_index,
                    placement_mode="none",
                    placement_strategy="none",
                    position_offset=0,
                    motifs=subset,
                    starts_1based=(),
                    run_dir=run_dir,
                    dump_dir=dump_dir,
                    input_json_path=input_json_path,
                    command=command,
                )
            )
    return runs


def planned_run_row(run: PlannedRun) -> dict[str, object]:
    planned_pairs = (
        ";".join(
            f"{motif}:{start}"
            for motif, start in zip(run.motifs, run.starts_1based, strict=True)
        )
        if run.starts_1based
        else ""
    )
    return {
        "variant": run.variant,
        "guidance_mode": run.guidance_mode,
        "target_count": run.target_count,
        "seed": run.seed,
        "seed_index": run.seed_index,
        "placement_mode": run.placement_mode,
        "placement_strategy": run.placement_strategy,
        "position_offset": run.position_offset,
        "motifs": ",".join(run.motifs),
        "planned_starts_1based": ",".join(str(start) for start in run.starts_1based),
        "planned_motif_position_pairs": planned_pairs,
        "run_dir": str(run.run_dir.resolve()),
        "dump_dir": str(run.dump_dir.resolve()),
        "input_json_path": str(run.input_json_path.resolve()),
        "command": " ".join(run.command),
    }


def write_plan_artifacts(args: argparse.Namespace, runs: list[PlannedRun]) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [planned_run_row(run) for run in runs]
    write_csv(args.out / "planned_runs.csv", rows)
    for run in runs:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        (run.run_dir / "command.txt").write_text(" ".join(run.command) + "\n")
        (run.run_dir / "run_manifest.json").write_text(
            json_dumps(
                {
                    "status": "planned",
                    "variant": run.variant,
                    "guidance_mode": run.guidance_mode,
                    "target_count": run.target_count,
                    "seed": run.seed,
                    "placement_mode": run.placement_mode,
                    "placement_strategy": run.placement_strategy,
                    "position_offset": run.position_offset,
                    "motifs": list(run.motifs),
                    "planned_starts_1based": list(run.starts_1based),
                    "run_dir": str(run.run_dir.resolve()),
                    "dump_dir": str(run.dump_dir.resolve()),
                    "command": list(run.command),
                    "created_at": utc_now(),
                }
            )
        )
    manifest = {
        "status": "dry_run" if args.dry_run else "planned",
        "script": "runner/guidance_codeocean_hardcommit_sweep.py",
        "created_at": utc_now(),
        "out": str(args.out.resolve()),
        "disco_root": str(args.disco_root.resolve()),
        "experiment": args.experiment,
        "length": args.length,
        "n_step": args.n_step,
        "n_cycle": args.n_cycle,
        "seeds": parse_int_list(args.seeds),
        "target_counts": parse_int_list(args.target_counts),
        "soft_aggressive": {
            "enabled": bool(args.include_soft_aggressive),
            "variant": f"soft_aggressive_target{int(args.soft_target_count):02d}",
            "seeds": parse_int_list(args.soft_seeds),
            "target_count": int(args.soft_target_count),
            "hard_commit_enabled": False,
            "logit_guidance_enabled": True,
            "gradient_enabled": True,
            "gradient_scale": args.soft_gradient_scale,
            "local_rescue_enabled": True,
            "local_rescue_bonus": args.soft_local_rescue_bonus,
            "motif_temperature": args.soft_motif_temperature,
            "interpretation": (
                "Exploratory control: tries to reach many motifs without hard "
                "sequence rewriting. Missing planned motifs are reported, not "
                "treated as validation failure."
            ),
        },
        "motifs_all15": parse_motifs(args.motifs),
        "motif_selection_rule": (
            "Deterministic rotating priority order. Long/rare motifs are first in "
            "the priority list, and seed_index rotates the list so every motif is "
            "covered across repeated seeds."
        ),
        "hard_commit_settings": {
            "hard_commit_enabled": True,
            "hard_commit_spread": True,
            "hard_commit_all_motifs": True,
            "hard_commit_max_motifs_per_step": "target_count",
            "hard_commit_max_changes_per_step": args.length,
            "hard_commit_position_offset": "0 for even, deterministic seed-level offset for shifted placements",
            "gradient_guidance_enabled": False,
            "local_rescue_enabled": False,
            "best_selection": "cogen_guidance.save_best motif_score_weight=1.0 confidence_weight=0.0",
        },
        "trajectory_persistence": {
            "enabled": not args.no_trajectory_dump,
            "saved_raw_tensors": ["x1_denoised", "xt_trajectory", "coordinate"],
            "max_steps": 0,
            "max_steps_meaning": "0 means no truncation; all returned timesteps are saved.",
            "dtype": args.trajectory_dtype,
            "lightweight_step_diagnostics": "DISCO guidance diagnostics JSONL/CSV are also written per sample when guidance is active.",
        },
        "external_esmfold": {
            "enabled": not args.skip_esmfold,
            "num_recycles": args.num_recycles,
            "chunk_size": args.chunk_size,
            "script": str(args.esmfold_script),
            "score_target": "exact final saved DISCO sequence from final_disco_designs.csv",
        },
        "pg_reference_external_esmfold": PG_REFERENCE,
        "planned_run_count": len(runs),
        "placement_mode": args.placement_mode,
        "shift_offsets": parse_shift_offsets(args.shift_offsets),
    }
    (args.out / "run_manifest.json").write_text(json_dumps(manifest))
    write_run_readme(args, runs)


def write_run_readme(args: argparse.Namespace, runs: list[PlannedRun]) -> None:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.guidance_mode] = counts.get(run.guidance_mode, 0) + 1
    command = (
        "python runner/guidance_codeocean_hardcommit_sweep.py "
        f"--out {args.out}"
    )
    lines = [
        "# CodeOcean DISCO 15-Motif Run",
        "",
        "## Execute",
        "",
        "Run from the DISCO repository root:",
        "",
        "```bash",
        command,
        "```",
        "",
        "## What Runs",
        "",
        f"- Total planned designs: {len(runs)}",
        f"- Hard-commit designs: {counts.get('hard_commit', 0)}",
        f"- Soft-aggressive exploratory designs: {counts.get('soft_aggressive', 0)}",
        f"- Length: {args.length}",
        f"- Diffusion steps: {args.n_step}",
        f"- n_cycle: {args.n_cycle}",
        f"- Hard target counts: {args.target_counts}",
        f"- Hard seeds: {args.seeds}",
        f"- Placement mode: {args.placement_mode}",
        f"- Soft target count: {args.soft_target_count}",
        f"- Soft seeds: {args.soft_seeds}",
        f"- Soft gradient scale: {args.soft_gradient_scale}",
        f"- Soft local rescue bonus: {args.soft_local_rescue_bonus}",
        "",
        "The hard-commit block is the main experiment. The soft-aggressive block is an exploratory control that tries to reach many motifs without hard sequence rewriting.",
        "",
        "## Outputs",
        "",
        "- `planned_runs.csv`: exact command per design.",
        "- `run_manifest.json`: immutable run configuration.",
        "- `final_disco_designs.csv`: final saved sequences and motif scores.",
        "- `normalized_designs.csv`: exact ESMFold input sequences.",
        "- `esmfold_plddt_scoring/`: external ESMFold pLDDT outputs and PDBs.",
        "- `final_scored_designs.csv`: joined final table.",
        "- `trajectory_summary.csv/json`: per-step diagnostic summaries.",
        "- `runs/*/*/disco_output/trajectories/*.pt`: full timestep trajectory tensors.",
        "- `strict_validation.json`: validation of sequence/scoring/trajectory consistency.",
        "- `CODEOCEAN_DISCO_15MOTIF_REPORT.md`: final report.",
        "- `results/tables/*.csv`: generated analysis tables.",
        "- `results/figures/*.png`: generated analysis diagrams.",
    ]
    (args.out / "README_CODEOCEAN_RUN.md").write_text("\n".join(lines) + "\n")


def run_one(run: PlannedRun) -> dict[str, object]:
    manifest_path = run.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"status": "running", "started_at": utc_now()})
    manifest_path.write_text(json_dumps(manifest))
    start = time.time()
    log_path = run.run_dir / "run.log"
    try:
        with log_path.open("w") as log_handle:
            subprocess.run(
                list(run.command),
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        status = "ok"
        error = ""
    except subprocess.CalledProcessError as exc:
        status = "error"
        error = f"CalledProcessError exit_code={exc.returncode}"
    runtime = time.time() - start
    manifest.update({"status": status, "finished_at": utc_now(), "runtime_seconds": runtime, "error": error})
    manifest_path.write_text(json_dumps(manifest))
    return {
        "variant": run.variant,
        "guidance_mode": run.guidance_mode,
        "seed": run.seed,
        "status": status,
        "runtime_seconds": runtime,
        "error": error,
    }


def execute_runs(args: argparse.Namespace, runs: list[PlannedRun]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if args.max_parallel <= 1:
        for index, run in enumerate(runs, start=1):
            print(f"[{index}/{len(runs)}] run {run.variant} seed={run.seed}", flush=True)
            results.append(run_one(run))
        return results
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(run_one, run): run for run in runs}
        for future in as_completed(futures):
            run = futures[future]
            print(f"finished {run.variant} seed={run.seed}", flush=True)
            results.append(future.result())
    return sorted(results, key=lambda row: (str(row["variant"]), int(row["seed"])))


def analyze_run(run: PlannedRun, all_motifs: list[str], length: int) -> list[dict[str, object]]:
    rows = analyze_disco_dump_rows(
        dump_dir=run.dump_dir,
        motifs=run.motifs,
        motif_all_weight=1.0,
        motif_many_weight=0.0,
    )
    all_scorer = HardMotifScorer(motifs=all_motifs, motif_all_weight=1.0, motif_many_weight=0.0)
    final_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        sequence = str(row["sequence"]).upper()
        all_score = all_scorer.score_sequence(sequence)
        design_id = f"{run.variant}_seed{run.seed:03d}_sample{row_index:02d}"
        final_rows.append(
            {
                "system": "DISCO",
                "mode": "codeocean_15motif_sweep",
                "variant": run.variant,
                "guidance_mode": run.guidance_mode,
                "target_count": run.target_count,
                "seed": run.seed,
                "design_id": design_id,
                "placement_mode": run.placement_mode,
                "placement_strategy": run.placement_strategy,
                "position_offset": run.position_offset,
                "length": len(sequence),
                "expected_length": length,
                "sequence": sequence,
                "planned_motifs": ",".join(run.motifs),
                "planned_starts_1based": ",".join(str(start) for start in run.starts_1based),
                "planned_unique_motifs": int(row["unique_motifs"]),
                "planned_total_motifs": int(row["total_motifs"]),
                "planned_present_motifs": row["present_motifs"],
                "planned_missing_motifs": row["missing_motifs"],
                "all15_unique_motifs": all_score.num_unique,
                "all15_total_motifs": all_score.num_total,
                "all15_present_motifs": ",".join(all_score.present),
                "all15_missing_motifs": ",".join(all_score.missing),
                "confidence": row["confidence"],
                "confidence_metric": row["confidence_metric"],
                "sequence_path": row["sequence_path"],
                "structure_path": row.get("structure_path", ""),
                "confidence_path": row["confidence_path"],
                "confidence_match_status": row["confidence_match_status"],
                "structure_match_status": row.get("structure_match_status", ""),
                "diagnostic_best_step": row.get("diagnostic_best_step", ""),
                "diagnostic_best_score": row.get("diagnostic_best_score", ""),
                "diagnostic_best_output_selected": row.get("diagnostic_best_output_selected", ""),
                "trajectory_manifest": row.get("trajectory_manifest", ""),
                "trajectory_num_saved_tensors": row.get("trajectory_num_saved_tensors", ""),
                "trajectory_tensor_paths": row.get("trajectory_tensor_paths", ""),
                "trajectory_truncated_keys": row.get("trajectory_truncated_keys", ""),
                "dump_dir": str(run.dump_dir.resolve()),
                "run_dir": str(run.run_dir.resolve()),
            }
        )
    return final_rows


def write_final_disco_designs(args: argparse.Namespace, runs: list[PlannedRun]) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    all_motifs = parse_motifs(args.motifs)
    for run in runs:
        if not (run.dump_dir / "sequences").exists():
            continue
        all_rows.extend(analyze_run(run, all_motifs, args.length))
    write_csv(args.out / "final_disco_designs.csv", all_rows)
    normalized_rows = [
        {
            "system": "DISCO",
            "mode": "codeocean_15motif_sweep",
            "variant": row["variant"],
            "design_id": row["design_id"],
            "length": row["length"],
            "sequence": row["sequence"],
            "source_path": row["sequence_path"],
            "confidence": row["confidence"],
            "confidence_metric": row["confidence_metric"],
        }
        for row in all_rows
    ]
    write_csv(args.out / "normalized_designs.csv", normalized_rows)
    return all_rows


def run_esmfold(args: argparse.Namespace) -> None:
    scoring_dir = args.out / "esmfold_plddt_scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_executable,
        str(args.esmfold_script),
        "--normalized-csv",
        str((args.out / "normalized_designs.csv").resolve()),
        "--out-dir",
        str(scoring_dir.resolve()),
        "--num-recycles",
        str(args.num_recycles),
        "--chunk-size",
        str(args.chunk_size),
    ]
    (scoring_dir / "command.txt").write_text(" ".join(command) + "\n")
    with (scoring_dir / "run.log").open("w") as log_handle:
        subprocess.run(command, cwd=args.disco_root, stdout=log_handle, stderr=subprocess.STDOUT, check=True)


def merge_scoring(args: argparse.Namespace, final_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    score_path = args.out / "esmfold_plddt_scoring" / "esmfold_plddt_scores.csv"
    if not score_path.exists():
        write_csv(args.out / "final_scored_designs.csv", final_rows)
        return final_rows
    score_rows = read_csv(score_path)
    scores_by_design_id = {row["design_id"]: row for row in score_rows}
    merged: list[dict[str, object]] = []
    for row in final_rows:
        score = scores_by_design_id.get(str(row["design_id"]), {})
        merged_row = dict(row)
        merged_row.update(
            {
                "esmfold_status": score.get("status", ""),
                "esmfold_error": score.get("error", ""),
                "esmfold_input_sequence": score.get("esmfold_input_sequence", ""),
                "esmfold_pdb_path": score.get("esmfold_pdb_path", ""),
                "esmfold_num_ca": score.get("esmfold_num_ca", ""),
                "esmfold_plddt_mean_0_1": score.get("esmfold_ca_plddt_mean_0_1", ""),
                "esmfold_plddt_mean_0_100": score.get("esmfold_ca_plddt_mean_0_100", ""),
                "esmfold_runtime_seconds": score.get("runtime_seconds", ""),
            }
        )
        merged.append(merged_row)
    write_csv(args.out / "final_scored_designs.csv", merged)
    return merged


def write_trajectory_summary(args: argparse.Namespace, runs: list[PlannedRun]) -> dict[str, object]:
    diagnostics_paths: list[Path] = []
    sample_ids: list[str] = []
    for run in runs:
        diagnostics_dir = run.dump_dir / "diagnostics"
        for path in sorted(diagnostics_dir.glob("*_guidance_diagnostics.jsonl")):
            diagnostics_paths.append(path.resolve())
            sample_ids.append(f"{run.variant}_seed{run.seed:03d}_{path.stem}")
    if not diagnostics_paths:
        summary = {
            "status": "missing",
            "reason": "No *_guidance_diagnostics.jsonl files found.",
            "trajectory_csv": str((args.out / "trajectory_summary.csv").resolve()),
            "summary_json": str((args.out / "trajectory_summary.json").resolve()),
            "aggregate": {
                "num_samples": 0,
                "num_ok_samples": 0,
                "num_empty_samples": 0,
                "total_trajectory_steps": 0,
                "total_scored_steps": 0,
                "samples_where_best_beats_final": 0,
            },
        }
        (args.out / "trajectory_summary.json").write_text(json_dumps(summary))
        write_csv(args.out / "trajectory_summary.csv", [])
        return summary
    summary = write_trajectory_outputs(
        diagnostics_jsonl_paths=diagnostics_paths,
        out_csv=(args.out / "trajectory_summary.csv").resolve(),
        out_summary_json=(args.out / "trajectory_summary.json").resolve(),
        sample_ids=sample_ids,
    )
    summary["status"] = "ok"
    return summary


def validate_outputs(args: argparse.Namespace, runs: list[PlannedRun], rows: list[dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        design_id = str(row.get("design_id", ""))
        if design_id in seen_ids:
            errors.append(f"duplicate design_id: {design_id}")
        seen_ids.add(design_id)
        if int(row.get("length", -1)) != args.length:
            errors.append(f"{design_id}: length {row.get('length')} != {args.length}")
        if not row.get("sequence"):
            errors.append(f"{design_id}: empty sequence")
        if (
            row.get("guidance_mode") != "soft_aggressive"
            and int(row.get("planned_unique_motifs", -1)) < int(row.get("target_count", 0))
        ):
            errors.append(f"{design_id}: missing planned motifs {row.get('planned_missing_motifs')}")
        if not args.no_trajectory_dump:
            manifest_path = str(row.get("trajectory_manifest", ""))
            if not manifest_path:
                errors.append(f"{design_id}: missing trajectory manifest path")
            elif not Path(manifest_path).exists():
                errors.append(f"{design_id}: trajectory manifest does not exist: {manifest_path}")
            else:
                trajectory_manifest = json.loads(Path(manifest_path).read_text())
                saved_keys = {
                    item.get("key")
                    for item in trajectory_manifest.get("saved_tensors", [])
                }
                for required_key in ["x1_denoised", "xt_trajectory", "coordinate"]:
                    if required_key not in saved_keys:
                        errors.append(f"{design_id}: missing trajectory tensor {required_key}")
                if trajectory_manifest.get("truncated_keys"):
                    errors.append(
                        f"{design_id}: trajectory tensors were truncated: {trajectory_manifest.get('truncated_keys')}"
                    )
                if str(trajectory_manifest.get("max_steps", "")) not in {"", "0"}:
                    errors.append(f"{design_id}: trajectory max_steps is not unbounded")
        if row.get("esmfold_status"):
            if row.get("esmfold_status") != "ok":
                errors.append(f"{design_id}: ESMFold status {row.get('esmfold_status')} {row.get('esmfold_error')}")
            if row.get("esmfold_input_sequence") != row.get("sequence"):
                errors.append(f"{design_id}: ESMFold input sequence differs from final sequence")
            if str(row.get("esmfold_num_ca")) != str(row.get("length")):
                errors.append(f"{design_id}: ESMFold CA count {row.get('esmfold_num_ca')} != length {row.get('length')}")
            pdb_path = str(row.get("esmfold_pdb_path", ""))
            if pdb_path and not Path(pdb_path).exists():
                errors.append(f"{design_id}: missing ESMFold PDB {pdb_path}")
    expected_rows = len(runs)
    if not args.dry_run and len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} final designs, found {len(rows)}")
    if args.dry_run:
        warnings.append("dry_run: inference and ESMFold validation intentionally not executed")
    validation = {
        "status": "ok" if not errors else "error",
        "created_at": utc_now(),
        "expected_designs": expected_rows,
        "observed_designs": len(rows),
        "errors": errors,
        "warnings": warnings,
    }
    (args.out / "strict_validation.json").write_text(json_dumps(validation))
    return validation


def group_stats(rows: list[dict[str, object]], value_key: str) -> dict[str, dict[str, float | int | str]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(value_key, "")
        if value in {"", None}:
            continue
        grouped.setdefault(str(row["variant"]), []).append(float(value))
    stats: dict[str, dict[str, float | int | str]] = {}
    for variant, values in grouped.items():
        stats[variant] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return stats


def generate_results_analysis(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    """Write reader-facing summaries and plots into `results/`."""

    results_dir = (args.out / "results").resolve()
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "no_data",
        "created_at": utc_now(),
        "results_dir": str(results_dir.resolve()),
        "figures": [],
        "tables": [],
        "notes": [],
    }

    if not rows:
        manifest["notes"].append("No final scored designs available yet. Dry-run produced plan artifacts only.")
        (results_dir / "analysis_manifest.json").write_text(json_dumps(manifest))
        (results_dir / "RESULTS_ANALYSIS.md").write_text(
            "# CodeOcean DISCO Results Analysis\n\n"
            "No final scored designs are available yet. This is expected for a dry-run plan.\n"
        )
        return manifest

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows).copy()
    expected_columns = [
        "variant",
        "target_count",
        "seed",
        "position_offset",
        "placement_strategy",
        "guidance_mode",
        "design_id",
        "planned_unique_motifs",
        "planned_total_motifs",
        "all15_unique_motifs",
        "all15_total_motifs",
        "all15_present_motifs",
        "confidence",
        "esmfold_plddt_mean_0_1",
        "esmfold_runtime_seconds",
    ]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA
    numeric_columns = [
        "target_count",
        "seed",
        "position_offset",
        "length",
        "planned_unique_motifs",
        "planned_total_motifs",
        "all15_unique_motifs",
        "all15_total_motifs",
        "confidence",
        "esmfold_plddt_mean_0_1",
        "esmfold_runtime_seconds",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    grouped = (
        df.groupby(["guidance_mode", "variant", "target_count"], dropna=False)
        .agg(
            n=("design_id", "count"),
            all15_unique_mean=("all15_unique_motifs", "mean"),
            all15_unique_median=("all15_unique_motifs", "median"),
            all15_unique_min=("all15_unique_motifs", "min"),
            all15_unique_max=("all15_unique_motifs", "max"),
            all15_total_mean=("all15_total_motifs", "mean"),
            planned_unique_mean=("planned_unique_motifs", "mean"),
            esmfold_plddt_mean=("esmfold_plddt_mean_0_1", "mean"),
            esmfold_plddt_median=("esmfold_plddt_mean_0_1", "median"),
            esmfold_plddt_min=("esmfold_plddt_mean_0_1", "min"),
            esmfold_plddt_max=("esmfold_plddt_mean_0_1", "max"),
            confidence_mean=("confidence", "mean"),
        )
        .reset_index()
    )
    grouped_path = tables_dir / "summary_by_target_count.csv"
    grouped.to_csv(grouped_path, index=False)
    manifest["tables"].append(str(grouped_path.resolve()))

    sort_columns = [column for column in ["esmfold_plddt_mean_0_1", "all15_unique_motifs", "all15_total_motifs"] if column in df.columns]
    top_df = df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(25) if sort_columns else df.head(25)
    top_path = tables_dir / "top_designs_by_esmfold_plddt.csv"
    top_df.to_csv(top_path, index=False)
    manifest["tables"].append(str(top_path.resolve()))

    motif_rows: list[dict[str, object]] = []
    all_motifs = parse_motifs(args.motifs)
    for motif in all_motifs:
        motif_rows.append(
            {
                "motif": motif,
                "designs_present": int(df["all15_present_motifs"].fillna("").str.split(",").apply(lambda items: motif in items).sum()),
                "design_fraction": float(df["all15_present_motifs"].fillna("").str.split(",").apply(lambda items: motif in items).mean()),
            }
        )
    motif_df = pd.DataFrame(motif_rows)
    motif_path = tables_dir / "motif_frequency.csv"
    motif_df.to_csv(motif_path, index=False)
    manifest["tables"].append(str(motif_path.resolve()))

    placement_path = tables_dir / "summary_by_placement.csv"
    placement_grouped = (
        df.groupby(["placement_strategy", "position_offset"], dropna=False)
        .agg(
            n=("design_id", "count"),
            all15_unique_mean=("all15_unique_motifs", "mean"),
            esmfold_plddt_mean=("esmfold_plddt_mean_0_1", "mean"),
            esmfold_plddt_max=("esmfold_plddt_mean_0_1", "max"),
        )
        .reset_index()
    )
    placement_grouped.to_csv(placement_path, index=False)
    manifest["tables"].append(str(placement_path.resolve()))

    guidance_path = tables_dir / "summary_by_guidance_mode.csv"
    guidance_grouped = (
        df.groupby(["guidance_mode"], dropna=False)
        .agg(
            n=("design_id", "count"),
            all15_unique_mean=("all15_unique_motifs", "mean"),
            all15_unique_max=("all15_unique_motifs", "max"),
            all15_total_mean=("all15_total_motifs", "mean"),
            esmfold_plddt_mean=("esmfold_plddt_mean_0_1", "mean"),
            esmfold_plddt_max=("esmfold_plddt_mean_0_1", "max"),
        )
        .reset_index()
    )
    guidance_grouped.to_csv(guidance_path, index=False)
    manifest["tables"].append(str(guidance_path.resolve()))

    def save_current_figure(filename: str) -> None:
        path = figures_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        manifest["figures"].append(str(path.resolve()))

    if df["esmfold_plddt_mean_0_1"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.scatter(
            df["all15_unique_motifs"],
            df["esmfold_plddt_mean_0_1"],
            c=df["target_count"],
            cmap="viridis",
            edgecolors="black",
            linewidths=0.4,
        )
        plt.colorbar(label="target motif count")
        plt.xlabel("unique motifs present from 15-motif list")
        plt.ylabel("external ESMFold pLDDT")
        plt.title("ESMFold pLDDT vs motif coverage")
        save_current_figure("plddt_vs_unique_motifs.png")

        plt.figure(figsize=(8, 5))
        df.boxplot(column="esmfold_plddt_mean_0_1", by="target_count", grid=False)
        plt.suptitle("")
        plt.title("Mean ESMFold pLDDT by target motif count")
        plt.xlabel("target motif count")
        plt.ylabel("mean external ESMFold pLDDT")
        save_current_figure("plddt_by_target_count.png")

        plt.figure(figsize=(8, 5))
        placement_plot = placement_grouped.sort_values("position_offset")
        plt.scatter(
            placement_plot["position_offset"],
            placement_plot["esmfold_plddt_mean"],
            s=60,
            c=["#1f77b4" if value == "even" else "#ff7f0e" for value in placement_plot["placement_strategy"]],
        )
        plt.axvline(0, color="black", linewidth=1, alpha=0.4)
        plt.xlabel("hard-commit position offset")
        plt.ylabel("mean external ESMFold pLDDT")
        plt.title("Placement shift vs pLDDT")
        save_current_figure("plddt_by_position_offset.png")

    plt.figure(figsize=(8, 5))
    grouped.sort_values("target_count").plot(
        x="target_count",
        y=["all15_unique_mean", "planned_unique_mean"],
        kind="bar",
        ax=plt.gca(),
    )
    plt.xlabel("target motif count")
    plt.ylabel("mean unique motifs")
    plt.title("Motif coverage by target count")
    plt.legend(["all 15 motif list", "planned subset"])
    save_current_figure("motif_coverage_by_target_count.png")

    plt.figure(figsize=(9, 5))
    motif_df.sort_values("designs_present", ascending=False).plot(
        x="motif",
        y="designs_present",
        kind="bar",
        ax=plt.gca(),
        legend=False,
    )
    plt.xlabel("motif")
    plt.ylabel("designs containing motif")
    plt.title("Motif frequency across final designs")
    save_current_figure("motif_frequency.png")

    trajectory_path = args.out / "trajectory_summary.csv"
    if trajectory_path.exists() and trajectory_path.stat().st_size > 0:
        try:
            traj = pd.read_csv(trajectory_path)
            if not traj.empty and {"step", "unique_motifs"}.issubset(traj.columns):
                traj["step"] = pd.to_numeric(traj["step"], errors="coerce")
                traj["unique_motifs"] = pd.to_numeric(traj["unique_motifs"], errors="coerce")
                by_step = traj.groupby("step", dropna=True).agg(
                    unique_motifs_mean=("unique_motifs", "mean"),
                    total_motifs_mean=("total_motifs", "mean") if "total_motifs" in traj.columns else ("unique_motifs", "mean"),
                ).reset_index()
                by_step_path = tables_dir / "trajectory_motif_progress_by_step.csv"
                by_step.to_csv(by_step_path, index=False)
                manifest["tables"].append(str(by_step_path.resolve()))
                plt.figure(figsize=(8, 5))
                plt.plot(by_step["step"], by_step["unique_motifs_mean"], marker="o", label="unique motifs")
                if "total_motifs_mean" in by_step.columns:
                    plt.plot(by_step["step"], by_step["total_motifs_mean"], marker="o", label="total motifs")
                plt.xlabel("diffusion step")
                plt.ylabel("mean motifs")
                plt.title("Motif progress over diffusion steps")
                plt.legend()
                save_current_figure("trajectory_motif_progress.png")
        except Exception as exc:
            manifest["notes"].append(f"Trajectory plot skipped: {type(exc).__name__}: {exc}")

    summary_md = [
        "# CodeOcean DISCO Results Analysis",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Tables",
        "",
        *[f"- `{Path(path).resolve().relative_to(results_dir)}`" for path in manifest["tables"]],
        "",
        "## Figures",
        "",
        *[f"- `{Path(path).resolve().relative_to(results_dir)}`" for path in manifest["figures"]],
    ]
    if manifest["notes"]:
        summary_md.extend(["", "## Notes", "", *[f"- {note}" for note in manifest["notes"]]])
    (results_dir / "RESULTS_ANALYSIS.md").write_text("\n".join(summary_md) + "\n")
    manifest["status"] = "ok"
    (results_dir / "analysis_manifest.json").write_text(json_dumps(manifest))
    return manifest


def write_report(args: argparse.Namespace, rows: list[dict[str, object]], validation: dict[str, object]) -> None:
    unique_stats = group_stats(rows, "all15_unique_motifs")
    planned_stats = group_stats(rows, "planned_unique_motifs")
    total_stats = group_stats(rows, "all15_total_motifs")
    plddt_stats = group_stats(rows, "esmfold_plddt_mean_0_1")
    lines = [
        "# CodeOcean DISCO 15-Motif Hard-Commit Sweep",
        "",
        f"Created: {utc_now()}",
        "",
        "## Setup",
        "",
        f"- DISCO main mode: hard commit, unconditional L{args.length}, T={args.n_step}, n_cycle={args.n_cycle}",
        f"- Hard-commit target motif counts: {args.target_counts}",
        f"- Seeds: {args.seeds}",
        f"- Placement mode: {args.placement_mode}",
        f"- Soft-aggressive exploratory block: {'enabled' if args.include_soft_aggressive else 'disabled'}",
        f"- Soft-aggressive settings when enabled: target_count={args.soft_target_count}, seeds={args.soft_seeds}, gradient_scale={args.soft_gradient_scale}, local_rescue_bonus={args.soft_local_rescue_bonus}, motif_temperature={args.soft_motif_temperature}",
        "- Motif subsets are selected before scoring by deterministic rotating priority order.",
        "- In `even_shifted`, seed 0 uses even spread and later seeds apply deterministic spread offsets.",
        "- ESMFold pLDDT is computed only from the exact final saved sequence listed in `final_disco_designs.csv`.",
        "- Full trajectory persistence is enabled: raw `x1_denoised`, `xt_trajectory`, and final `coordinate` tensors are saved for every sample without timestep truncation.",
        "",
        "## Validation",
        "",
        f"- Status: {validation['status']}",
        f"- Expected designs: {validation['expected_designs']}",
        f"- Observed designs: {validation['observed_designs']}",
    ]
    if validation["errors"]:
        lines.extend(["", "Errors:", *[f"- {error}" for error in validation["errors"]]])
    if validation["warnings"]:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in validation["warnings"]]])
    lines.extend(["", "## DISCO Results", ""])
    if rows:
        lines.append("| variant | n | guidance mode | planned unique mean | all15 unique mean | all15 total mean | ESMFold pLDDT mean | ESMFold pLDDT max |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        mode_by_variant = {
            str(row["variant"]): str(row.get("guidance_mode", ""))
            for row in rows
        }
        for variant in sorted(unique_stats):
            n = int(unique_stats[variant]["n"])
            planned_mean = planned_stats.get(variant, {}).get("mean", float("nan"))
            unique_mean = unique_stats[variant]["mean"]
            total_mean = total_stats.get(variant, {}).get("mean", float("nan"))
            plddt_mean = plddt_stats.get(variant, {}).get("mean", float("nan"))
            plddt_max = plddt_stats.get(variant, {}).get("max", float("nan"))
            lines.append(
                f"| {variant} | {n} | {mode_by_variant.get(variant, '')} | {planned_mean:.2f} | {unique_mean:.2f} | "
                f"{total_mean:.2f} | {plddt_mean:.3f} | {plddt_max:.3f} |"
            )
    else:
        lines.append("No inference results yet. This is expected for `--dry-run`.")
    lines.extend(
        [
            "",
            "## PG Contrast",
            "",
            "These PG statistics are from the fixed 50-candidate external ESMFold reevaluation. "
            "Candidates were selected before external ESMFold scoring by motif statistics and PG-internal pLDDT.",
            "",
            "| PG reference | n | unique motifs mean/median/max | total motifs mean/median/max | PG pLDDT mean/median | external ESMFold pLDDT mean/median/min/max | caveat |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for key, ref in PG_REFERENCE.items():
        lines.append(
            f"| {key} | {ref['n']} | "
            f"{ref['unique_motifs_mean']:.2f}/{ref['unique_motifs_median']:.2f}/{ref['unique_motifs_max']} | "
            f"{ref['total_motifs_mean']:.2f}/{ref['total_motifs_median']:.2f}/{ref['total_motifs_max']} | "
            f"{ref['pg_plddt_mean']:.3f}/{ref['pg_plddt_median']:.3f} | "
            f"{ref['mean_external_esmfold_plddt']:.3f}/{ref['median_external_esmfold_plddt']:.3f}/"
            f"{ref['min_external_esmfold_plddt']:.3f}/{ref['max_external_esmfold_plddt']:.3f} | "
            f"{ref['note']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `run_manifest.json`: immutable run setup and PG reference metrics.",
            "- `planned_runs.csv`: exact commands and selected motif subset per design.",
            "- `final_disco_designs.csv`: final saved DISCO sequences and motif scores.",
            "- `trajectory_summary.csv`: per-step diagnostic table when DISCO wrote guidance diagnostics.",
            "- `runs/*/*/disco_output/trajectories/*_trajectory_manifest.json`: raw per-sample tensor trajectory manifests.",
            "- `runs/*/*/disco_output/trajectories/*.pt`: raw coordinate trajectory tensors for later analysis.",
            "- `normalized_designs.csv`: exact ESMFold input table.",
            "- `esmfold_plddt_scoring/esmfold_plddt_scores.csv`: external pLDDT and saved PDBs.",
            "- `final_scored_designs.csv`: joined final analysis table.",
            "- `strict_validation.json`: machine-readable validation.",
            "- `results/RESULTS_ANALYSIS.md`: automatic post-run analysis index.",
            "- `results/tables/*.csv`: generated summary tables.",
            "- `results/figures/*.png`: generated result diagrams.",
        ]
    )
    (args.out / "CODEOCEAN_DISCO_15MOTIF_REPORT.md").write_text("\n".join(lines) + "\n")


def run_unit_checks(args: argparse.Namespace) -> None:
    assert parse_int_list("0-2,5") == [0, 1, 2, 5]
    motifs = parse_motifs(args.motifs)
    for target_count in [6, 8, 10, 12, 15]:
        observed = set()
        for seed_index in range(20):
            subset = choose_motif_subset(motifs, target_count, seed_index)
            assert len(subset) == target_count
            assert len(set(subset)) == target_count
            assert set(subset).issubset(set(motifs))
            starts = ideal_hard_commit_starts_1based(subset, args.length)
            assert len(starts) == target_count
            shifted_starts = ideal_hard_commit_starts_1based(
                subset,
                args.length,
                position_offset=3,
            )
            assert len(shifted_starts) == target_count
            observed.update(subset)
        assert observed == set(motifs)
    sample = choose_motif_subset(motifs, 6, 0)
    command = build_inference_command(
        python_executable=args.python_executable,
        disco_root=args.disco_root,
        experiment=args.experiment,
        input_json_path=Path("/tmp/input.json"),
        dump_dir=Path("/tmp/dump"),
        seed=0,
        motifs=sample,
        length=args.length,
        n_step=args.n_step,
        n_cycle=args.n_cycle,
        position_offset=3,
        save_trajectories=True,
        trajectory_dtype=args.trajectory_dtype,
    )
    command_text = " ".join(command)
    assert "hard_commit_enabled=true" in command_text
    assert "hard_commit_position_offset=3" in command_text
    assert "hard_commit_max_motifs_per_step=6" in command_text
    assert "cogen_guidance.trajectory_dump.enabled=true" in command_text
    assert "cogen_guidance.trajectory_dump.max_steps=0" in command_text
    assert f"cogen_guidance.motifs={hydra_list(sample)}" in command_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "codeocean_hardcommit_15motif_sweep_t50_l80_2026_06_08")
    parser.add_argument("--disco-root", type=Path, default=ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--esmfold-script", type=Path, default=ROOT / "outputs" / "external_quality_scoring" / "score_esmfold_plddt.py")
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--target-counts", default="6,8,10,12,15")
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--n-step", type=int, default=50)
    parser.add_argument("--n-cycle", type=int, default=4)
    parser.add_argument("--experiment", default="designable")
    parser.add_argument(
        "--include-soft-aggressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a 10-run aggressive soft/local-rescue guidance control block.",
    )
    parser.add_argument("--soft-seeds", default="0-9")
    parser.add_argument("--soft-target-count", type=int, default=10)
    parser.add_argument("--soft-gradient-scale", type=float, default=50.0)
    parser.add_argument("--soft-local-rescue-bonus", type=float, default=10.0)
    parser.add_argument("--soft-motif-temperature", type=float, default=0.35)
    parser.add_argument(
        "--placement-mode",
        choices=["even", "even_shifted"],
        default="even_shifted",
        help=(
            "even keeps the same spread plan for every seed. even_shifted uses "
            "seed-dependent offsets, where offset 0 is the even-spread baseline."
        ),
    )
    parser.add_argument(
        "--shift-offsets",
        default="0,-3,3,-5,5,-2,2,-4,4,-6,6,-1,1,-7,7,-8,8,-9,9,-10",
        help="Comma/range list of deterministic position offsets used by even_shifted.",
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--num-recycles", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--no-trajectory-dump",
        action="store_true",
        help="Disable raw coordinate trajectory tensor persistence. Do not use for the main CodeOcean run.",
    )
    parser.add_argument(
        "--trajectory-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Storage dtype for raw trajectory tensors. float32 preserves maximum fidelity.",
    )
    parser.add_argument("--skip-esmfold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-unit-checks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    args.disco_root = args.disco_root.resolve()
    args.esmfold_script = args.esmfold_script.resolve()
    if args.run_unit_checks:
        run_unit_checks(args)
    runs = plan_runs(args)
    write_plan_artifacts(args, runs)
    if args.dry_run:
        write_trajectory_summary(args, runs)
        generate_results_analysis(args, [])
        validation = validate_outputs(args, runs, [])
        write_report(args, [], validation)
        print(f"Dry-run plan written to {args.out}", flush=True)
        return 0
    run_results = execute_runs(args, runs)
    write_csv(args.out / "run_results.csv", run_results)
    final_rows = write_final_disco_designs(args, runs)
    write_trajectory_summary(args, runs)
    if final_rows and not args.skip_esmfold:
        run_esmfold(args)
    scored_rows = merge_scoring(args, final_rows)
    generate_results_analysis(args, scored_rows)
    validation = validate_outputs(args, runs, scored_rows)
    write_report(args, scored_rows, validation)
    print(f"Completed with validation={validation['status']} at {args.out}", flush=True)
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
