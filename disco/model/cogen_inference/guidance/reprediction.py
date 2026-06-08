"""Second-pass DISCO reprediction planning for completed guided designs.

ProteinGenerator's Immunogen workflow used an explicit reprediction step to
avoid mixing a sequence from one state with confidence or structure from
another.  This module provides the DISCO-side equivalent at the planning layer:
it reads completed analysis rows, writes full-sequence DISCO input JSONs, and
records the exact inference/analyzer commands needed for a second pass.

The functions do not execute DISCO.  That keeps the workflow safe on laptops
and makes the reproducibility artifact available before expensive runs start.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from disco.model.cogen_inference.guidance.evaluation_provenance import (
    row_has_accepted_same_artifact_provenance,
)
from disco.model.cogen_inference.guidance.experiment_plan import (
    FixedFragmentPlan,
    command_to_string,
    hydra_list,
)
from disco.model.cogen_inference.guidance.scoring import parse_motifs


AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RepredictionPlanConfig:
    """Configuration for materializing second-pass DISCO commands."""

    analysis_csv: Path
    out_dir: Path
    disco_root: Path
    motifs: list[str]
    seeds: list[int]
    n_cycle: int = 4
    n_step: int = 100
    experiment: str = "designable"
    python: str = "python"
    max_designs: int | None = None
    rank_by: str = "comparison_score"
    min_motif_score: float | None = None
    min_confidence: float | None = None
    require_fixed_preserved: bool = False
    require_same_evaluation_artifact: bool = False
    fixed_fragments: list[FixedFragmentPlan] | None = None
    tcell_motifs: list[str] | None = None
    cleavage_motifs: list[str] | None = None
    variant: str = "reprediction"
    require_confidence_analysis: bool = True


@dataclass(frozen=True)
class RepredictionRun:
    """One row in a second-pass reprediction plan."""

    design_id: str
    source_row_index: int
    sequence: str
    status: str
    reason: str
    run_dir: Path
    input_json_path: Path
    dump_dir: Path
    summary_csv_path: Path
    command: list[str]
    analysis_command: list[str]
    source_row: dict[str, str]

    def csv_row(self) -> dict[str, object]:
        return {
            "design_id": self.design_id,
            "source_row_index": self.source_row_index,
            "sequence": self.sequence,
            "sequence_length": len(self.sequence),
            "status": self.status,
            "reason": self.reason,
            "run_dir": str(self.run_dir),
            "input_json_path": str(self.input_json_path),
            "dump_dir": str(self.dump_dir),
            "summary_csv_path": str(self.summary_csv_path),
            "command": command_to_string(self.command) if self.command else "",
            "analysis_command": (
                command_to_string(self.analysis_command)
                if self.analysis_command
                else ""
            ),
            "source_confidence": self.source_row.get("confidence", ""),
            "source_motif_score": self.source_row.get("motif_score", ""),
            "source_comparison_score": self.source_row.get("comparison_score", ""),
            "source_evaluation_status": self.source_row.get("evaluation_status", ""),
            "source_confidence_match_status": self.source_row.get(
                "confidence_match_status", ""
            ),
            "source_fixed_fragments_preserved": self.source_row.get(
                "fixed_fragments_preserved", ""
            ),
        }


def parse_seed_list(value: str | Iterable[int]) -> list[int]:
    """Parse comma-separated seeds or return an integer iterable as a list."""

    if isinstance(value, str):
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        seeds = [int(item) for item in value]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def safe_design_id(value: str, fallback: str) -> str:
    """Return a stable filesystem-safe design identifier."""

    raw = value.strip() or fallback
    safe = SAFE_NAME_RE.sub("_", raw).strip("._-")
    return safe or fallback


def _float_or_none(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def valid_full_sequence(sequence: str) -> bool:
    """Return whether a sequence is a concrete amino-acid string."""

    return bool(sequence) and all(aa in AMINO_ACIDS for aa in sequence)


def read_analysis_rows(path: Path) -> list[dict[str, str]]:
    """Read a DISCO/normalized analysis CSV as dictionaries."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_full_sequence_input_json(path: Path, *, name: str, sequence: str) -> Path:
    """Write a DISCO input JSON with a fully fixed protein sequence."""

    if not valid_full_sequence(sequence):
        raise ValueError(f"not a concrete amino-acid sequence: {sequence!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": 1,
                    }
                }
            ],
            "name": name,
        }
    ]
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _fixed_fragment_cli_parts(fragments: Iterable[FixedFragmentPlan]) -> list[str]:
    parts: list[str] = []
    for fragment in fragments:
        value = f"{fragment.start_1based}:{fragment.sequence}"
        if fragment.label:
            value = f"{value}:{fragment.label}"
        parts.extend(["--fixed-fragment", value])
    return parts


def build_reprediction_command(
    *,
    config: RepredictionPlanConfig,
    input_json_path: Path,
    dump_dir: Path,
    seed: int,
) -> list[str]:
    """Build the DISCO command for one fixed-sequence second-pass run."""

    return [
        config.python,
        str(config.disco_root / "runner" / "inference.py"),
        f"experiment={config.experiment}",
        f"input_json_path={input_json_path}",
        f"dump_dir={dump_dir}",
        f"seeds=[{seed}]",
        f"sample_diffusion.N_step={config.n_step}",
        f"model.N_cycle={config.n_cycle}",
        "cogen_guidance.enabled=false",
    ]


def build_reprediction_analysis_command(
    *,
    config: RepredictionPlanConfig,
    dump_dir: Path,
    summary_csv_path: Path,
    seed_sequence: str,
) -> list[str]:
    """Build the analyzer command for one completed second-pass run."""

    command = [
        config.python,
        str(config.disco_root / "runner" / "guidance_analyze_disco.py"),
        "--dump-dir",
        str(dump_dir),
        "--motifs",
        ",".join(parse_motifs(config.motifs)),
        "--out-csv",
        str(summary_csv_path),
        "--variant",
        config.variant,
        "--seed-sequence",
        seed_sequence,
    ]
    if config.require_confidence_analysis:
        command.append("--require-confidence")
    command.extend(_fixed_fragment_cli_parts(config.fixed_fragments or []))
    if config.tcell_motifs:
        command.extend(["--tcell-motifs", ",".join(parse_motifs(config.tcell_motifs))])
    if config.cleavage_motifs:
        command.extend(
            ["--cleavage-motifs", ",".join(parse_motifs(config.cleavage_motifs))]
        )
    return command


def _row_skip_reason(row: dict[str, str], config: RepredictionPlanConfig) -> str:
    sequence = str(row.get("sequence", "")).strip().upper()
    if not valid_full_sequence(sequence):
        return "missing_or_nonconcrete_sequence"

    confidence = _float_or_none(row.get("confidence", ""))
    if config.min_confidence is not None and (
        confidence is None or confidence < config.min_confidence
    ):
        return "below_min_confidence"

    motif_score = _float_or_none(row.get("motif_score", ""))
    if config.min_motif_score is not None and (
        motif_score is None or motif_score < config.min_motif_score
    ):
        return "below_min_motif_score"

    if config.require_fixed_preserved and not _truthy(
        row.get("fixed_fragments_preserved", "")
    ):
        return "fixed_fragments_not_preserved"

    if config.require_same_evaluation_artifact and not row_has_accepted_same_artifact_provenance(row):
        return "same_evaluation_artifact_not_proven"

    return ""


def _ranking_value(row: dict[str, str], rank_by: str) -> float:
    if rank_by == "input_order":
        return 0.0
    value = _float_or_none(row.get(rank_by, ""))
    if value is not None:
        return value
    fallback_order = ["comparison_score", "motif_score", "confidence"]
    for key in fallback_order:
        value = _float_or_none(row.get(key, ""))
        if value is not None:
            return value
    return float("-inf")


def _rank_rows(
    rows: list[tuple[int, dict[str, str]]],
    rank_by: str,
) -> list[tuple[int, dict[str, str]]]:
    if rank_by == "input_order":
        return rows
    return sorted(
        rows,
        key=lambda item: (_ranking_value(item[1], rank_by), -item[0]),
        reverse=True,
    )


def plan_reprediction_runs(config: RepredictionPlanConfig) -> list[RepredictionRun]:
    """Create all planned/skipped reprediction rows and write input JSONs."""

    rows = read_analysis_rows(config.analysis_csv)
    indexed_rows = list(enumerate(rows))
    eligible: list[tuple[int, dict[str, str]]] = []
    skipped_reasons: dict[int, str] = {}
    for index, row in indexed_rows:
        reason = _row_skip_reason(row, config)
        if reason:
            skipped_reasons[index] = reason
        else:
            eligible.append((index, row))

    ranked = _rank_rows(eligible, config.rank_by)
    planned_indices = {
        index
        for index, _row in (
            ranked
            if config.max_designs is None
            else ranked[: config.max_designs]
        )
    }
    if config.max_designs is not None:
        for index, _row in ranked[config.max_designs :]:
            skipped_reasons[index] = "beyond_max_designs"

    runs: list[RepredictionRun] = []
    for index, row in indexed_rows:
        design_id = safe_design_id(
            str(row.get("design_id") or row.get("base_id") or ""),
            fallback=f"row_{index}",
        )
        sequence = str(row.get("sequence", "")).strip().upper()
        run_dir = config.out_dir / f"{index:04d}_{design_id}"
        input_json_path = run_dir / "input.json"
        dump_dir = run_dir / "disco_reprediction_output"
        summary_csv_path = run_dir / "reprediction_analysis.csv"

        if index in planned_indices:
            write_full_sequence_input_json(
                input_json_path,
                name=f"{design_id}_repredict",
                sequence=sequence,
            )
            seed = config.seeds[0]
            command = build_reprediction_command(
                config=config,
                input_json_path=input_json_path,
                dump_dir=dump_dir,
                seed=seed,
            )
            analysis_command = build_reprediction_analysis_command(
                config=config,
                dump_dir=dump_dir,
                summary_csv_path=summary_csv_path,
                seed_sequence=sequence,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "command.txt").write_text(command_to_string(command) + "\n")
            (run_dir / "analysis_command.txt").write_text(
                command_to_string(analysis_command) + "\n"
            )
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "run_type": "disco_second_pass_reprediction",
                        "source_analysis_csv": str(config.analysis_csv),
                        "source_row_index": index,
                        "design_id": design_id,
                        "sequence_length": len(sequence),
                        "motifs": parse_motifs(config.motifs),
                        "seed": seed,
                        "n_cycle": config.n_cycle,
                        "n_step": config.n_step,
                        "rank_by": config.rank_by,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            status = "planned"
            reason = ""
        else:
            command = []
            analysis_command = []
            status = "skipped"
            reason = skipped_reasons.get(index, "not_selected")

        runs.append(
            RepredictionRun(
                design_id=design_id,
                source_row_index=index,
                sequence=sequence,
                status=status,
                reason=reason,
                run_dir=run_dir,
                input_json_path=input_json_path,
                dump_dir=dump_dir,
                summary_csv_path=summary_csv_path,
                command=command,
                analysis_command=analysis_command,
                source_row=dict(row),
            )
        )
    return runs


def write_reprediction_plan(config: RepredictionPlanConfig) -> dict[str, object]:
    """Write top-level second-pass plan CSV and manifest."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    runs = plan_reprediction_runs(config)
    csv_path = config.out_dir / "planned_repredictions.csv"
    rows = [run.csv_row() for run in runs]
    fieldnames = sorted({key for row in rows for key in row})
    priority = [
        "design_id",
        "source_row_index",
        "status",
        "reason",
        "sequence_length",
        "source_confidence",
        "source_motif_score",
        "source_comparison_score",
        "source_evaluation_status",
        "source_confidence_match_status",
        "source_fixed_fragments_preserved",
        "run_dir",
        "input_json_path",
        "dump_dir",
        "summary_csv_path",
        "command",
        "analysis_command",
    ]
    ordered = [key for key in priority if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = {
        status: sum(1 for run in runs if run.status == status)
        for status in sorted({run.status for run in runs})
    }
    reason_counts = {
        reason: sum(1 for run in runs if run.reason == reason)
        for reason in sorted({run.reason for run in runs if run.reason})
    }
    manifest = {
        "run_type": "disco_second_pass_reprediction_plan",
        "analysis_csv": str(config.analysis_csv),
        "out_dir": str(config.out_dir),
        "planned_csv": str(csv_path),
        "motifs": parse_motifs(config.motifs),
        "seeds": config.seeds,
        "n_cycle": config.n_cycle,
        "n_step": config.n_step,
        "rank_by": config.rank_by,
        "max_designs": config.max_designs,
        "filters": {
            "min_confidence": config.min_confidence,
            "min_motif_score": config.min_motif_score,
            "require_fixed_preserved": config.require_fixed_preserved,
            "require_same_evaluation_artifact": config.require_same_evaluation_artifact,
        },
        "num_source_rows": len(runs),
        "num_planned": status_counts.get("planned", 0),
        "num_skipped": status_counts.get("skipped", 0),
        "status_counts": status_counts,
        "skip_reason_counts": reason_counts,
    }
    manifest_path = config.out_dir / "reprediction_plan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
