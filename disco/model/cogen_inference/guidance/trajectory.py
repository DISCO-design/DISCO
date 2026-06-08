"""Offline trajectory analysis for DISCO guidance diagnostics.

DISCO's inference loop writes one diagnostic record per diffusion step when the
motif-guidance controller is active.  This module turns those raw JSONL records
into a compact, analysis-ready table.  It does not call the model, mutate
samples, or depend on torch; it is deliberately safe to run after completed
experiments on a laptop.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .diagnostics import scalar_or_json, to_jsonable


DIAGNOSTICS_SUFFIX = "_guidance_diagnostics"

TRAJECTORY_FIELD_PRIORITY = [
    "sample_id",
    "source_path",
    "trajectory_index",
    "step",
    "sequence",
    "sequence_length",
    "combined_score",
    "motif_score",
    "all_score",
    "many_score",
    "sequence_confidence",
    "unique_motifs",
    "total_motifs",
    "present_motifs",
    "missing_motifs",
    "new_present_motifs",
    "lost_present_motifs",
    "combined_score_delta",
    "unique_motifs_delta",
    "total_motifs_delta",
    "sequence_changes_from_previous",
    "best_step_so_far",
    "best_score_so_far",
    "is_best_combined",
    "is_best_update",
    "is_final_step",
    "best_output_selected_record",
    "num_seq_changes",
    "num_masked",
    "num_unmasked",
    "num_newly_unmasked",
    "entropy_mean",
    "entropy_std",
    "entropy_min",
    "entropy_max",
    "t_next",
    "c_tau",
    "c_tau_last",
    "score_type",
]


def read_diagnostic_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a DISCO guidance diagnostics JSONL file."""

    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def sample_id_from_diagnostics_path(path: Path) -> str:
    """Infer the DISCO sample id from `*_guidance_diagnostics.jsonl`."""

    stem = path.stem
    if stem.endswith(DIAGNOSTICS_SUFFIX):
        return stem[: -len(DIAGNOSTICS_SUFFIX)]
    return stem


def as_float(value: Any) -> float | None:
    if value in ["", None]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value in ["", None]:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def split_motif_field(value: Any) -> list[str]:
    """Normalize comma/semicolon-separated motif fields from diagnostics."""

    if value in ["", None]:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).replace(";", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def hamming_distance(left: str, right: str) -> int | str:
    """Return sequence Hamming distance, or blank when lengths are incompatible."""

    if not left or not right or len(left) != len(right):
        return ""
    return sum(1 for a, b in zip(left, right) if a != b)


def _delta(current: int | float | None, previous: int | float | None) -> int | float | str:
    if current is None or previous is None:
        return ""
    value = current - previous
    if isinstance(value, float):
        return round(value, 12)
    return value


def _copy_sampler_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "num_seq_changes",
        "num_masked",
        "num_unmasked",
        "num_newly_unmasked",
        "entropy_mean",
        "entropy_std",
        "entropy_min",
        "entropy_max",
        "t_next",
        "c_tau",
        "c_tau_last",
        "score_type",
    ]
    return {field: record.get(field, "") for field in fields}


def trajectory_rows(
    records: list[dict[str, Any]],
    *,
    sample_id: str,
    source_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert raw per-step diagnostic records into analysis-ready rows."""

    rows: list[dict[str, Any]] = []
    best_score: float | None = None
    best_step: int | str = ""
    best_indices: set[int] = set()

    scored_records = [
        (idx, as_float(record.get("guidance_combined_score")))
        for idx, record in enumerate(records)
    ]
    valid_scores = [score for _, score in scored_records if score is not None]
    global_best = max(valid_scores) if valid_scores else None
    if global_best is not None:
        best_indices = {
            idx
            for idx, score in scored_records
            if score is not None and score == global_best
        }

    previous_sequence = ""
    previous_score: float | None = None
    previous_unique: int | None = None
    previous_total: int | None = None
    previous_present: set[str] = set()

    for index, record in enumerate(records):
        step_value = as_int(record.get("step"))
        step: int | str = "" if step_value is None else step_value
        sequence = str(record.get("guidance_sequence") or "")
        combined_score = as_float(record.get("guidance_combined_score"))
        unique_motifs = as_int(record.get("guidance_num_unique"))
        total_motifs = as_int(record.get("guidance_num_total"))
        present = set(split_motif_field(record.get("guidance_present")))
        missing = set(split_motif_field(record.get("guidance_missing")))

        if combined_score is not None and (
            best_score is None or combined_score > best_score
        ):
            best_score = combined_score
            best_step = step

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_path": "" if source_path is None else str(source_path),
            "trajectory_index": index,
            "step": step,
            "sequence": sequence,
            "sequence_length": len(sequence) if sequence else "",
            "combined_score": "" if combined_score is None else combined_score,
            "motif_score": record.get("guidance_motif_score", ""),
            "all_score": record.get("guidance_all_score", ""),
            "many_score": record.get("guidance_many_score", ""),
            "sequence_confidence": record.get("guidance_sequence_confidence", ""),
            "unique_motifs": "" if unique_motifs is None else unique_motifs,
            "total_motifs": "" if total_motifs is None else total_motifs,
            "present_motifs": ",".join(sorted(present)),
            "missing_motifs": ",".join(sorted(missing)),
            "new_present_motifs": ",".join(sorted(present - previous_present)),
            "lost_present_motifs": ",".join(sorted(previous_present - present)),
            "combined_score_delta": _delta(combined_score, previous_score),
            "unique_motifs_delta": _delta(unique_motifs, previous_unique),
            "total_motifs_delta": _delta(total_motifs, previous_total),
            "sequence_changes_from_previous": hamming_distance(
                previous_sequence,
                sequence,
            ),
            "best_step_so_far": best_step,
            "best_score_so_far": "" if best_score is None else best_score,
            "is_best_combined": index in best_indices,
            "is_best_update": bool(record.get("guidance_best_updated", False)),
            "is_final_step": index == len(records) - 1,
            "best_output_selected_record": bool(
                record.get("guidance_best_output_selected", False)
            ),
            **_copy_sampler_fields(record),
        }
        rows.append(row)

        if sequence:
            previous_sequence = sequence
        previous_score = combined_score
        previous_unique = unique_motifs
        previous_total = total_motifs
        previous_present = present

    return rows


def trajectory_summary(
    rows: list[dict[str, Any]],
    *,
    sample_id: str,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Summarize one sample trajectory."""

    if not rows:
        return {
            "sample_id": sample_id,
            "source_path": "" if source_path is None else str(source_path),
            "status": "empty",
            "num_trajectory_steps": 0,
            "num_scored_steps": 0,
        }

    scored = [row for row in rows if as_float(row.get("combined_score")) is not None]
    best_row = None
    if scored:
        best_row = max(scored, key=lambda row: float(row["combined_score"]))
    final_row = rows[-1]
    final_score = as_float(final_row.get("combined_score"))
    best_score = None if best_row is None else as_float(best_row.get("combined_score"))

    return {
        "sample_id": sample_id,
        "source_path": "" if source_path is None else str(source_path),
        "status": "ok",
        "num_trajectory_steps": len(rows),
        "num_scored_steps": len(scored),
        "first_step": rows[0].get("step", ""),
        "final_step": final_row.get("step", ""),
        "best_step": "" if best_row is None else best_row.get("step", ""),
        "best_combined_score": "" if best_score is None else best_score,
        "final_combined_score": "" if final_score is None else final_score,
        "best_unique_motifs": "" if best_row is None else best_row.get("unique_motifs", ""),
        "final_unique_motifs": final_row.get("unique_motifs", ""),
        "best_sequence": "" if best_row is None else best_row.get("sequence", ""),
        "final_sequence": final_row.get("sequence", ""),
        "total_sequence_changes": sum(
            int(value)
            for value in (
                row.get("sequence_changes_from_previous", "") for row in rows
            )
            if isinstance(value, int)
        ),
        "best_output_selected": any(
            bool(row.get("best_output_selected_record", False)) for row in rows
        ),
        "best_beats_final": bool(
            best_score is not None
            and final_score is not None
            and best_score > final_score
        ),
    }


def write_trajectory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write trajectory rows with stable high-signal columns first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    ordered = [key for key in TRAJECTORY_FIELD_PRIORITY if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_or_json(row.get(key)) for key in ordered})


def analyze_trajectory_files(
    diagnostics_jsonl_paths: list[Path],
    *,
    sample_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze one or more diagnostics JSONL files."""

    if sample_ids is not None and len(sample_ids) != len(diagnostics_jsonl_paths):
        raise ValueError("sample_ids must match diagnostics_jsonl_paths length")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, path in enumerate(diagnostics_jsonl_paths):
        resolved = path.resolve()
        sample_id = (
            sample_ids[index]
            if sample_ids is not None
            else sample_id_from_diagnostics_path(resolved)
        )
        records = read_diagnostic_jsonl(resolved)
        rows = trajectory_rows(
            records,
            sample_id=sample_id,
            source_path=resolved,
        )
        all_rows.extend(rows)
        summaries.append(
            trajectory_summary(
                rows,
                sample_id=sample_id,
                source_path=resolved,
            )
        )
    return all_rows, summaries


def aggregate_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact aggregate over per-sample trajectory summaries."""

    return {
        "num_samples": len(summaries),
        "num_ok_samples": sum(1 for item in summaries if item.get("status") == "ok"),
        "num_empty_samples": sum(
            1 for item in summaries if item.get("status") == "empty"
        ),
        "total_trajectory_steps": sum(
            int(item.get("num_trajectory_steps", 0)) for item in summaries
        ),
        "total_scored_steps": sum(
            int(item.get("num_scored_steps", 0)) for item in summaries
        ),
        "samples_where_best_beats_final": sum(
            1 for item in summaries if bool(item.get("best_beats_final", False))
        ),
    }


def write_trajectory_outputs(
    *,
    diagnostics_jsonl_paths: list[Path],
    out_csv: Path,
    out_summary_json: Path,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Write trajectory CSV and JSON summary for one or more diagnostics files."""

    rows, summaries = analyze_trajectory_files(
        diagnostics_jsonl_paths,
        sample_ids=sample_ids,
    )
    write_trajectory_csv(out_csv, rows)
    summary = {
        "trajectory_csv": str(out_csv),
        "summary_json": str(out_summary_json),
        "aggregate": aggregate_summary(summaries),
        "samples": summaries,
    }
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary
