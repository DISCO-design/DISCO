"""Serialization helpers for DISCO guidance diagnostics and trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def to_jsonable(value: Any) -> Any:
    """Convert tensors and nested containers into JSON-safe values."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def scalar_or_json(value: Any) -> str | int | float | bool | None:
    """Return scalar CSV values and JSON-encode nested structures."""

    value = to_jsonable(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: scalar_or_json(record.get(key)) for key in fieldnames})


def diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact run-level fields from per-step diagnostic records."""

    if not records:
        return {
            "num_diagnostic_records": 0,
            "best_step": "",
            "best_score": "",
            "final_sequence": "",
        }

    step_records = [record for record in records if "step" in record]
    scored_records = step_records or records
    best_record = None
    for record in scored_records:
        if "guidance_combined_score" not in record:
            continue
        if best_record is None or float(record["guidance_combined_score"]) > float(
            best_record["guidance_combined_score"]
        ):
            best_record = record

    final_record = step_records[-1] if step_records else records[-1]
    return {
        "num_diagnostic_records": len(records),
        "num_step_records": len(step_records),
        "best_step": "" if best_record is None else best_record.get("step", ""),
        "best_score": ""
        if best_record is None
        else best_record.get("guidance_combined_score", ""),
        "best_sequence": "" if best_record is None else best_record.get("guidance_sequence", ""),
        "final_step": final_record.get("step", ""),
        "final_sequence": final_record.get("guidance_sequence", ""),
        "best_output_selected": any(
            bool(record.get("guidance_best_output_selected", False))
            for record in records
        ),
    }


def write_diagnostic_outputs(
    *,
    diagnostics_dir: Path,
    sample_id: str,
    records: list[dict[str, Any]],
    runtime_seconds: float,
    sample_name: str,
    seed: int,
) -> dict[str, Any]:
    """Write diagnostics JSONL/CSV/manifest and return path metadata."""

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = diagnostics_dir / f"{sample_id}_guidance_diagnostics.jsonl"
    csv_path = diagnostics_dir / f"{sample_id}_guidance_diagnostics.csv"
    manifest_path = diagnostics_dir / f"{sample_id}_run_metadata.json"

    records = [to_jsonable(record) for record in records]
    write_jsonl(jsonl_path, records)
    write_csv(csv_path, records)
    summary = diagnostic_summary(records)
    manifest = {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "diagnostics_jsonl": str(jsonl_path),
        "diagnostics_csv": str(csv_path),
        **summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "diagnostics_jsonl": str(jsonl_path),
        "diagnostics_csv": str(csv_path),
        "diagnostics_manifest": str(manifest_path),
        "runtime_seconds": runtime_seconds,
        **summary,
    }
