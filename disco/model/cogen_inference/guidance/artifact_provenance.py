"""Artifact provenance manifests for DISCO guidance inference runs.

The comparison pipeline relies on a strict contract: a score is only comparable
when it is tied to the exact sequence/structure artifact that was saved.  This
module writes small sidecar manifests that bind final structures, final
sequence text, confidence JSONs, diagnostics, and optional trajectory tensors.
Intermediate trajectory states can be analyzed later from tensors and
diagnostic records, but they are explicitly marked as not having final
pLDDT/confidence unless a downstream tool computes it for that saved state.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .diagnostics import to_jsonable


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    """Return a SHA256 digest for `path`, optionally over the first bytes."""

    digest = hashlib.sha256()
    remaining = max_bytes
    with path.open("rb") as handle:
        while True:
            size = 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                size = min(size, remaining)
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def file_record(path: str | Path | None) -> dict[str, Any]:
    """Return existence, size, and digest metadata for one file path."""

    if path in [None, ""]:
        return {"path": "", "exists": False}
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_json(path: str | Path | None) -> dict[str, Any]:
    if path in [None, ""]:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text())


def final_score_binding(
    *,
    structure_path: str | Path | None,
    sequence_path: str | Path | None,
    confidence_paths: list[str],
    diagnostics_manifest: str | Path | None,
) -> dict[str, Any]:
    """Describe exactly which saved artifacts final scores are bound to."""

    confidence_records = [file_record(path) for path in confidence_paths]
    confidence_json = read_json(confidence_paths[0]) if confidence_paths else {}
    diagnostics = read_json(diagnostics_manifest)
    return {
        "score_scope": "final_saved_output",
        "structure_artifact": file_record(structure_path),
        "sequence_artifact": file_record(sequence_path),
        "confidence_artifacts": confidence_records,
        "diagnostics_manifest": file_record(diagnostics_manifest),
        "confidence_metric": next(
            (
                key
                for key in ["ranking_score", "plddt", "mean_plddt", "ptm"]
                if key in confidence_json
            ),
            "",
        ),
        "confidence_value": next(
            (
                confidence_json[key]
                for key in ["ranking_score", "plddt", "mean_plddt", "ptm"]
                if key in confidence_json
            ),
            "",
        ),
        "guidance_best_step": diagnostics.get("best_step", ""),
        "guidance_best_score": diagnostics.get("best_score", ""),
        "guidance_final_step": diagnostics.get("final_step", ""),
        "guidance_final_sequence": diagnostics.get("final_sequence", ""),
        "provenance_rule": (
            "Final motif/sequence scores are computed from the saved sequence; "
            "final confidence is read from the confidence JSON written for the "
            "same saved sample. Intermediate states require their own row-level "
            "score artifacts before being used in comparisons."
        ),
    }


def trajectory_tensor_map(trajectory_manifest: str | Path | None) -> dict[str, Any]:
    manifest = read_json(trajectory_manifest)
    tensors = {}
    for item in manifest.get("saved_tensors", []):
        key = item.get("key", "")
        if key:
            tensors[key] = item
    return {
        "manifest": manifest,
        "tensors": tensors,
    }


def write_intermediate_state_index(
    *,
    path: Path,
    sample_id: str,
    diagnostic_records: list[dict[str, Any]],
    trajectory_manifest: str | Path | None,
) -> int:
    """Write a CSV index linking diagnostics to optional trajectory tensors."""

    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = trajectory_tensor_map(trajectory_manifest)
    tensors = trajectory["tensors"]
    fieldnames = [
        "sample_id",
        "state_index",
        "step",
        "structure_state_kind",
        "sequence",
        "motif_score",
        "combined_score",
        "sequence_confidence",
        "trajectory_tensor_path",
        "trajectory_tensor_index",
        "saved_structure_path",
        "structure_confidence_status",
        "provenance_note",
    ]
    state_kinds = [
        key for key in ["x1_denoised", "xt_trajectory"] if key in tensors
    ] or ["diagnostic_sequence_only"]
    step_records = [record for record in diagnostic_records if "step" in record]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(step_records):
            for state_kind in state_kinds:
                tensor = tensors.get(state_kind, {})
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "state_index": index,
                        "step": record.get("step", ""),
                        "structure_state_kind": state_kind,
                        "sequence": record.get("guidance_sequence", ""),
                        "motif_score": record.get("guidance_motif_score", ""),
                        "combined_score": record.get("guidance_combined_score", ""),
                        "sequence_confidence": record.get(
                            "guidance_sequence_confidence", ""
                        ),
                        "trajectory_tensor_path": tensor.get("path", ""),
                        "trajectory_tensor_index": index if tensor else "",
                        "saved_structure_path": "",
                        "structure_confidence_status": (
                            "not_computed_for_intermediate_state"
                        ),
                        "provenance_note": (
                            "Intermediate motif scores come from guidance "
                            "diagnostics for this sequence state and are repeated "
                            "per saved structure-state kind. No pLDDT/confidence "
                            "is claimed unless a downstream analysis writes a "
                            "per-state structure and confidence artifact."
                        ),
                    }
                )
    return len(step_records) * len(state_kinds)


def write_artifact_provenance(
    *,
    provenance_dir: Path,
    sample_id: str,
    sample_name: str,
    seed: int,
    structure_path: str | Path | None,
    sequence_path: str | Path | None,
    confidence_paths: list[str],
    diagnostics_manifest: str | Path | None,
    diagnostic_records: list[dict[str, Any]],
    trajectory_manifest: str | Path | None,
    run_command: str | None = None,
) -> dict[str, Any]:
    """Write final and intermediate provenance sidecars for one sample."""

    provenance_dir.mkdir(parents=True, exist_ok=True)
    state_index_path = provenance_dir / f"{sample_id}_intermediate_states.csv"
    num_states = write_intermediate_state_index(
        path=state_index_path,
        sample_id=sample_id,
        diagnostic_records=[to_jsonable(record) for record in diagnostic_records],
        trajectory_manifest=trajectory_manifest,
    )
    manifest_path = provenance_dir / f"{sample_id}_artifact_provenance.json"
    manifest = {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "seed": seed,
        "run_command": run_command or "",
        "final": final_score_binding(
            structure_path=structure_path,
            sequence_path=sequence_path,
            confidence_paths=confidence_paths,
            diagnostics_manifest=diagnostics_manifest,
        ),
        "intermediate_state_index": file_record(state_index_path),
        "num_intermediate_states": num_states,
        "trajectory_manifest": file_record(trajectory_manifest),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "artifact_provenance_manifest": str(manifest_path),
        "intermediate_state_index": str(state_index_path),
        "num_intermediate_states": num_states,
    }
