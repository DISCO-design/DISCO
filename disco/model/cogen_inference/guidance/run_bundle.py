"""Reproducible run-bundle manifests for DISCO guidance experiments.

The comparison planner knows what should be run, diagnostics know what happened
inside one guided DISCO sample, and the analysis CSV knows how completed outputs
scored.  This module ties those layers together into one small JSON manifest so
later PG-vs-DISCO comparisons can audit parameters, files, runtime, provenance,
and score summaries from a single artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .diagnostics import to_jsonable


RUN_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RunBundleConfig:
    """Inputs for one post-run reproducibility bundle."""

    run_dir: Path
    out_json: Path
    system: str = "DISCO"
    mode: str = ""
    variant: str = ""
    replicate: str = ""
    seed: str = ""
    dump_dir: Path | None = None
    summary_csv: Path | None = None
    planned_run_manifest: Path | None = None
    analysis_manifest: Path | None = None
    diagnostics_dir: Path | None = None
    repo_dir: Path | None = None
    sha256_max_bytes: int = 10_000_000
    include_git: bool = True


@dataclass(frozen=True)
class RunBundleIssue:
    """One structured finding from run-bundle validation."""

    severity: str
    check: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def safe_read_json(path: Path | None) -> dict[str, Any]:
    """Read a JSON object, returning an error object instead of throwing."""

    if path is None:
        return {}
    if not path.exists():
        return {"_status": "missing", "path": str(path)}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"_status": "invalid_json", "path": str(path), "error": str(exc)}
    if not isinstance(data, dict):
        return {"_status": "not_object", "path": str(path)}
    return data


def sha256_file(path: Path) -> str:
    """Compute a SHA256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | None, *, sha256_max_bytes: int) -> dict[str, Any]:
    """Return stable metadata for a file path."""

    if path is None:
        return {"path": "", "exists": False, "status": "not_provided"}
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "exists": False, "status": "missing"}
    if not path.is_file():
        return {"path": str(path), "exists": True, "status": "not_file"}

    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "status": "ok",
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime,
            timezone.utc,
        ).isoformat(),
    }
    if stat.st_size <= sha256_max_bytes:
        record["sha256"] = sha256_file(path)
        record["sha256_status"] = "computed"
    else:
        record["sha256"] = ""
        record["sha256_status"] = "skipped_large_file"
    return record


def summarize_directory(
    path: Path | None,
    *,
    patterns: Iterable[str],
    sha256_max_bytes: int,
    max_records: int = 50,
) -> dict[str, Any]:
    """Summarize matching files in a directory without embedding file contents."""

    if path is None:
        return {"path": "", "exists": False, "files": [], "num_files": 0}
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "exists": False, "files": [], "num_files": 0}
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(item for item in path.glob(pattern) if item.is_file()))
    unique_files = sorted(set(files))
    return {
        "path": str(path),
        "exists": True,
        "num_files": len(unique_files),
        "files_truncated": len(unique_files) > max_records,
        "files": [
            file_record(item, sha256_max_bytes=sha256_max_bytes)
            for item in unique_files[:max_records]
        ],
    }


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    """Read CSV rows when a summary path exists."""

    if path is None or not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value in ["", None]:
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def summarize_numeric_column(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    values = numeric_values(rows, column)
    if not values:
        return {"column": column, "num_values": 0}
    return {
        "column": column,
        "num_values": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def summarize_summary_csv(path: Path | None) -> dict[str, Any]:
    """Return compact, comparison-relevant metadata from an analysis CSV."""

    rows = read_csv_rows(path)
    fieldnames = list(rows[0].keys()) if rows else []
    score_columns = [
        "comparison_score",
        "confidence",
        "plddt",
        "motif_score",
        "num_unique_motifs_present",
        "num_motif_occurrences_total",
        "runtime_seconds",
    ]
    summary: dict[str, Any] = {
        "path": "" if path is None else str(path.resolve()),
        "exists": bool(path is not None and path.exists()),
        "num_rows": len(rows),
        "fieldnames": fieldnames,
        "numeric_columns": [
            summarize_numeric_column(rows, column)
            for column in score_columns
            if column in fieldnames
        ],
    }
    for column in [
        "system",
        "mode",
        "variant",
        "score_provenance_status",
        "evaluation_status",
        "confidence_match_status",
        "fixed_fragments_preserved",
    ]:
        if column in fieldnames:
            summary[f"{column}_counts"] = dict(
                sorted(Counter(row.get(column, "") for row in rows).items())
            )
    return summary


def diagnostics_metadata_records(path: Path | None) -> list[dict[str, Any]]:
    """Read all DISCO `*_run_metadata.json` files in a diagnostics directory."""

    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(path.glob("*_run_metadata.json")):
        record = safe_read_json(metadata_path)
        record["metadata_path"] = str(metadata_path.resolve())
        records.append(record)
    return records


def summarize_diagnostics(path: Path | None, *, sha256_max_bytes: int) -> dict[str, Any]:
    """Summarize per-sample diagnostics metadata and files."""

    records = diagnostics_metadata_records(path)
    runtimes = [
        float(record["runtime_seconds"])
        for record in records
        if str(record.get("runtime_seconds", "")).strip()
    ]
    diagnostic_counts = [
        int(record["num_diagnostic_records"])
        for record in records
        if str(record.get("num_diagnostic_records", "")).strip()
    ]
    return {
        "path": "" if path is None else str(path.resolve()),
        "exists": bool(path is not None and path.exists()),
        "num_metadata_records": len(records),
        "total_runtime_seconds": sum(runtimes) if runtimes else "",
        "mean_runtime_seconds": sum(runtimes) / len(runtimes) if runtimes else "",
        "total_diagnostic_records": sum(diagnostic_counts) if diagnostic_counts else 0,
        "num_best_output_selected": sum(
            1 for record in records if bool(record.get("best_output_selected", False))
        ),
        "metadata_records": records,
        "files": summarize_directory(
            path,
            patterns=["*_run_metadata.json", "*_guidance_diagnostics.jsonl", "*_guidance_diagnostics.csv"],
            sha256_max_bytes=sha256_max_bytes,
        ),
    }


def git_metadata(repo_dir: Path) -> dict[str, Any]:
    """Return lightweight git metadata without mutating the working tree."""

    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "repo_dir": str(repo_dir.resolve()),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def environment_metadata(repo_dir: Path, *, include_git: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    if include_git:
        metadata["git"] = git_metadata(repo_dir)
    return metadata


def create_run_bundle(config: RunBundleConfig) -> dict[str, Any]:
    """Build a reproducibility bundle from existing run artifacts."""

    run_dir = config.run_dir.resolve()
    planned_manifest = (
        config.planned_run_manifest.resolve()
        if config.planned_run_manifest is not None
        else run_dir / "run_manifest.json"
    )
    analysis_manifest = (
        config.analysis_manifest.resolve()
        if config.analysis_manifest is not None
        else (
            config.summary_csv.resolve().with_suffix(".manifest.json")
            if config.summary_csv is not None
            else None
        )
    )
    diagnostics_dir = (
        config.diagnostics_dir.resolve()
        if config.diagnostics_dir is not None
        else (
            config.dump_dir.resolve() / "diagnostics"
            if config.dump_dir is not None
            else None
        )
    )

    bundle = {
        "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
        "bundle_type": "disco_guidance_run_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "system": config.system,
            "mode": config.mode,
            "variant": config.variant,
            "replicate": config.replicate,
            "seed": config.seed,
        },
        "paths": {
            "run_dir": str(run_dir),
            "dump_dir": "" if config.dump_dir is None else str(config.dump_dir.resolve()),
            "summary_csv": ""
            if config.summary_csv is None
            else str(config.summary_csv.resolve()),
            "planned_run_manifest": str(planned_manifest),
            "analysis_manifest": "" if analysis_manifest is None else str(analysis_manifest),
            "diagnostics_dir": "" if diagnostics_dir is None else str(diagnostics_dir),
        },
        "planned_run": safe_read_json(planned_manifest),
        "analysis_manifest": safe_read_json(analysis_manifest),
        "summary_csv": summarize_summary_csv(config.summary_csv),
        "diagnostics": summarize_diagnostics(
            diagnostics_dir,
            sha256_max_bytes=config.sha256_max_bytes,
        ),
        "artifact_files": {
            "command_txt": file_record(
                run_dir / "command.txt",
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "analysis_command_txt": file_record(
                run_dir / "analysis_command.txt",
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "planned_run_manifest": file_record(
                planned_manifest,
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "analysis_manifest": file_record(
                analysis_manifest,
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "summary_csv": file_record(
                config.summary_csv,
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "sequences": summarize_directory(
                None if config.dump_dir is None else config.dump_dir / "sequences",
                patterns=["*.txt"],
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "structures": summarize_directory(
                None if config.dump_dir is None else config.dump_dir / "pdbs",
                patterns=["*.pdb", "*.cif"],
                sha256_max_bytes=config.sha256_max_bytes,
            ),
            "trajectories": summarize_directory(
                None if config.dump_dir is None else config.dump_dir / "trajectories",
                patterns=["*.pt", "*_trajectory_manifest.json"],
                sha256_max_bytes=config.sha256_max_bytes,
            ),
        },
        "environment": environment_metadata(
            Path.cwd() if config.repo_dir is None else config.repo_dir.resolve(),
            include_git=config.include_git,
        ),
    }
    return to_jsonable(bundle)


def issue(
    *,
    severity: str,
    check: str,
    path: str,
    message: str,
) -> RunBundleIssue:
    return RunBundleIssue(
        severity=severity,
        check=check,
        path=path,
        message=message,
    )


def validate_run_bundle(
    bundle: dict[str, Any],
    *,
    require_summary: bool = True,
    require_diagnostics: bool = False,
    require_plan: bool = False,
) -> list[RunBundleIssue]:
    """Validate that a run bundle has enough evidence for later comparison."""

    issues: list[RunBundleIssue] = []
    paths = bundle.get("paths", {})
    planned = bundle.get("planned_run", {})
    summary = bundle.get("summary_csv", {})
    diagnostics = bundle.get("diagnostics", {})

    if require_plan and planned.get("_status") in {
        "missing",
        "invalid_json",
        "not_object",
    }:
        issues.append(
            issue(
                severity="error",
                check="planned_manifest",
                path=str(paths.get("planned_run_manifest", "")),
                message="planned run manifest is required but missing or invalid",
            )
        )

    if require_summary:
        if not summary.get("exists", False):
            issues.append(
                issue(
                    severity="error",
                    check="summary_csv_exists",
                    path=str(paths.get("summary_csv", "")),
                    message="summary CSV is required but missing",
                )
            )
        elif int(summary.get("num_rows", 0)) == 0:
            issues.append(
                issue(
                    severity="error",
                    check="summary_csv_rows",
                    path=str(paths.get("summary_csv", "")),
                    message="summary CSV has no design rows",
                )
            )

    if require_diagnostics:
        if not diagnostics.get("exists", False):
            issues.append(
                issue(
                    severity="error",
                    check="diagnostics_dir_exists",
                    path=str(paths.get("diagnostics_dir", "")),
                    message="diagnostics directory is required but missing",
                )
            )
        elif int(diagnostics.get("num_metadata_records", 0)) == 0:
            issues.append(
                issue(
                    severity="error",
                    check="diagnostics_metadata",
                    path=str(paths.get("diagnostics_dir", "")),
                    message="diagnostics directory has no run metadata records",
                )
            )

    artifact_files = bundle.get("artifact_files", {})
    for key in ["command_txt", "analysis_command_txt"]:
        record = artifact_files.get(key, {})
        if record.get("exists", False):
            continue
        issues.append(
            issue(
                severity="warning",
                check=f"{key}_exists",
                path=str(record.get("path", "")),
                message=f"{key} is missing; command reproduction will be weaker",
            )
        )

    return issues


def write_run_bundle(config: RunBundleConfig) -> dict[str, Any]:
    """Create, validate lightly, write, and return one run bundle."""

    bundle = create_run_bundle(config)
    config.out_json.parent.mkdir(parents=True, exist_ok=True)
    config.out_json.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle
