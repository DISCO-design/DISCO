"""Validation gates for planned PG-vs-DISCO comparison studies.

The comparison planner makes runs reproducible; this module checks whether the
planned artifacts and completed summary CSVs are complete enough to support a
scientific comparison.  It intentionally reports structured issues instead of
silently normalizing partial data.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding for a planned or completed comparison run."""

    severity: str
    check: str
    system: str
    mode: str
    variant: str
    replicate: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_issue_csv(path: Path, issues: list[ValidationIssue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "severity",
        "check",
        "system",
        "mode",
        "variant",
        "replicate",
        "path",
        "message",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([issue.as_dict() for issue in issues])


def parse_parameters(row: dict[str, str]) -> dict[str, object]:
    try:
        return json.loads(row.get("parameters_json", "{}"))
    except json.JSONDecodeError:
        return {}


def issue(
    row: dict[str, str],
    *,
    severity: str,
    check: str,
    path: str | Path,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        check=check,
        system=row.get("system", ""),
        mode=row.get("mode", ""),
        variant=row.get("variant", ""),
        replicate=row.get("replicate", ""),
        path=str(path),
        message=message,
    )


def first_existing_value(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ["", None]:
            return str(value)
    return ""


def looks_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def truthy_string(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate_required_systems(
    plan_rows: list[dict[str, str]],
    required_systems: Iterable[str],
) -> list[ValidationIssue]:
    """Check that every mode/variant group contains the required systems."""

    required = {system for system in required_systems if system}
    if not required:
        return []

    issues: list[ValidationIssue] = []
    groups: dict[tuple[str, str], set[str]] = {}
    for row in plan_rows:
        groups.setdefault((row.get("mode", ""), row.get("variant", "")), set()).add(
            row.get("system", "")
        )

    for (mode, variant), observed in groups.items():
        missing = sorted(required - observed)
        if not missing:
            continue
        issues.append(
            issue(
                {
                    "system": "",
                    "mode": mode,
                    "variant": variant,
                    "replicate": "",
                },
                severity="error",
                check="required_systems",
                path="planned_runs.csv",
                message=(
                    "comparison group is missing required system(s): "
                    + ",".join(missing)
                ),
            )
        )
    return issues


def validate_matrix_balance(plan_rows: list[dict[str, str]]) -> list[ValidationIssue]:
    """Check that planned runs are balanced across systems for each variant."""

    issues: list[ValidationIssue] = []
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in plan_rows:
        grouped.setdefault(
            (row.get("mode", ""), row.get("variant", ""), row.get("system", "")),
            [],
        ).append(row)

    by_mode_variant: dict[tuple[str, str], dict[str, int]] = {}
    for (mode, variant, system), rows in grouped.items():
        by_mode_variant.setdefault((mode, variant), {})[system] = len(rows)

    for (mode, variant), counts in by_mode_variant.items():
        if len(counts) <= 1:
            continue
        unique_counts = set(counts.values())
        if len(unique_counts) == 1:
            continue
        row = {
            "system": "",
            "mode": mode,
            "variant": variant,
            "replicate": "",
        }
        issues.append(
            issue(
                row,
                severity="error",
                check="matrix_balance",
                path="planned_runs.csv",
                message=f"unequal planned run counts by system: {counts}",
            )
        )
    return issues


def validate_parameter_consistency(
    plan_rows: list[dict[str, str]],
) -> list[ValidationIssue]:
    """Check shared comparison knobs within each `(mode, variant)` group."""

    issues: list[ValidationIssue] = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in plan_rows:
        groups.setdefault((row.get("mode", ""), row.get("variant", "")), []).append(row)

    keys = ["n_cycle", "length", "motifs", "designs_per_replicate"]
    for (mode, variant), rows in groups.items():
        expected = parse_parameters(rows[0])
        for row in rows[1:]:
            observed = parse_parameters(row)
            for key in keys:
                if observed.get(key) == expected.get(key):
                    continue
                issues.append(
                    issue(
                        row,
                        severity="error",
                        check="parameter_consistency",
                        path=row.get("run_dir", ""),
                        message=(
                            f"{key} differs within {mode}/{variant}: "
                            f"{observed.get(key)!r} != {expected.get(key)!r}"
                        ),
                    )
                )
    return issues


def validate_planned_artifacts(row: dict[str, str]) -> list[ValidationIssue]:
    """Check planner-created files that should exist before inference."""

    issues = []
    run_dir = Path(row.get("run_dir", ""))
    for filename in ["command.txt", "analysis_command.txt", "run_manifest.json"]:
        path = run_dir / filename
        if path.exists():
            continue
        issues.append(
            issue(
                row,
                severity="error",
                check="planned_artifact_exists",
                path=path,
                message=f"missing planner artifact {filename}",
            )
        )
    input_json = row.get("input_json_path", "")
    if input_json and not Path(input_json).exists():
        issues.append(
            issue(
                row,
                severity="error",
                check="input_json_exists",
                path=input_json,
                message="planned DISCO input JSON is missing",
            )
        )
    return issues


def validate_summary_rows(
    *,
    plan_row: dict[str, str],
    summary_rows: list[dict[str, str]],
    require_confidence: bool,
    require_diagnostics: bool,
    require_fixed_preserved: bool,
) -> list[ValidationIssue]:
    """Validate completed summary rows against the plan row."""

    issues: list[ValidationIssue] = []
    system = plan_row.get("system", "")
    mode = plan_row.get("mode", "")
    summary_path = plan_row.get("summary_csv_path", "")
    parameters = parse_parameters(plan_row)
    expected_length = int(parameters.get("length", 0) or 0)

    for index, row in enumerate(summary_rows):
        row_path = f"{summary_path}#row{index + 1}"
        sequence = row.get("sequence", "")
        if not sequence:
            issues.append(
                issue(
                    plan_row,
                    severity="error",
                    check="sequence_present",
                    path=row_path,
                    message="summary row has no sequence",
                )
            )
        elif expected_length and len(sequence) != expected_length:
            issues.append(
                issue(
                    plan_row,
                    severity="error",
                    check="sequence_length",
                    path=row_path,
                    message=(
                        f"sequence length {len(sequence)} does not match planned "
                        f"length {expected_length}"
                    ),
                )
            )

        confidence = first_existing_value(
            row,
            ["confidence", "pg_plddt_mean", "mean_plddt", "plddt", "ranking_score"],
        )
        if require_confidence:
            if not confidence:
                issues.append(
                    issue(
                        plan_row,
                        severity="error",
                        check="confidence_present",
                        path=row_path,
                        message="no confidence/pLDDT-like field found",
                    )
                )
            elif not looks_float(confidence):
                issues.append(
                    issue(
                        plan_row,
                        severity="error",
                        check="confidence_numeric",
                        path=row_path,
                        message=f"confidence value is not numeric: {confidence!r}",
                    )
                )

        if system == "DISCO" and require_confidence:
            match_status = row.get("confidence_match_status", "")
            if match_status not in {"exact_sample", "single_confidence_fallback"}:
                issues.append(
                    issue(
                        plan_row,
                        severity="error",
                        check="disco_confidence_match",
                        path=row_path,
                        message=(
                            "DISCO confidence is not matched to the sequence sample: "
                            f"{match_status!r}"
                        ),
                    )
                )

        if system == "DISCO" and require_diagnostics:
            diagnostics_path = row.get("diagnostics_jsonl", "")
            if not diagnostics_path:
                issues.append(
                    issue(
                        plan_row,
                        severity="error",
                        check="diagnostics_present",
                        path=row_path,
                        message="DISCO diagnostics_jsonl field is missing",
                    )
                )
            elif not Path(diagnostics_path).exists():
                issues.append(
                    issue(
                        plan_row,
                        severity="error",
                        check="diagnostics_exists",
                        path=diagnostics_path,
                        message="DISCO diagnostics JSONL file is missing",
                    )
                )

        if mode == "immunogen" and require_fixed_preserved:
            fixed = row.get("fixed_fragments_preserved", "")
            if not truthy_string(fixed):
                issues.append(
                    issue(
                        plan_row,
                        severity="error",
                        check="fixed_fragments_preserved",
                        path=row_path,
                        message=(
                            "immunogen row does not prove fixed fragments were "
                            f"preserved: {fixed!r}"
                        ),
                    )
                )
    return issues


def validate_comparison_plan(
    *,
    planned_runs_csv: Path,
    allow_missing_summaries: bool = False,
    require_confidence: bool = True,
    require_diagnostics: bool = False,
    require_fixed_preserved: bool = True,
    required_systems: Iterable[str] | None = ("DISCO", "ProteinGenerator"),
) -> tuple[list[ValidationIssue], dict[str, object]]:
    """Validate a planned comparison matrix and any completed summaries."""

    plan_rows = read_csv_dicts(planned_runs_csv)
    issues: list[ValidationIssue] = []
    if not plan_rows:
        issues.append(
            ValidationIssue(
                severity="error",
                check="plan_nonempty",
                system="",
                mode="",
                variant="",
                replicate="",
                path=str(planned_runs_csv),
                message="planned_runs.csv has no planned rows",
            )
        )
    required_system_list = list(required_systems or [])
    issues.extend(validate_required_systems(plan_rows, required_system_list))
    issues.extend(validate_matrix_balance(plan_rows))
    issues.extend(validate_parameter_consistency(plan_rows))

    completed_summary_count = 0
    expected_summary_count = 0
    completed_design_rows = 0
    for row in plan_rows:
        status = row.get("status", "")
        if status != "planned":
            issues.append(
                issue(
                    row,
                    severity="error",
                    check="run_status",
                    path=row.get("run_dir", ""),
                    message=f"run is not executable/planned: {status}",
                )
            )
            continue

        issues.extend(validate_planned_artifacts(row))
        expected_summary_count += 1
        summary_path_value = row.get("summary_csv_path", "")
        if not summary_path_value:
            issues.append(
                issue(
                    row,
                    severity="error",
                    check="summary_path_present",
                    path=row.get("run_dir", ""),
                    message="planned row has no summary_csv_path",
                )
            )
            continue
        summary_path = Path(summary_path_value)
        if not summary_path.exists():
            issues.append(
                issue(
                    row,
                    severity="warning" if allow_missing_summaries else "error",
                    check="summary_exists",
                    path=summary_path,
                    message="planned summary CSV is missing",
                )
            )
            continue

        summary_rows = read_csv_dicts(summary_path)
        if not summary_rows:
            issues.append(
                issue(
                    row,
                    severity="error",
                    check="summary_nonempty",
                    path=summary_path,
                    message="summary CSV has no design rows",
                )
            )
            continue
        completed_summary_count += 1
        completed_design_rows += len(summary_rows)
        issues.extend(
            validate_summary_rows(
                plan_row=row,
                summary_rows=summary_rows,
                require_confidence=require_confidence,
                require_diagnostics=require_diagnostics,
                require_fixed_preserved=require_fixed_preserved,
            )
        )

    error_count = sum(1 for item in issues if item.severity == "error")
    warning_count = sum(1 for item in issues if item.severity == "warning")
    summary = {
        "planned_runs_csv": str(planned_runs_csv),
        "num_planned_rows": len(plan_rows),
        "expected_summary_count": expected_summary_count,
        "completed_summary_count": completed_summary_count,
        "completed_design_rows": completed_design_rows,
        "required_systems": required_system_list,
        "error_count": error_count,
        "warning_count": warning_count,
        "status": "pass" if error_count == 0 else "fail",
    }
    return issues, summary


def write_validation_report(
    *,
    report_json_path: Path,
    issues_csv_path: Path,
    issues: list[ValidationIssue],
    summary: dict[str, object],
) -> None:
    """Write validation summary JSON and issue CSV."""

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    issues_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(
            {
                **summary,
                "issues": [item.as_dict() for item in issues],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_issue_csv(issues_csv_path, issues)
