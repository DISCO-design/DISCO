"""Category-specific validation for Immunogen motif-design summaries.

The generic quality validator checks confidence, low-complexity guards, and
basic motif coverage.  Immunogen experiments need stricter category semantics:
protected B-cell/fixed fragments must remain unchanged, T-cell motifs should be
covered as their own group, and cleavage motifs should be covered as their own
group.  This module validates those category-level requirements on completed
CSV rows; it never edits designs and never launches inference.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from disco.model.cogen_inference.guidance.alternating import parse_motif_list
from disco.model.cogen_inference.guidance.evaluation_provenance import (
    row_has_accepted_same_artifact_provenance,
)
from disco.model.cogen_inference.guidance.quality_validation import (
    as_int,
    first_value,
    read_csv_dicts,
    truthy,
)


@dataclass(frozen=True)
class ImmunogenValidationConfig:
    """Category gates for one Immunogen validation run."""

    tcell_motifs: list[str] = field(default_factory=list)
    cleavage_motifs: list[str] = field(default_factory=list)
    require_fixed_preserved: bool = True
    min_tcell_coverage: float | None = 1.0
    min_cleavage_coverage: float | None = 1.0
    min_tcell_total: int | None = None
    min_cleavage_total: int | None = None
    require_same_evaluation_artifact: bool = False


@dataclass(frozen=True)
class ImmunogenValidationIssue:
    """One Immunogen category validation finding."""

    severity: str
    category: str
    check: str
    design_id: str
    row_index: int
    path: str
    observed: str
    threshold: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def as_float(value: str, default: float | None = None) -> float | None:
    if value in ["", None]:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def issue(
    row: dict[str, str],
    *,
    row_index: int,
    source_path: Path,
    category: str,
    check: str,
    observed: object,
    threshold: object,
    message: str,
    severity: str = "error",
) -> ImmunogenValidationIssue:
    return ImmunogenValidationIssue(
        severity=severity,
        category=category,
        check=check,
        design_id=row.get("design_id", row.get("individual", "")),
        row_index=row_index,
        path=str(source_path),
        observed="" if observed is None else str(observed),
        threshold="" if threshold is None else str(threshold),
        message=message,
    )


def split_motifs(value: str) -> list[str]:
    return parse_motif_list(value)


def category_coverage(
    row: dict[str, str],
    *,
    prefix: str,
    motifs: list[str],
) -> tuple[float | None, int | None, int | None, list[str]]:
    """Return coverage, unique count, total count, and missing motifs."""

    required = len(set(motifs))
    unique = as_int(row.get(f"{prefix}_unique_motifs", ""), default=None)
    total = as_int(row.get(f"{prefix}_total_motifs", ""), default=None)
    missing = split_motifs(row.get(f"{prefix}_missing_motifs", ""))
    if required == 0:
        return None, unique, total, missing
    if missing:
        coverage = max(0.0, (required - len(set(missing))) / required)
    elif unique is not None:
        coverage = min(1.0, unique / required)
    else:
        coverage = None
    return coverage, unique, total, missing


def validate_fixed_fragment_category(
    row: dict[str, str],
    *,
    config: ImmunogenValidationConfig,
    row_index: int,
    source_path: Path,
) -> list[ImmunogenValidationIssue]:
    findings: list[ImmunogenValidationIssue] = []
    if not config.require_fixed_preserved:
        return findings

    fixed = first_value(row, ["fixed_fragments_preserved", "fixed_preserved"])
    mismatches = as_int(row.get("num_fixed_mismatches", ""), default=0)
    if not truthy(fixed):
        findings.append(
            issue(
                row,
                row_index=row_index,
                source_path=source_path,
                category="bcell_fixed",
                check="fixed_fragments_preserved",
                observed=fixed,
                threshold=True,
                message="protected B-cell/fixed fragments were not proven preserved",
            )
        )
    if mismatches not in {None, 0}:
        findings.append(
            issue(
                row,
                row_index=row_index,
                source_path=source_path,
                category="bcell_fixed",
                check="fixed_fragment_mismatches",
                observed=mismatches,
                threshold=0,
                message=f"fixed mismatch count is {mismatches}",
            )
        )
    return findings


def validate_motif_category(
    row: dict[str, str],
    *,
    prefix: str,
    category: str,
    motifs: list[str],
    min_coverage: float | None,
    min_total: int | None,
    row_index: int,
    source_path: Path,
) -> list[ImmunogenValidationIssue]:
    findings: list[ImmunogenValidationIssue] = []
    if not motifs:
        return findings

    coverage, unique, total, missing = category_coverage(
        row,
        prefix=prefix,
        motifs=motifs,
    )
    if min_coverage is not None:
        if coverage is None:
            findings.append(
                issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    category=category,
                    check=f"{prefix}_coverage_present",
                    observed="",
                    threshold=min_coverage,
                    message=f"{category} coverage cannot be derived from row fields",
                )
            )
        elif coverage < min_coverage:
            findings.append(
                issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    category=category,
                    check=f"{prefix}_coverage",
                    observed=f"{coverage:.6f}",
                    threshold=min_coverage,
                    message=(
                        f"{category} coverage {coverage:.3f} < "
                        f"{min_coverage:.3f}; missing={','.join(missing)}"
                    ),
                )
            )

    if min_total is not None:
        if total is None:
            findings.append(
                issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    category=category,
                    check=f"{prefix}_total_present",
                    observed="",
                    threshold=min_total,
                    message=f"{category} total count is missing",
                )
            )
        elif total < min_total:
            findings.append(
                issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    category=category,
                    check=f"{prefix}_total",
                    observed=total,
                    threshold=min_total,
                    message=f"{category} total motif count {total} is too low",
                )
            )

    return findings


def validate_immunogen_row(
    row: dict[str, str],
    *,
    config: ImmunogenValidationConfig,
    row_index: int,
    source_path: Path,
) -> list[ImmunogenValidationIssue]:
    """Validate one completed Immunogen row against category gates."""

    findings = validate_fixed_fragment_category(
        row,
        config=config,
        row_index=row_index,
        source_path=source_path,
    )
    findings.extend(
        validate_motif_category(
            row,
            prefix="tcell",
            category="tcell",
            motifs=config.tcell_motifs,
            min_coverage=config.min_tcell_coverage,
            min_total=config.min_tcell_total,
            row_index=row_index,
            source_path=source_path,
        )
    )
    findings.extend(
        validate_motif_category(
            row,
            prefix="cleavage",
            category="cleavage",
            motifs=config.cleavage_motifs,
            min_coverage=config.min_cleavage_coverage,
            min_total=config.min_cleavage_total,
            row_index=row_index,
            source_path=source_path,
        )
    )
    if config.require_same_evaluation_artifact:
        if not row_has_accepted_same_artifact_provenance(row):
            observed = first_value(
                row,
                ["evaluation_status", "score_provenance_status"],
            )
            findings.append(
                issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    category="evaluation",
                    check="same_evaluation_artifact",
                    observed=observed,
                    threshold="same-sample or same-PDB provenance",
                    message=(
                        "sequence, structure, and confidence/pLDDT are not "
                        "proven to come from the same evaluated artifact"
                    ),
                )
            )
    return findings


def category_passed(
    rows: list[dict[str, str]],
    issues: list[ImmunogenValidationIssue],
    category: str,
) -> int:
    failed_indices = {
        item.row_index
        for item in issues
        if item.category == category and item.severity == "error"
    }
    return len(rows) - len(failed_indices)


def validate_immunogen_summary_csv(
    *,
    summary_csv: Path,
    config: ImmunogenValidationConfig,
) -> tuple[list[ImmunogenValidationIssue], dict[str, object]]:
    """Validate every row in one Immunogen summary CSV."""

    rows = read_csv_dicts(summary_csv)
    findings: list[ImmunogenValidationIssue] = []
    if not rows:
        findings.append(
            ImmunogenValidationIssue(
                severity="error",
                category="summary",
                check="summary_nonempty",
                design_id="",
                row_index=0,
                path=str(summary_csv),
                observed="0",
                threshold=">0",
                message="summary CSV has no rows",
            )
        )
    for row_index, row in enumerate(rows, start=1):
        findings.extend(
            validate_immunogen_row(
                row,
                config=config,
                row_index=row_index,
                source_path=summary_csv,
            )
        )

    error_count = sum(1 for item in findings if item.severity == "error")
    failed_row_indices = {
        item.row_index for item in findings if item.row_index > 0
    }
    summary = {
        "summary_csv": str(summary_csv),
        "num_rows": len(rows),
        "error_count": error_count,
        "warning_count": sum(1 for item in findings if item.severity == "warning"),
        "passed_rows": len(rows) - len(failed_row_indices),
        "fixed_passed_rows": category_passed(rows, findings, "bcell_fixed"),
        "tcell_passed_rows": category_passed(rows, findings, "tcell"),
        "cleavage_passed_rows": category_passed(rows, findings, "cleavage"),
        "evaluation_passed_rows": category_passed(rows, findings, "evaluation"),
        "status": "pass" if error_count == 0 else "fail",
        "config": asdict(config),
    }
    return findings, summary


def write_immunogen_issues_csv(
    path: Path,
    issues: list[ImmunogenValidationIssue],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "severity",
        "category",
        "check",
        "design_id",
        "row_index",
        "path",
        "observed",
        "threshold",
        "message",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([item.as_dict() for item in issues])


def write_immunogen_validation_report(
    *,
    report_json_path: Path,
    issues_csv_path: Path,
    issues: list[ImmunogenValidationIssue],
    summary: dict[str, object],
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
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
    write_immunogen_issues_csv(issues_csv_path, issues)
