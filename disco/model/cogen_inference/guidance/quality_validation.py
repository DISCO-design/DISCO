"""Quality gates for completed motif-design summary CSVs.

This module ports the ProteinGenerator Immunogen validation idea to the shared
PG-vs-DISCO analysis layer.  It consumes completed summary or normalized CSVs
and reports structured gate failures; it never edits designs and never launches
model inference.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from disco.model.cogen_inference.guidance.evaluation_provenance import (
    row_has_accepted_same_artifact_provenance,
)


@dataclass(frozen=True)
class QualityGateConfig:
    """Thresholds and required-provenance switches for one validation run."""

    min_confidence: float | None = 0.60
    min_motif_coverage: float | None = 1.0
    min_unique_motifs: int | None = 1
    max_homopolymer: int | None = 4
    min_entropy: float | None = 0.50
    max_rmsd: float | None = None
    min_spread_score: float | None = None
    require_fixed_preserved: bool = False
    require_exact_confidence: bool = False
    allow_confidence_fallback: bool = False
    require_diagnostics: bool = False
    require_structure: bool = False
    require_same_evaluation_artifact: bool = False


@dataclass(frozen=True)
class QualityIssue:
    """One quality-gate finding for one design row."""

    severity: str
    check: str
    design_id: str
    row_index: int
    path: str
    observed: str
    threshold: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def first_value(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ["", None]:
            return str(value)
    return ""


def as_float(value: str, default: float | None = None) -> float | None:
    if value in ["", None]:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int | None = None) -> int | None:
    if value in ["", None]:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def split_list(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def count_fields(row: dict[str, str]) -> list[int]:
    fields = [
        key
        for key in row
        if key.endswith("_count") and key[: -len("_count")].isupper()
    ]
    counts = []
    for key in fields:
        value = as_int(row.get(key, ""), default=None)
        if value is not None:
            counts.append(value)
    return counts


def motif_coverage(row: dict[str, str]) -> float | None:
    explicit = first_value(row, ["motif_coverage", "motif_coverage_fraction"])
    if explicit:
        return as_float(explicit)

    present = split_list(first_value(row, ["present_motifs", "motifs_present"]))
    missing = split_list(first_value(row, ["missing_motifs", "motifs_missing"]))
    required = len(set(present + missing))
    if required:
        return len(set(present)) / required

    counts = count_fields(row)
    if counts:
        return sum(1 for count in counts if count > 0) / len(counts)

    return None


def unique_motifs(row: dict[str, str]) -> int | None:
    explicit = first_value(row, ["unique_motifs", "num_unique_motifs_present"])
    if explicit:
        return as_int(explicit)
    counts = count_fields(row)
    if counts:
        return sum(1 for count in counts if count > 0)
    return None


def row_issue(
    row: dict[str, str],
    *,
    row_index: int,
    source_path: Path,
    check: str,
    observed: object,
    threshold: object,
    message: str,
    severity: str = "error",
) -> QualityIssue:
    return QualityIssue(
        severity=severity,
        check=check,
        design_id=row.get("design_id", row.get("individual", "")),
        row_index=row_index,
        path=str(source_path),
        observed="" if observed is None else str(observed),
        threshold="" if threshold is None else str(threshold),
        message=message,
    )


def validate_design_row(
    row: dict[str, str],
    *,
    config: QualityGateConfig,
    row_index: int,
    source_path: Path,
) -> list[QualityIssue]:
    """Validate one completed design summary row against quality gates."""

    issues: list[QualityIssue] = []
    confidence = as_float(
        first_value(row, ["confidence", "pg_plddt_mean", "mean_plddt", "plddt"]),
        default=None,
    )
    if config.min_confidence is not None:
        if confidence is None:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="confidence_present",
                    observed="",
                    threshold=config.min_confidence,
                    message="confidence/pLDDT field is missing",
                )
            )
        elif confidence < config.min_confidence:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="min_confidence",
                    observed=f"{confidence:.6f}",
                    threshold=config.min_confidence,
                    message=(
                        f"confidence {confidence:.3f} < "
                        f"{config.min_confidence:.3f}"
                    ),
                )
            )

    if config.require_exact_confidence:
        status = first_value(row, ["confidence_match_status"])
        exact_statuses = {"exact_sample", "same_pdb_ca_bfactor"}
        fallback_statuses = {"single_confidence_fallback"}
        allowed = set(exact_statuses)
        if config.allow_confidence_fallback:
            allowed.update(fallback_statuses)
        if status not in allowed:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="exact_confidence_provenance",
                    observed=status,
                    threshold=",".join(sorted(allowed)),
                    message="confidence provenance is not accepted for this gate",
                )
            )

    coverage = motif_coverage(row)
    if config.min_motif_coverage is not None:
        if coverage is None:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="motif_coverage_present",
                    observed="",
                    threshold=config.min_motif_coverage,
                    message="motif coverage cannot be derived from row fields",
                )
            )
        elif coverage < config.min_motif_coverage:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="min_motif_coverage",
                    observed=f"{coverage:.6f}",
                    threshold=config.min_motif_coverage,
                    message=(
                        f"motif coverage {coverage:.3f} < "
                        f"{config.min_motif_coverage:.3f}"
                    ),
                )
            )

    unique = unique_motifs(row)
    if config.min_unique_motifs is not None:
        if unique is None:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="unique_motifs_present",
                    observed="",
                    threshold=config.min_unique_motifs,
                    message="unique motif count cannot be derived from row fields",
                )
            )
        elif unique < config.min_unique_motifs:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="min_unique_motifs",
                    observed=unique,
                    threshold=config.min_unique_motifs,
                    message=f"unique motif count {unique} is too low",
                )
            )

    if config.require_fixed_preserved:
        fixed = first_value(row, ["fixed_fragments_preserved", "fixed_preserved"])
        mismatches = as_int(row.get("num_fixed_mismatches", ""), default=0)
        if not truthy(fixed):
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="fixed_fragments_preserved",
                    observed=fixed,
                    threshold=True,
                    message="fixed fragments were not proven preserved",
                )
            )
        if mismatches not in {None, 0}:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="fixed_fragment_mismatches",
                    observed=mismatches,
                    threshold=0,
                    message=f"fixed mismatch count is {mismatches}",
                )
            )

    longest = as_int(row.get("longest_homopolymer", ""), default=None)
    if config.max_homopolymer is not None:
        if longest is None:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="homopolymer_present",
                    observed="",
                    threshold=config.max_homopolymer,
                    message="longest_homopolymer field is missing",
                )
            )
        elif longest > config.max_homopolymer:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="max_homopolymer",
                    observed=longest,
                    threshold=config.max_homopolymer,
                    message=f"longest homopolymer {longest} is too high",
                )
            )

    entropy = as_float(row.get("aa_entropy_norm", ""), default=None)
    if config.min_entropy is not None:
        if entropy is None:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="entropy_present",
                    observed="",
                    threshold=config.min_entropy,
                    message="aa_entropy_norm field is missing",
                )
            )
        elif entropy < config.min_entropy:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="min_entropy",
                    observed=f"{entropy:.6f}",
                    threshold=config.min_entropy,
                    message=f"entropy {entropy:.3f} is too low",
                )
            )

    rmsd = as_float(row.get("ca_rmsd_to_seed", ""), default=None)
    if config.max_rmsd is not None and rmsd is not None and rmsd > config.max_rmsd:
        issues.append(
            row_issue(
                row,
                row_index=row_index,
                source_path=source_path,
                check="max_rmsd",
                observed=f"{rmsd:.6f}",
                threshold=config.max_rmsd,
                message=f"CA-RMSD {rmsd:.3f} is too high",
            )
        )

    spread = as_float(row.get("motif_spread_score", ""), default=None)
    if (
        config.min_spread_score is not None
        and spread is not None
        and spread < config.min_spread_score
    ):
        issues.append(
            row_issue(
                row,
                row_index=row_index,
                source_path=source_path,
                check="min_spread_score",
                observed=f"{spread:.6f}",
                threshold=config.min_spread_score,
                message=f"motif spread score {spread:.3f} is too low",
            )
        )

    if config.require_diagnostics:
        diagnostics_path = row.get("diagnostics_jsonl", "")
        if not diagnostics_path:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="diagnostics_present",
                    observed="",
                    threshold="path",
                    message="diagnostics_jsonl field is missing",
                )
            )
        elif not Path(diagnostics_path).exists():
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="diagnostics_exists",
                    observed=diagnostics_path,
                    threshold="existing file",
                    message="diagnostics JSONL file does not exist",
                )
            )

    if config.require_structure:
        structure_path = row.get("structure_path", "")
        if not structure_path:
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="structure_present",
                    observed="",
                    threshold="path",
                    message="structure_path field is missing",
                )
            )
        elif not Path(structure_path).exists():
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="structure_exists",
                    observed=structure_path,
                    threshold="existing file",
                    message="structure file does not exist",
                )
            )

    if config.require_same_evaluation_artifact:
        if not row_has_accepted_same_artifact_provenance(row):
            observed = first_value(
                row,
                ["evaluation_status", "score_provenance_status"],
            )
            issues.append(
                row_issue(
                    row,
                    row_index=row_index,
                    source_path=source_path,
                    check="same_evaluation_artifact",
                    observed=observed,
                    threshold=(
                        "same_sample_complete,same_sample_no_diagnostics,"
                        "pg_sequence_and_plddt_from_same_saved_pdb,"
                        "disco_exact_sequence_structure_confidence_match"
                    ),
                    message=(
                        "sequence, structure, and confidence/pLDDT are not "
                        "proven to come from the same evaluated artifact"
                    ),
                )
            )

    return issues


def validate_summary_csv(
    *,
    summary_csv: Path,
    config: QualityGateConfig,
) -> tuple[list[QualityIssue], dict[str, object]]:
    """Validate every row in one completed summary CSV."""

    rows = read_csv_dicts(summary_csv)
    issues: list[QualityIssue] = []
    if not rows:
        issues.append(
            QualityIssue(
                severity="error",
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
        issues.extend(
            validate_design_row(
                row,
                config=config,
                row_index=row_index,
                source_path=summary_csv,
            )
        )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    failed_row_indices = {
        issue.row_index for issue in issues if issue.row_index > 0
    }
    summary = {
        "summary_csv": str(summary_csv),
        "num_rows": len(rows),
        "error_count": error_count,
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
        "passed_rows": len(rows) - len(failed_row_indices),
        "status": "pass" if error_count == 0 else "fail",
        "config": asdict(config),
    }
    return issues, summary


def write_quality_issues_csv(path: Path, issues: list[QualityIssue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "severity",
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
        writer.writerows([issue.as_dict() for issue in issues])


def write_quality_report(
    *,
    report_json_path: Path,
    issues_csv_path: Path,
    issues: list[QualityIssue],
    summary: dict[str, object],
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(
            {
                **summary,
                "issues": [issue.as_dict() for issue in issues],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_quality_issues_csv(issues_csv_path, issues)
