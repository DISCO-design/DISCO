"""Merge optional external quality metrics into motif-design summaries.

ProteinGenerator's post-hoc evaluator could run ESM-2, RMSD, and Rosetta
backends directly.  DISCO should not assume those optional dependencies exist
on a laptop or in this repository, so this module implements the reusable
comparison layer instead: consume an existing summary CSV, merge externally
produced metric rows, compute rank-normalized component scores, and write an
augmented table plus a compact report.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExternalQualityWeights:
    """Weights matching the documented ProteinGenerator post-hoc ranker."""

    confidence: float = 0.30
    esm2: float = 0.20
    rmsd: float = 0.20
    rosetta: float = 0.20
    motif: float = 0.10


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    priority = [
        "design_id",
        "system",
        "mode",
        "variant",
        "sequence",
        "confidence",
        "motif_score",
        "comparison_score",
        "esm2_mean_token_probability",
        "esm2_perplexity",
        "ca_rmsd_to_reference",
        "ca_rmsd_to_seed",
        "rosetta_score_per_residue",
        "esm2_rank_score",
        "structure_consistency_score",
        "rosetta_rank_score",
        "posthoc_quality_score",
        "external_metric_status",
        "external_metric_missing",
    ]
    ordered = [key for key in priority if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def optional_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    return value_float if math.isfinite(value_float) else None


def first_float(row: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = optional_float(row.get(key, ""))
        if value is not None:
            return value
    return None


def first_value(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def percentile_rank(values: list[float | None], value: float | None, *, higher_is_better: bool) -> float:
    """Return a 0..1 percentile-like score, matching the PG evaluator style."""

    clean = sorted(item for item in values if item is not None and math.isfinite(item))
    if not clean or value is None or not math.isfinite(value):
        return 0.0
    count = sum(1 for item in clean if item <= value)
    rank = count / len(clean)
    return rank if higher_is_better else 1.0 - rank + (1.0 / len(clean))


def metric_key(row: dict[str, Any], key_field: str) -> str:
    return first_value(row, [key_field, "design_id", "base_id", "candidate_id"])


def merge_external_rows(
    summary_rows: list[dict[str, str]],
    external_rows: list[dict[str, str]],
    *,
    key_field: str = "design_id",
) -> list[dict[str, Any]]:
    """Merge external metric columns into summary rows by design id."""

    external_by_key = {
        metric_key(row, key_field): row for row in external_rows if metric_key(row, key_field)
    }
    merged: list[dict[str, Any]] = []
    for summary in summary_rows:
        key = metric_key(summary, key_field)
        row: dict[str, Any] = dict(summary)
        external = external_by_key.get(key, {})
        for metric in [
            "esm2_status",
            "esm2_mean_token_probability",
            "esm2_perplexity",
            "ca_rmsd_to_reference",
            "ca_rmsd_to_seed",
            "rmsd_status",
            "rosetta_status",
            "rosetta_score",
            "rosetta_score_per_residue",
            "external_structure_path",
            "reference_pdb_path",
        ]:
            if external.get(metric, "") not in ["", None]:
                row[metric] = external[metric]
        merged.append(row)
    return merged


def motif_score_for_row(row: dict[str, Any]) -> float:
    score = first_float(row, ["motif_score", "motif_coverage"])
    if score is not None:
        return score
    unique = first_float(row, ["unique_motifs", "num_unique_motifs_present"])
    required = first_float(row, ["required_motifs", "num_required_motifs"])
    if unique is not None and required not in [None, 0.0]:
        return unique / max(required or 1.0, 1.0)
    return 0.0


def confidence_for_row(row: dict[str, Any]) -> float:
    return first_float(row, ["confidence", "pg_plddt_mean", "mean_plddt", "plddt"]) or 0.0


def annotate_external_quality_rows(
    rows: list[dict[str, Any]],
    *,
    weights: ExternalQualityWeights = ExternalQualityWeights(),
) -> list[dict[str, Any]]:
    """Add rank component fields and `posthoc_quality_score` to rows."""

    esm_prob_values = [
        first_float(row, ["esm2_mean_token_probability"]) for row in rows
    ]
    esm_perplexity_values = [first_float(row, ["esm2_perplexity"]) for row in rows]
    rmsd_values = [
        first_float(row, ["ca_rmsd_to_reference", "ca_rmsd_to_seed"]) for row in rows
    ]
    rosetta_values = [
        first_float(row, ["rosetta_score_per_residue", "rosetta_score"])
        for row in rows
    ]

    annotated: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        esm_prob = first_float(current, ["esm2_mean_token_probability"])
        esm_perplexity = first_float(current, ["esm2_perplexity"])
        if esm_prob is not None:
            esm_score = percentile_rank(
                esm_prob_values,
                esm_prob,
                higher_is_better=True,
            )
            esm_source = "esm2_mean_token_probability"
        else:
            esm_score = percentile_rank(
                esm_perplexity_values,
                esm_perplexity,
                higher_is_better=False,
            )
            esm_source = "esm2_perplexity" if esm_perplexity is not None else ""

        rmsd = first_float(current, ["ca_rmsd_to_reference", "ca_rmsd_to_seed"])
        rosetta = first_float(current, ["rosetta_score_per_residue", "rosetta_score"])
        rmsd_score = percentile_rank(rmsd_values, rmsd, higher_is_better=False)
        rosetta_score = percentile_rank(
            rosetta_values,
            rosetta,
            higher_is_better=False,
        )
        confidence = confidence_for_row(current)
        motif_score = motif_score_for_row(current)
        posthoc_score = (
            weights.confidence * confidence
            + weights.esm2 * esm_score
            + weights.rmsd * rmsd_score
            + weights.rosetta * rosetta_score
            + weights.motif * motif_score
        )

        missing = []
        if not esm_source:
            missing.append("esm2")
        if rmsd is None:
            missing.append("rmsd")
        if rosetta is None:
            missing.append("rosetta")

        current["esm2_rank_score"] = f"{esm_score:.6f}"
        current["esm2_rank_source"] = esm_source
        current["structure_consistency_score"] = f"{rmsd_score:.6f}"
        current["rosetta_rank_score"] = f"{rosetta_score:.6f}"
        current["posthoc_quality_score"] = f"{posthoc_score:.6f}"
        current["posthoc_score_weights_json"] = json.dumps(
            asdict(weights),
            sort_keys=True,
        )
        current["external_metric_missing"] = ",".join(missing)
        if not missing:
            current["external_metric_status"] = "complete"
        elif len(missing) == 3:
            current["external_metric_status"] = "no_external_metrics"
        else:
            current["external_metric_status"] = "partial"
        annotated.append(current)
    return annotated


def summarize_external_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact report for augmented post-hoc rows."""

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("external_metric_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    scored = [
        (optional_float(row.get("posthoc_quality_score")), str(row.get("design_id", "")))
        for row in rows
    ]
    scored = [(score, design_id) for score, design_id in scored if score is not None]
    best_score, best_design = (None, "") if not scored else max(scored, key=lambda item: item[0])
    return {
        "num_rows": len(rows),
        "status_counts": status_counts,
        "best_posthoc_quality_score": (
            "" if best_score is None else f"{best_score:.6f}"
        ),
        "best_posthoc_quality_design_id": best_design,
        "metric_columns": [
            "esm2_mean_token_probability",
            "esm2_perplexity",
            "ca_rmsd_to_reference",
            "ca_rmsd_to_seed",
            "rosetta_score_per_residue",
            "rosetta_score",
        ],
    }


def write_external_quality_outputs(
    *,
    summary_csv: Path,
    out_csv: Path,
    report_json: Path,
    external_metrics_csv: Path | None = None,
    top_csv: Path | None = None,
    key_field: str = "design_id",
    weights: ExternalQualityWeights = ExternalQualityWeights(),
) -> dict[str, Any]:
    """Merge external metrics, compute scores, and write CSV/JSON outputs."""

    summary_rows = read_csv_rows(summary_csv)
    external_rows = read_csv_rows(external_metrics_csv) if external_metrics_csv else []
    merged = merge_external_rows(
        summary_rows,
        external_rows,
        key_field=key_field,
    )
    annotated = annotate_external_quality_rows(merged, weights=weights)
    write_csv_rows(out_csv, annotated)

    sorted_rows = sorted(
        annotated,
        key=lambda row: optional_float(row.get("posthoc_quality_score")) or float("-inf"),
        reverse=True,
    )
    if top_csv is not None:
        write_csv_rows(top_csv, sorted_rows[:100])

    report = {
        **summarize_external_quality(annotated),
        "summary_csv": str(summary_csv),
        "external_metrics_csv": "" if external_metrics_csv is None else str(external_metrics_csv),
        "out_csv": str(out_csv),
        "top_csv": "" if top_csv is None else str(top_csv),
        "key_field": key_field,
        "weights": asdict(weights),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
