"""Markdown reporting for PG-vs-DISCO comparison outputs.

This module intentionally consumes already-normalized CSVs and optional
validation JSON.  It does not launch inference and it does not alter metrics.
Its job is to make score provenance, validation status, and top designs easy to
inspect before anyone draws scientific conclusions.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_validation_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value: str | float, digits: int = 3) -> str:
    numeric = as_float(value)
    return f"{numeric:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def top_design_rows(
    rows: list[dict[str, str]],
    *,
    metric: str,
    limit: int = 5,
) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: as_float(row.get(metric, "")), reverse=True)[
        :limit
    ]


def render_comparison_report(
    *,
    normalized_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    validation_report: dict[str, object] | None = None,
    title: str = "PG-vs-DISCO Motif Comparison Report",
) -> str:
    """Render a compact Markdown report from comparison CSV rows."""

    lines: list[str] = [f"# {title}", ""]

    if validation_report is None:
        lines.extend(
            [
                "## Validation",
                "",
                "No validation report was supplied.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Validation",
                "",
                (
                    f"Status: `{validation_report.get('status', 'unknown')}`; "
                    f"errors: {validation_report.get('error_count', '')}; "
                    f"warnings: {validation_report.get('warning_count', '')}; "
                    f"completed summaries: "
                    f"{validation_report.get('completed_summary_count', '')}/"
                    f"{validation_report.get('expected_summary_count', '')}."
                ),
                "",
            ]
        )

    lines.extend(["## Aggregate Results", ""])
    if summary_rows:
        aggregate = []
        for row in summary_rows:
            aggregate.append(
                [
                    row.get("system", ""),
                    row.get("mode", ""),
                    row.get("variant", ""),
                    row.get("num_designs", ""),
                    format_float(row.get("mean_confidence", "")),
                    format_float(row.get("best_confidence", "")),
                    format_float(row.get("mean_motif_score", "")),
                    format_float(row.get("best_comparison_score", "")),
                    row.get("best_comparison_design_id", ""),
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "system",
                    "mode",
                    "variant",
                    "n",
                    "mean_conf",
                    "best_conf",
                    "mean_motif",
                    "best_score",
                    "best_score_design",
                ],
                aggregate,
            )
        )
    else:
        lines.append("No summary rows were supplied.")
    lines.append("")

    lines.extend(["## Top Designs", ""])
    top_by_score = top_design_rows(
        normalized_rows,
        metric="comparison_score",
        limit=5,
    )
    if top_by_score:
        lines.extend(
            markdown_table(
                [
                    "rank",
                    "system",
                    "mode",
                    "variant",
                    "design",
                    "comparison_score",
                    "confidence",
                    "motif_score",
                    "provenance",
                ],
                [
                    [
                        index + 1,
                        row.get("system", ""),
                        row.get("mode", ""),
                        row.get("variant", ""),
                        row.get("design_id", ""),
                        format_float(row.get("comparison_score", "")),
                        format_float(row.get("confidence", "")),
                        format_float(row.get("motif_score", "")),
                        row.get("score_provenance_status", ""),
                    ]
                    for index, row in enumerate(top_by_score)
                ],
            )
        )
    else:
        lines.append("No normalized design rows were supplied.")
    lines.append("")

    provenance_counts = Counter(
        row.get("score_provenance_status", "") for row in normalized_rows
    )
    lines.extend(["## Provenance", ""])
    if provenance_counts:
        lines.extend(
            markdown_table(
                ["score_provenance_status", "count"],
                [
                    [status or "missing", count]
                    for status, count in sorted(provenance_counts.items())
                ],
            )
        )
    else:
        lines.append("No provenance rows were supplied.")
    lines.append("")

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- `comparison_score` is a transparent first-pass ranking score: 0.5 confidence + 0.5 motif_score.",
            "- Immunogen rows with failed fixed-fragment preservation receive comparison score 0.",
            "- DISCO confidence is only cleanly comparable when `confidence_match_status` is `exact_sample` or an explicitly accepted fallback.",
            "- This report summarizes existing CSVs; it does not prove that model budgets, hardware, or sample sizes were scientifically sufficient.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_report(
    *,
    normalized_csv: Path,
    summary_csv: Path,
    out_md: Path,
    validation_report_json: Path | None = None,
    title: str = "PG-vs-DISCO Motif Comparison Report",
) -> None:
    """Write a Markdown comparison report from existing CSV/JSON artifacts."""

    report = render_comparison_report(
        normalized_rows=read_csv_dicts(normalized_csv),
        summary_rows=read_csv_dicts(summary_csv),
        validation_report=read_validation_json(validation_report_json),
        title=title,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report + "\n")
