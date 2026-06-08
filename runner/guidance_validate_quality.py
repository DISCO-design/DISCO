#!/usr/bin/env python3
"""Validate completed motif-design summary CSVs against quality gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.quality_validation import (
    QualityGateConfig,
    validate_summary_csv,
    write_quality_report,
)


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "skip", ""}:
        return None
    return float(value)


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "skip", ""}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-report-json", required=True, type=Path)
    parser.add_argument("--out-issues-csv", required=True, type=Path)
    parser.add_argument("--min-confidence", type=parse_optional_float, default=0.60)
    parser.add_argument(
        "--min-motif-coverage",
        type=parse_optional_float,
        default=1.0,
    )
    parser.add_argument("--min-unique-motifs", type=parse_optional_int, default=1)
    parser.add_argument("--max-homopolymer", type=parse_optional_int, default=4)
    parser.add_argument("--min-entropy", type=parse_optional_float, default=0.50)
    parser.add_argument("--max-rmsd", type=parse_optional_float)
    parser.add_argument("--min-spread-score", type=parse_optional_float)
    parser.add_argument("--require-fixed-preserved", action="store_true")
    parser.add_argument("--require-exact-confidence", action="store_true")
    parser.add_argument("--allow-confidence-fallback", action="store_true")
    parser.add_argument("--require-diagnostics", action="store_true")
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--require-same-evaluation-artifact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = QualityGateConfig(
        min_confidence=args.min_confidence,
        min_motif_coverage=args.min_motif_coverage,
        min_unique_motifs=args.min_unique_motifs,
        max_homopolymer=args.max_homopolymer,
        min_entropy=args.min_entropy,
        max_rmsd=args.max_rmsd,
        min_spread_score=args.min_spread_score,
        require_fixed_preserved=args.require_fixed_preserved,
        require_exact_confidence=args.require_exact_confidence,
        allow_confidence_fallback=args.allow_confidence_fallback,
        require_diagnostics=args.require_diagnostics,
        require_structure=args.require_structure,
        require_same_evaluation_artifact=args.require_same_evaluation_artifact,
    )
    issues, summary = validate_summary_csv(
        summary_csv=args.summary_csv,
        config=config,
    )
    write_quality_report(
        report_json_path=args.out_report_json,
        issues_csv_path=args.out_issues_csv,
        issues=issues,
        summary=summary,
    )
    print(
        "Quality validation {status}: {errors} error(s), {warnings} warning(s), "
        "{passed}/{rows} passed rows".format(
            status=summary["status"],
            errors=summary["error_count"],
            warnings=summary["warning_count"],
            passed=summary["passed_rows"],
            rows=summary["num_rows"],
        )
    )
    print(f"Report JSON: {args.out_report_json}")
    print(f"Issues CSV: {args.out_issues_csv}")
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
