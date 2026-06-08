#!/usr/bin/env python3
"""Validate completed Immunogen summary CSVs against category gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.immunogen_validation import (
    ImmunogenValidationConfig,
    validate_immunogen_summary_csv,
    write_immunogen_validation_report,
)


def parse_motifs(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip().upper() for part in value.split(",") if part.strip()]


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
    parser.add_argument("--tcell-motifs")
    parser.add_argument("--cleavage-motifs")
    parser.add_argument(
        "--require-fixed-preserved",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-tcell-coverage", type=parse_optional_float, default=1.0)
    parser.add_argument(
        "--min-cleavage-coverage",
        type=parse_optional_float,
        default=1.0,
    )
    parser.add_argument("--min-tcell-total", type=parse_optional_int)
    parser.add_argument("--min-cleavage-total", type=parse_optional_int)
    parser.add_argument("--require-same-evaluation-artifact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ImmunogenValidationConfig(
        tcell_motifs=parse_motifs(args.tcell_motifs),
        cleavage_motifs=parse_motifs(args.cleavage_motifs),
        require_fixed_preserved=args.require_fixed_preserved,
        min_tcell_coverage=args.min_tcell_coverage,
        min_cleavage_coverage=args.min_cleavage_coverage,
        min_tcell_total=args.min_tcell_total,
        min_cleavage_total=args.min_cleavage_total,
        require_same_evaluation_artifact=args.require_same_evaluation_artifact,
    )
    issues, summary = validate_immunogen_summary_csv(
        summary_csv=args.summary_csv,
        config=config,
    )
    write_immunogen_validation_report(
        report_json_path=args.out_report_json,
        issues_csv_path=args.out_issues_csv,
        issues=issues,
        summary=summary,
    )
    print(
        "Immunogen validation {status}: {errors} error(s), {warnings} warning(s), "
        "{passed}/{rows} passed rows".format(
            status=summary["status"],
            errors=summary["error_count"],
            warnings=summary["warning_count"],
            passed=summary["passed_rows"],
            rows=summary["num_rows"],
        )
    )
    print(
        "Category pass rows: fixed={fixed} tcell={tcell} cleavage={cleavage} "
        "evaluation={evaluation}".format(
            fixed=summary["fixed_passed_rows"],
            tcell=summary["tcell_passed_rows"],
            cleavage=summary["cleavage_passed_rows"],
            evaluation=summary["evaluation_passed_rows"],
        )
    )
    print(f"Report JSON: {args.out_report_json}")
    print(f"Issues CSV: {args.out_issues_csv}")
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
