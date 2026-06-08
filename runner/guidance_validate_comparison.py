#!/usr/bin/env python3
"""Validate planned or completed PG-vs-DISCO comparison studies.

The comparison planner writes reproducible run directories; this runner checks
that the matrix, per-run scaffolding, and completed summary CSVs are complete
enough for scientific comparison.  It does not execute DISCO or
ProteinGenerator inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.comparison_validation import (
    validate_comparison_plan,
    write_validation_report,
)


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned-runs-csv", required=True, type=Path)
    parser.add_argument("--out-report-json", required=True, type=Path)
    parser.add_argument("--out-issues-csv", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-summaries",
        action="store_true",
        help="Report missing summary CSVs as warnings instead of errors.",
    )
    parser.add_argument(
        "--no-require-confidence",
        action="store_false",
        dest="require_confidence",
        help="Do not require confidence/pLDDT-like fields in summary CSVs.",
    )
    parser.add_argument(
        "--require-diagnostics",
        action="store_true",
        help="Require DISCO summary rows to point to diagnostics JSONL files.",
    )
    parser.add_argument(
        "--no-require-fixed-preserved",
        action="store_false",
        dest="require_fixed_preserved",
        help="Do not require immunogen rows to prove fixed-fragment preservation.",
    )
    parser.add_argument(
        "--required-systems",
        default="DISCO,ProteinGenerator",
        help=(
            "Comma-separated systems required in every mode/variant group. "
            "Use an empty string for single-system smoke checks."
        ),
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit nonzero when warnings are present.",
    )
    parser.set_defaults(require_confidence=True, require_fixed_preserved=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues, summary = validate_comparison_plan(
        planned_runs_csv=args.planned_runs_csv,
        allow_missing_summaries=args.allow_missing_summaries,
        require_confidence=args.require_confidence,
        require_diagnostics=args.require_diagnostics,
        require_fixed_preserved=args.require_fixed_preserved,
        required_systems=parse_csv(args.required_systems),
    )
    write_validation_report(
        report_json_path=args.out_report_json,
        issues_csv_path=args.out_issues_csv,
        issues=issues,
        summary=summary,
    )

    print(
        "Validation {status}: {errors} error(s), {warnings} warning(s), "
        "{completed}/{expected} completed summaries".format(
            status=summary["status"],
            errors=summary["error_count"],
            warnings=summary["warning_count"],
            completed=summary["completed_summary_count"],
            expected=summary["expected_summary_count"],
        )
    )
    print(f"Report JSON: {args.out_report_json}")
    print(f"Issues CSV: {args.out_issues_csv}")

    if summary["error_count"]:
        return 1
    if args.fail_on_warning and summary["warning_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
