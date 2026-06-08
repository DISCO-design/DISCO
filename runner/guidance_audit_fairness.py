"""Audit fairness of a planned PG-vs-DISCO comparison matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance import (
    audit_fairness_csv,
    write_fairness_report,
)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned-runs-csv", required=True, type=Path)
    parser.add_argument("--out-report-json", required=True, type=Path)
    parser.add_argument("--out-issues-csv", required=True, type=Path)
    parser.add_argument(
        "--required-systems",
        default="DISCO,ProteinGenerator",
        help="Comma-separated systems that every mode/variant group must contain.",
    )
    parser.add_argument("--recommended-n-cycle", type=int, default=4)
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Exit zero when only warnings are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues, summary = audit_fairness_csv(
        args.planned_runs_csv.resolve(),
        required_systems=parse_csv(args.required_systems),
        recommended_n_cycle=args.recommended_n_cycle,
    )
    write_fairness_report(
        report_json_path=args.out_report_json.resolve(),
        issues_csv_path=args.out_issues_csv.resolve(),
        issues=issues,
        summary=summary,
    )
    print(
        "Fairness audit "
        + summary["status"]
        + f": {summary['error_count']} error(s), {summary['warning_count']} warning(s)"
    )
    print(f"Wrote report to {args.out_report_json.resolve()}")
    if summary["error_count"] > 0:
        raise SystemExit(1)
    if summary["warning_count"] > 0 and not args.allow_warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
