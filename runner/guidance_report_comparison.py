#!/usr/bin/env python3
"""Write a Markdown report from PG-vs-DISCO comparison CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.comparison_report import (
    write_comparison_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--validation-report-json", type=Path)
    parser.add_argument(
        "--title",
        default="PG-vs-DISCO Motif Comparison Report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_comparison_report(
        normalized_csv=args.normalized_csv,
        summary_csv=args.summary_csv,
        out_md=args.out_md,
        validation_report_json=args.validation_report_json,
        title=args.title,
    )
    print(f"Wrote comparison report to {args.out_md}")


if __name__ == "__main__":
    main()
