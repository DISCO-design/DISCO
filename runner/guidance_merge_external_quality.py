#!/usr/bin/env python3
"""Merge optional ESM/RMSD/Rosetta metrics into completed design summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.external_quality import (
    ExternalQualityWeights,
    write_external_quality_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--external-metrics-csv", type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--top-csv", type=Path)
    parser.add_argument("--key-field", default="design_id")
    parser.add_argument("--confidence-weight", type=float, default=0.30)
    parser.add_argument("--esm2-weight", type=float, default=0.20)
    parser.add_argument("--rmsd-weight", type=float, default=0.20)
    parser.add_argument("--rosetta-weight", type=float, default=0.20)
    parser.add_argument("--motif-weight", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_external_quality_outputs(
        summary_csv=args.summary_csv.resolve(),
        external_metrics_csv=(
            args.external_metrics_csv.resolve()
            if args.external_metrics_csv is not None
            else None
        ),
        out_csv=args.out_csv.resolve(),
        report_json=args.report_json.resolve(),
        top_csv=args.top_csv.resolve() if args.top_csv is not None else None,
        key_field=args.key_field,
        weights=ExternalQualityWeights(
            confidence=args.confidence_weight,
            esm2=args.esm2_weight,
            rmsd=args.rmsd_weight,
            rosetta=args.rosetta_weight,
            motif=args.motif_weight,
        ),
    )
    print(
        "External quality merge: {rows} rows, best={best} ({score})".format(
            rows=report["num_rows"],
            best=report["best_posthoc_quality_design_id"],
            score=report["best_posthoc_quality_score"],
        )
    )
    print(f"Output CSV: {args.out_csv.resolve()}")
    print(f"Report JSON: {args.report_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
