#!/usr/bin/env python3
"""Audit which ProteinGenerator motif features are covered by the DISCO port."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.parity import (
    audit_parity,
    write_parity_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disco-root",
        type=Path,
        default=ROOT,
        help="DISCO checkout root. Defaults to this repository root.",
    )
    parser.add_argument(
        "--pg-root",
        type=Path,
        help="Optional ProteinGenerator_MOTIFS root for source-handle checks.",
    )
    parser.add_argument("--out-report-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument(
        "--require-all-ported",
        action="store_true",
        help=(
            "Exit nonzero if any feature is partial/not_ported, or if required "
            "DISCO evidence is missing. Useful only for final completion gates."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, summary = audit_parity(
        disco_root=args.disco_root,
        pg_root=args.pg_root,
    )
    write_parity_report(
        report_json_path=args.out_report_json,
        report_md_path=args.out_md,
        records=records,
        summary=summary,
    )

    print(
        "Parity audit {status}: {ported} ported, {partial} partial, "
        "{not_ported} not ported, {not_applicable} not applicable".format(
            status=summary["status"],
            ported=summary["ported_count"],
            partial=summary["partial_count"],
            not_ported=summary["not_ported_count"],
            not_applicable=summary["not_applicable_count"],
        )
    )
    print(f"Report JSON: {args.out_report_json.resolve()}")
    print(f"Report Markdown: {args.out_md.resolve()}")

    if summary["missing_disco_evidence_count"] or summary["missing_test_evidence_count"]:
        return 1
    if args.require_all_ported and (
        summary["partial_count"] > 0 or summary["not_ported_count"] > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
