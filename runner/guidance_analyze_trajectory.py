#!/usr/bin/env python3
"""Analyze DISCO guidance diagnostics into per-step trajectory tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.trajectory import write_trajectory_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-jsonl",
        action="append",
        required=True,
        type=Path,
        help="Repeatable path to a *_guidance_diagnostics.jsonl file.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help=(
            "Optional repeatable sample id. If provided, count must match "
            "--diagnostics-jsonl count. Otherwise ids are inferred from filenames."
        ),
    )
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-summary-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = args.sample_id if args.sample_id else None
    summary = write_trajectory_outputs(
        diagnostics_jsonl_paths=[path.resolve() for path in args.diagnostics_jsonl],
        out_csv=args.out_csv.resolve(),
        out_summary_json=args.out_summary_json.resolve(),
        sample_ids=sample_ids,
    )
    aggregate = summary["aggregate"]
    print(
        "Trajectory analysis wrote {steps} step row(s) across {samples} sample(s).".format(
            steps=aggregate["total_trajectory_steps"],
            samples=aggregate["num_samples"],
        )
    )
    print(f"Trajectory CSV: {args.out_csv.resolve()}")
    print(f"Summary JSON: {args.out_summary_json.resolve()}")


if __name__ == "__main__":
    main()
