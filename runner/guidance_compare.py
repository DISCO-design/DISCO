#!/usr/bin/env python3
"""Compare ProteinGenerator and DISCO motif-guidance summary CSVs.

The runner does not launch inference. It normalizes existing summary CSVs into
one common schema and writes aggregate comparison metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.comparison import (
    read_normalized_csv,
    summarize_by_group,
    write_normalized_rows,
    write_summary_rows,
)


def parse_source(value: str) -> tuple[str, str, str, Path]:
    """Parse `system:variant:path` or `system:mode:variant:path` source specs."""

    parts = value.split(":", 3)
    if len(parts) == 3:
        system, variant, path = parts
        mode = ""
    elif len(parts) == 4:
        system, mode, variant, path = parts
    else:
        raise argparse.ArgumentTypeError(
            "sources must use system:variant:path or system:mode:variant:path"
        )
    if system not in {"DISCO", "ProteinGenerator"}:
        raise argparse.ArgumentTypeError(
            "system must be DISCO or ProteinGenerator"
        )
    return system, mode, variant, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source,
        help="system:variant:path; repeat for each CSV",
    )
    parser.add_argument("--out-normalized", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = []
    for system, mode, variant, path in args.source:
        if not path.exists():
            if args.allow_missing:
                print(f"Skipping missing source: {path}")
                continue
            raise SystemExit(f"Missing source CSV: {path}")
        all_rows.extend(
            read_normalized_csv(
                path,
                system=system,
                variant=variant,
                mode=mode,
            )
        )

    if not all_rows:
        raise SystemExit("No rows available for comparison")

    write_normalized_rows(args.out_normalized, all_rows)
    write_summary_rows(args.out_summary, summarize_by_group(all_rows))
    print(f"Wrote normalized rows to {args.out_normalized}")
    print(f"Wrote summary rows to {args.out_summary}")


if __name__ == "__main__":
    main()
