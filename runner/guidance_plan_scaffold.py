#!/usr/bin/env python3
"""Plan a DISCO sequence-level scaffold input JSON.

This is a planning utility only.  It does not run DISCO inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.experiment_plan import (
    parse_fixed_fragment_plan,
)
from disco.model.cogen_inference.guidance.scaffold_plan import (
    fixed_fragments_from_json,
    parse_fixed_range,
    write_scaffold_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--name", default="disco_scaffold")
    parser.add_argument("--count", default=1, type=int)
    parser.add_argument("--seed-sequence")
    parser.add_argument(
        "--fixed-fragment",
        action="append",
        type=parse_fixed_fragment_plan,
        help="Repeatable fixed fragment: start_1based:sequence[:label]",
    )
    parser.add_argument(
        "--fixed-fragments-json",
        action="append",
        type=Path,
        help="Repeatable PG/DISCO immunogen fixed-fragments JSON file.",
    )
    parser.add_argument(
        "--fixed-range",
        action="append",
        type=parse_fixed_range,
        help="Repeatable seed range to keep fixed: start_1based:end_1based[:label]",
    )
    parser.add_argument(
        "--pg-seed-pdb",
        type=Path,
        help="Recorded for provenance; not converted to a DISCO structure template.",
    )
    parser.add_argument(
        "--pg-contig",
        action="append",
        default=[],
        help="Recorded/audited PG contig argument. Structural contigs are unsupported.",
    )
    parser.add_argument(
        "--pg-inpaint-seq",
        action="append",
        default=[],
        help="Recorded/audited PG inpaint_seq argument. Not converted to DISCO masks.",
    )
    parser.add_argument(
        "--pg-inpaint-str",
        action="append",
        default=[],
        help="Recorded/audited PG inpaint_str argument. Not converted to DISCO masks.",
    )
    parser.add_argument("--out-input-json", required=True, type=Path)
    parser.add_argument("--out-manifest-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixed_fragments = list(args.fixed_fragment or [])
    for json_path in args.fixed_fragments_json or []:
        fixed_fragments.extend(fixed_fragments_from_json(json_path))

    result = write_scaffold_plan(
        input_json_path=args.out_input_json.resolve(),
        manifest_json_path=args.out_manifest_json.resolve(),
        length=args.length,
        name=args.name,
        count=args.count,
        fixed_fragments=fixed_fragments,
        fixed_ranges=args.fixed_range or [],
        seed_sequence=args.seed_sequence,
        pg_seed_pdb=args.pg_seed_pdb.resolve() if args.pg_seed_pdb is not None else None,
        pg_contigs=args.pg_contig,
        pg_inpaint_seq=args.pg_inpaint_seq,
        pg_inpaint_str=args.pg_inpaint_str,
    )

    print(f"Scaffold plan: {result.status}")
    print(f"Input JSON: {result.input_json_path}")
    print(f"Manifest: {result.manifest_json_path}")
    print(f"Sequence: {result.sequence}")
    if result.issues:
        print(f"Issues: {len(result.issues)}")
        for issue in result.issues:
            print(f"- {issue.severity} {issue.code}: {issue.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
