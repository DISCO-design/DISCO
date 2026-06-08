#!/usr/bin/env python3
"""Analyze completed DISCO output folders for motif-comparison CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.disco_analysis import analyze_disco_dump_rows
from disco.model.cogen_inference.guidance.immunogen import spec_from_dict
from disco.model.cogen_inference.guidance.scoring import parse_motifs


def parse_optional_motif_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def parse_fixed_fragment(value: str) -> dict[str, object]:
    """Parse `start:sequence[:label]` into an immunogen fixed-fragment dict."""

    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "fixed fragments must use start_1based:sequence[:label]"
        )
    fragment: dict[str, object] = {
        "start_1based": int(parts[0]),
        "sequence": parts[1].upper(),
    }
    if len(parts) == 3:
        fragment["label"] = parts[2]
    return fragment


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    priority = [
        "design_id",
        "base_id",
        "sequence_index",
        "sequence",
        "sequence_length",
        "longest_homopolymer",
        "aa_entropy_norm",
        "seed_sequence_identity",
        "num_sequence_changes_vs_seed",
        "ca_rmsd_to_seed",
        "confidence",
        "confidence_metric",
        "confidence_match_status",
        "unique_motifs",
        "total_motifs",
        "motif_score",
        "fixed_fragments_preserved",
        "num_fixed_mismatches",
        "alternating_combined_score",
        "sequence_path",
        "structure_path",
        "structure_match_status",
        "confidence_path",
    ]
    ordered = [key for key in priority if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def fixed_spec_from_args(args: argparse.Namespace):
    fragments = []
    if args.fixed_fragments_json is not None:
        data = json.loads(args.fixed_fragments_json.read_text())
        fragments.extend(data.get("fixed_fragments", []))
    fragments.extend(args.fixed_fragment or [])
    if not fragments:
        return None
    return spec_from_dict({"fixed_fragments": fragments})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--motifs", required=True)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--motif-all-weight", default=0.8, type=float)
    parser.add_argument("--motif-many-weight", default=0.2, type=float)
    parser.add_argument("--motif-many-cap", default=0, type=int)
    parser.add_argument("--fixed-fragments-json", type=Path)
    parser.add_argument(
        "--fixed-fragment",
        action="append",
        type=parse_fixed_fragment,
        help="Repeatable immunogen fragment: start_1based:sequence[:label]",
    )
    parser.add_argument("--tcell-motifs")
    parser.add_argument("--cleavage-motifs")
    parser.add_argument("--alternating-list1")
    parser.add_argument("--alternating-list2")
    parser.add_argument("--expected-start-group", default="list1")
    parser.add_argument("--seed-sequence")
    parser.add_argument("--seed-pdb", type=Path)
    parser.add_argument(
        "--require-confidence",
        action="store_true",
        help="Fail if any sequence row has no exact or fallback confidence file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dump_dir = args.dump_dir.resolve()
    args.out_csv = args.out_csv.resolve()
    manifest_path = (
        args.manifest_json.resolve()
        if args.manifest_json is not None
        else args.out_csv.with_suffix(".manifest.json")
    )

    motifs = parse_motifs(args.motifs)
    rows = analyze_disco_dump_rows(
        dump_dir=args.dump_dir,
        motifs=motifs,
        motif_all_weight=args.motif_all_weight,
        motif_many_weight=args.motif_many_weight,
        motif_many_cap=args.motif_many_cap,
        fixed_spec=fixed_spec_from_args(args),
        tcell_motifs=parse_optional_motif_list(args.tcell_motifs),
        cleavage_motifs=parse_optional_motif_list(args.cleavage_motifs),
        alternating_list1=parse_optional_motif_list(args.alternating_list1),
        alternating_list2=parse_optional_motif_list(args.alternating_list2),
        expected_start_group=args.expected_start_group,
        seed_sequence=args.seed_sequence,
        seed_pdb=args.seed_pdb.resolve() if args.seed_pdb is not None else None,
    )
    if args.require_confidence:
        missing = [
            row["design_id"]
            for row in rows
            if row.get("confidence_match_status") in {"missing", "missing_exact_sample"}
        ]
        if missing:
            raise SystemExit(
                "Missing matched DISCO confidence files for: " + ",".join(missing)
            )

    write_csv(args.out_csv, rows)
    manifest = {
        "script_name": "guidance_analyze_disco.py",
        "run_type": "disco_guidance_analysis",
        "variant": args.variant,
        "dump_dir": str(args.dump_dir),
        "out_csv": str(args.out_csv),
        "motifs": motifs,
        "seed_sequence_provided": args.seed_sequence is not None,
        "seed_pdb": str(args.seed_pdb.resolve()) if args.seed_pdb is not None else "",
        "num_rows": len(rows),
        "confidence_match_status_counts": {
            status: sum(1 for row in rows if row["confidence_match_status"] == status)
            for status in sorted({str(row["confidence_match_status"]) for row in rows})
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote DISCO analysis rows to {args.out_csv}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
