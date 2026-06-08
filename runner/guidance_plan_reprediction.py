#!/usr/bin/env python3
"""Plan fixed-sequence DISCO second-pass reprediction runs."""

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
from disco.model.cogen_inference.guidance.reprediction import (
    RepredictionPlanConfig,
    parse_seed_list,
    write_reprediction_plan,
)
from disco.model.cogen_inference.guidance.scoring import parse_motifs


def parse_optional_float(value: str) -> float | None:
    if value.strip().lower() in {"", "none", "skip"}:
        return None
    return float(value)


def parse_optional_int(value: str) -> int | None:
    if value.strip().lower() in {"", "none", "skip"}:
        return None
    return int(value)


def parse_optional_motifs(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return parse_motifs(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--motifs", required=True)
    parser.add_argument("--disco-root", default=ROOT, type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-cycle", type=int, default=4)
    parser.add_argument("--n-step", type=int, default=100)
    parser.add_argument("--experiment", default="designable")
    parser.add_argument("--max-designs", type=parse_optional_int)
    parser.add_argument(
        "--rank-by",
        default="comparison_score",
        choices=["input_order", "comparison_score", "motif_score", "confidence"],
    )
    parser.add_argument("--min-motif-score", type=parse_optional_float)
    parser.add_argument("--min-confidence", type=parse_optional_float)
    parser.add_argument("--require-fixed-preserved", action="store_true")
    parser.add_argument("--require-same-evaluation-artifact", action="store_true")
    parser.add_argument(
        "--fixed-fragment",
        action="append",
        type=parse_fixed_fragment_plan,
        help="Repeatable immunogen fragment: start_1based:sequence[:label]",
    )
    parser.add_argument("--tcell-motifs")
    parser.add_argument("--cleavage-motifs")
    parser.add_argument("--variant", default="reprediction")
    parser.add_argument(
        "--no-require-confidence-analysis",
        action="store_true",
        help="Do not add --require-confidence to the second-pass analysis command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = write_reprediction_plan(
        RepredictionPlanConfig(
            analysis_csv=args.analysis_csv.resolve(),
            out_dir=args.out.resolve(),
            disco_root=args.disco_root.resolve(),
            motifs=parse_motifs(args.motifs),
            seeds=parse_seed_list(args.seeds),
            n_cycle=args.n_cycle,
            n_step=args.n_step,
            experiment=args.experiment,
            python=args.python,
            max_designs=args.max_designs,
            rank_by=args.rank_by,
            min_motif_score=args.min_motif_score,
            min_confidence=args.min_confidence,
            require_fixed_preserved=args.require_fixed_preserved,
            require_same_evaluation_artifact=args.require_same_evaluation_artifact,
            fixed_fragments=args.fixed_fragment or [],
            tcell_motifs=parse_optional_motifs(args.tcell_motifs),
            cleavage_motifs=parse_optional_motifs(args.cleavage_motifs),
            variant=args.variant,
            require_confidence_analysis=not args.no_require_confidence_analysis,
        )
    )
    print(
        "Reprediction plan: {planned} planned, {skipped} skipped".format(
            planned=manifest["num_planned"],
            skipped=manifest["num_skipped"],
        )
    )
    print(f"Plan CSV: {manifest['planned_csv']}")
    print(f"Manifest: {Path(manifest['out_dir']) / 'reprediction_plan_manifest.json'}")
    return 1 if manifest["num_planned"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
