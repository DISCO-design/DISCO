#!/usr/bin/env python3
"""Plan fair PG-vs-DISCO motif-guidance comparison runs.

This runner writes commands, manifests, length-specific DISCO input JSONs, and
comparison scaffolding.  It intentionally does not execute inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disco.model.cogen_inference.guidance.experiment_plan import (
    ComparisonPlanConfig,
    parse_fixed_fragment_plan,
    plan_runs,
)
from disco.model.cogen_inference.guidance.scoring import parse_motifs


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--disco-root", type=Path, default=ROOT)
    parser.add_argument(
        "--pg-root",
        type=Path,
        default=ROOT.parent / "ProteinGenerator_MOTIFS",
    )
    parser.add_argument(
        "--systems",
        default="DISCO,ProteinGenerator",
        help="Comma-separated systems: DISCO,ProteinGenerator",
    )
    parser.add_argument(
        "--modes",
        default="standard,immunogen,alternating",
        help="Comma-separated modes: standard,immunogen,alternating",
    )
    parser.add_argument("--variant-set", choices=["core", "exploratory"], default="core")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-cycle", type=int, default=4)
    parser.add_argument("--disco-n-step", type=int, default=100)
    parser.add_argument("--pg-standard-T", type=int, default=60)
    parser.add_argument("--pg-short-T", type=int, default=4)
    parser.add_argument("--designs-per-replicate", type=int, default=1)
    parser.add_argument("--standard-length", type=int, default=80)
    parser.add_argument("--immunogen-length", type=int, default=80)
    parser.add_argument("--alternating-length", type=int, default=24)
    parser.add_argument("--standard-motifs", default="AHK,GCP,KTTKS,VW")
    parser.add_argument("--immunogen-motifs", default="GP,DR,DK,VW")
    parser.add_argument("--alternating-list1", default="GP,DK")
    parser.add_argument("--alternating-list2", default="DR,VW")
    parser.add_argument("--expected-start-group", default="list1", choices=["list1", "list2"])
    parser.add_argument(
        "--fixed-fragment",
        action="append",
        default=[],
        help="Repeatable immunogen fragment: start_1based:sequence[:label]",
    )
    parser.add_argument("--pg-seed-pdb", type=Path)
    parser.add_argument(
        "--pg-seed-trb",
        type=Path,
        help="Optional ProteinGenerator TRB matching --pg-seed-pdb for direct Immunogen partial diffusion.",
    )
    parser.add_argument("--pg-seed-sequence")
    parser.add_argument("--pg-fixed-fragments-json", type=Path)
    parser.add_argument("--pg-checkpoint", type=Path)
    parser.add_argument(
        "--pg-immunogen-direct",
        action="store_true",
        help="Plan PG Immunogen via direct Conda Python instead of Docker workflow.",
    )
    parser.add_argument(
        "--pg-save-best-motif-only",
        action="store_true",
        help="For PG standard runs, select the saved best step by hard motif score only.",
    )
    parser.add_argument(
        "--no-disco-save-trajectories",
        action="store_true",
        help="Disable DISCO coordinate trajectory tensors in planned comparison runs.",
    )
    parser.add_argument("--disco-trajectory-dtype", default="float16")
    parser.add_argument(
        "--disco-trajectory-max-steps",
        type=int,
        default=0,
        help="0 keeps all saved trajectory states; positive values truncate.",
    )
    parser.add_argument("--disco-python", default=sys.executable)
    parser.add_argument("--pg-python", default=sys.executable)
    parser.add_argument("--docker-image", default="proteingenerator-motifs:local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ComparisonPlanConfig(
        out_dir=args.out.resolve(),
        disco_root=args.disco_root.resolve(),
        pg_root=args.pg_root.resolve(),
        modes=parse_csv(args.modes),
        systems=parse_csv(args.systems),
        variant_set=args.variant_set,
        seeds=parse_int_csv(args.seeds),
        n_cycle=args.n_cycle,
        standard_length=args.standard_length,
        immunogen_length=args.immunogen_length,
        alternating_length=args.alternating_length,
        disco_n_step=args.disco_n_step,
        pg_standard_T=args.pg_standard_T,
        pg_short_T=args.pg_short_T,
        designs_per_replicate=args.designs_per_replicate,
        standard_motifs=parse_motifs(args.standard_motifs),
        immunogen_motifs=parse_motifs(args.immunogen_motifs),
        alternating_list1=parse_csv(args.alternating_list1),
        alternating_list2=parse_csv(args.alternating_list2),
        expected_start_group=args.expected_start_group,
        fixed_fragments=[
            parse_fixed_fragment_plan(raw)
            for raw in args.fixed_fragment
        ],
        pg_seed_pdb=args.pg_seed_pdb.resolve() if args.pg_seed_pdb else None,
        pg_seed_trb=args.pg_seed_trb.resolve() if args.pg_seed_trb else None,
        pg_seed_sequence=args.pg_seed_sequence,
        pg_fixed_fragments_json=(
            args.pg_fixed_fragments_json.resolve()
            if args.pg_fixed_fragments_json
            else None
        ),
        pg_checkpoint=args.pg_checkpoint.resolve() if args.pg_checkpoint else None,
        pg_immunogen_direct=args.pg_immunogen_direct,
        pg_save_best_motif_only=args.pg_save_best_motif_only,
        disco_save_trajectories=not args.no_disco_save_trajectories,
        disco_trajectory_dtype=args.disco_trajectory_dtype,
        disco_trajectory_max_steps=args.disco_trajectory_max_steps,
        disco_python=args.disco_python,
        pg_python=args.pg_python,
        docker_image=args.docker_image,
    )
    runs = plan_runs(config)
    statuses = sorted({run.status for run in runs})
    print(f"Wrote {len(runs)} planned runs to {config.out_dir}")
    print(f"Statuses: {', '.join(statuses)}")
    print(f"Plan CSV: {config.out_dir / 'planned_runs.csv'}")
    print(f"Compare scaffold: {config.out_dir / 'compare_command.txt'}")


if __name__ == "__main__":
    main()
