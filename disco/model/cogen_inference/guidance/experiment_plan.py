"""Comparison experiment planning for DISCO and ProteinGenerator.

This module does not execute model inference.  It materializes reproducible
command plans, per-run manifests, and DISCO input JSONs so later PG-vs-DISCO
runs can be launched and analyzed without hand-copying long command lines.
"""

from __future__ import annotations

import csv
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from disco.model.cogen_inference.guidance.scoring import parse_motifs


@dataclass(frozen=True)
class FixedFragmentPlan:
    """Fixed immunogen fragment using the shared 1-based coordinate contract."""

    start_1based: int
    sequence: str
    label: str = ""

    @property
    def start_0based(self) -> int:
        return self.start_1based - 1

    @property
    def end_0based_exclusive(self) -> int:
        return self.start_0based + len(self.sequence)

    def as_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "start_1based": self.start_1based,
            "sequence": self.sequence,
        }
        if self.label:
            row["label"] = self.label
        return row


@dataclass(frozen=True)
class VariantPlan:
    """One named parameter set shared across systems where possible."""

    mode: str
    variant: str
    description: str
    motif_weight_all: float = 1.0
    motif_weight_many: float = 0.0
    motif_top_k: int = 1
    motif_temperature: float = 0.25
    gradient_scale: float = 1.0
    local_rescue_enabled: bool = False
    local_rescue_top_k: int = 3
    local_rescue_bonus: float = 1.0
    local_rescue_min_probability: float = 0.03
    save_best_confidence_weight: float = 0.0
    immunogen_spacing_aware: bool = False
    alternating_gradient_scale: float = 1.0
    alternating_guidance_mode: str = "slot"
    pg_potential: str = "motifs_bias"
    pg_potential_scale: float = 25.0


@dataclass(frozen=True)
class PlannedRun:
    """One planned command plus enough metadata to reproduce and compare it."""

    system: str
    mode: str
    variant: str
    replicate: int
    seed: int
    run_dir: Path
    command: list[str]
    analysis_command: list[str]
    status: str
    input_json_path: Path | None
    summary_csv_path: Path
    parameters: dict[str, object]
    fairness_notes: list[str]

    def command_text(self) -> str:
        return command_to_string(self.command)

    def analysis_command_text(self) -> str:
        return command_to_string(self.analysis_command)

    def csv_row(self) -> dict[str, object]:
        return {
            "system": self.system,
            "mode": self.mode,
            "variant": self.variant,
            "replicate": self.replicate,
            "seed": self.seed,
            "status": self.status,
            "run_dir": str(self.run_dir),
            "input_json_path": "" if self.input_json_path is None else str(self.input_json_path),
            "summary_csv_path": str(self.summary_csv_path),
            "command": self.command_text(),
            "analysis_command": self.analysis_command_text(),
            "parameters_json": json.dumps(self.parameters, sort_keys=True),
            "fairness_notes": " | ".join(self.fairness_notes),
        }


@dataclass(frozen=True)
class ComparisonPlanConfig:
    """Top-level knobs for one planned PG-vs-DISCO comparison matrix."""

    out_dir: Path
    disco_root: Path
    pg_root: Path
    modes: list[str]
    systems: list[str]
    variant_set: str
    seeds: list[int]
    n_cycle: int = 4
    standard_length: int = 80
    immunogen_length: int = 80
    alternating_length: int = 24
    disco_n_step: int = 100
    pg_standard_T: int = 60
    pg_short_T: int = 4
    designs_per_replicate: int = 1
    standard_motifs: list[str] | None = None
    immunogen_motifs: list[str] | None = None
    alternating_list1: list[str] | None = None
    alternating_list2: list[str] | None = None
    expected_start_group: str = "list1"
    fixed_fragments: list[FixedFragmentPlan] | None = None
    pg_seed_pdb: Path | None = None
    pg_seed_trb: Path | None = None
    pg_seed_sequence: str | None = None
    pg_fixed_fragments_json: Path | None = None
    pg_checkpoint: Path | None = None
    pg_immunogen_direct: bool = False
    pg_save_best_motif_only: bool = False
    disco_save_trajectories: bool = True
    disco_trajectory_dtype: str = "float16"
    disco_trajectory_max_steps: int = 0
    disco_python: str = "python"
    pg_python: str = "python"
    docker_image: str = "proteingenerator-motifs:local"


def command_to_string(command: Iterable[str]) -> str:
    """Return a shell-readable command string without changing the command list."""

    return shlex.join([str(part) for part in command])


def command_with_env(env: dict[str, str], command: Iterable[str]) -> list[str]:
    """Return a shell command list with portable inline environment variables."""

    return ["env"] + [f"{key}={value}" for key, value in env.items()] + list(command)


def hydra_list(values: Iterable[str]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def hydra_fixed_fragments(fragments: Iterable[FixedFragmentPlan]) -> str:
    rendered = []
    for fragment in fragments:
        fields = [
            f"start_1based:{fragment.start_1based}",
            f"sequence:{fragment.sequence}",
        ]
        if fragment.label:
            fields.append(f"label:{fragment.label}")
        rendered.append("{" + ",".join(fields) + "}")
    return "[" + ",".join(rendered) + "]"


def parse_fixed_fragment_plan(value: str) -> FixedFragmentPlan:
    """Parse `start:sequence[:label]` from planner CLI arguments."""

    parts = value.split(":", 2)
    if len(parts) < 2:
        raise ValueError("fixed fragments must use start_1based:sequence[:label]")
    return FixedFragmentPlan(
        start_1based=int(parts[0]),
        sequence=parts[1].strip().upper(),
        label=parts[2].strip() if len(parts) == 3 else "",
    )


def sequence_with_fixed_fragments(
    *,
    length: int,
    fixed_fragments: Iterable[FixedFragmentPlan] = (),
) -> str:
    """Return a DISCO proteinChain string with fixed letters and `-` masks."""

    sequence = ["-"] * length
    occupied: list[str | None] = [None] * length
    for fragment in fixed_fragments:
        if fragment.start_1based < 1:
            raise ValueError("fixed fragment start_1based must be >= 1")
        if fragment.end_0based_exclusive > length:
            raise ValueError(
                f"fixed fragment {fragment.sequence!r} at {fragment.start_1based} "
                f"extends past length {length}"
            )
        for offset, aa in enumerate(fragment.sequence):
            position = fragment.start_0based + offset
            owner = occupied[position]
            if owner is not None and sequence[position] != aa:
                raise ValueError(
                    f"conflicting fixed fragments at position {position + 1}: "
                    f"{owner!r} and {fragment.label or fragment.sequence!r}"
                )
            occupied[position] = fragment.label or fragment.sequence
            sequence[position] = aa
    return "".join(sequence)


def write_disco_input_json(
    *,
    path: Path,
    length: int,
    name: str,
    fixed_fragments: Iterable[FixedFragmentPlan] = (),
) -> Path:
    """Write a single-length DISCO input JSON and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = sequence_with_fixed_fragments(
        length=length,
        fixed_fragments=fixed_fragments,
    )
    record = [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": 1,
                    }
                }
            ],
            "name": name,
        }
    ]
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def default_variants(variant_set: str) -> list[VariantPlan]:
    """Return the core or exploratory comparison variants."""

    variants = [
        VariantPlan(
            mode="standard",
            variant="standard_gradient_coverage",
            description="Coverage-first motif guidance, matching PG's strongest documented baseline family.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=1,
            motif_temperature=0.25,
            pg_potential="motifs_bias",
            pg_potential_scale=25.0,
        ),
        VariantPlan(
            mode="immunogen",
            variant="immunogen_coverage_guided",
            description="Fixed fragments protected; motifs guided only into mutable windows.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=3,
            motif_temperature=0.5,
            local_rescue_enabled=True,
            local_rescue_top_k=4,
            local_rescue_bonus=1.5,
            pg_potential="immunogen_mutable_motifs_bias",
            pg_potential_scale=100.0,
        ),
        VariantPlan(
            mode="alternating",
            variant="alternating_slot_tail_guard",
            description="Slot-based alternating motifs with hard start/tail save ranking.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=3,
            motif_temperature=0.5,
            alternating_gradient_scale=1.0,
            pg_potential="alternating_motifs_bias",
            pg_potential_scale=50.0,
        ),
    ]
    if variant_set == "core":
        return variants
    if variant_set != "exploratory":
        raise ValueError(f"Unknown variant_set: {variant_set}")
    return variants + [
        VariantPlan(
            mode="standard",
            variant="standard_local_rescue",
            description="Coverage-first motif guidance plus local top-k rescue.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=3,
            motif_temperature=0.25,
            local_rescue_enabled=True,
            local_rescue_top_k=3,
            local_rescue_bonus=1.0,
            pg_potential="motifs_bias",
            pg_potential_scale=25.0,
        ),
        VariantPlan(
            mode="immunogen",
            variant="immunogen_spacing_aware",
            description="Coverage-guided immunogen with motif-spacing tie-breaker.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=3,
            motif_temperature=0.5,
            local_rescue_enabled=True,
            local_rescue_top_k=4,
            local_rescue_bonus=1.5,
            immunogen_spacing_aware=True,
            pg_potential="immunogen_mutable_motifs_bias",
            pg_potential_scale=100.0,
        ),
        VariantPlan(
            mode="alternating",
            variant="alternating_slot_tail_guard_strong",
            description="Stronger slot-based alternating guidance, same hard audit.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=3,
            motif_temperature=0.5,
            alternating_gradient_scale=2.0,
            pg_potential="alternating_motifs_bias",
            pg_potential_scale=75.0,
        ),
        VariantPlan(
            mode="alternating",
            variant="transition_only_tail_guard",
            description="Slot-free alternating comparison: ordered soft transitions plus the same hard audit.",
            motif_weight_all=1.0,
            motif_weight_many=0.0,
            motif_top_k=3,
            motif_temperature=0.5,
            alternating_gradient_scale=1.0,
            alternating_guidance_mode="transition",
            pg_potential="soft_transition_alternating_motifs_bias",
            pg_potential_scale=50.0,
        ),
    ]


def motifs_for_mode(config: ComparisonPlanConfig, mode: str) -> list[str]:
    if mode == "standard":
        return parse_motifs(config.standard_motifs or ["AHK", "GCP", "KTTKS", "VW"])
    if mode == "immunogen":
        return parse_motifs(config.immunogen_motifs or ["GP", "DR", "DK", "VW"])
    if mode == "alternating":
        list1 = config.alternating_list1 or ["GP", "DK"]
        list2 = config.alternating_list2 or ["DR", "VW"]
        return parse_motifs([*list1, *list2])
    raise ValueError(f"Unknown mode: {mode}")


def length_for_mode(config: ComparisonPlanConfig, mode: str) -> int:
    if mode == "standard":
        return config.standard_length
    if mode == "immunogen":
        return config.immunogen_length
    if mode == "alternating":
        return config.alternating_length
    raise ValueError(f"Unknown mode: {mode}")


def pg_T_for_mode(config: ComparisonPlanConfig, mode: str) -> int:
    return config.pg_standard_T if mode == "standard" else config.pg_short_T


def common_fairness_notes(config: ComparisonPlanConfig, mode: str) -> list[str]:
    return [
        f"n_cycle fixed to {config.n_cycle} for both systems",
        "PG --T and DISCO sample_diffusion.N_step are not mathematically identical; compare as budget knobs, not identical diffusion time",
        f"mode length fixed to {length_for_mode(config, mode)}",
    ]


def build_disco_command(
    *,
    config: ComparisonPlanConfig,
    variant: VariantPlan,
    run_dir: Path,
    input_json_path: Path,
    seed: int,
) -> list[str]:
    motifs = motifs_for_mode(config, variant.mode)
    dump_dir = run_dir / "disco_output"
    command = [
        config.disco_python,
        str(config.disco_root / "runner" / "inference.py"),
        "experiment=designable",
        f"input_json_path={input_json_path}",
        f"dump_dir={dump_dir}",
        f"seeds=[{seed}]",
        f"sample_diffusion.N_step={config.disco_n_step}",
        f"model.N_cycle={config.n_cycle}",
        "cogen_guidance.enabled=true",
        f"cogen_guidance.motifs={hydra_list(motifs)}",
        "cogen_guidance.save_best.enabled=true",
        "cogen_guidance.save_best.select_best_output=true",
        "cogen_guidance.save_best.motif_score_weight=1.0",
        f"cogen_guidance.save_best.confidence_weight={variant.save_best_confidence_weight}",
    ]
    if config.disco_save_trajectories:
        command.extend(
            [
                "cogen_guidance.trajectory_dump.enabled=true",
                "cogen_guidance.trajectory_dump.save_x1_denoised=true",
                "cogen_guidance.trajectory_dump.save_xt_trajectory=true",
                "cogen_guidance.trajectory_dump.save_final_coordinate=true",
                f"cogen_guidance.trajectory_dump.dtype={config.disco_trajectory_dtype}",
                f"cogen_guidance.trajectory_dump.max_steps={config.disco_trajectory_max_steps}",
            ]
        )
    if variant.mode == "standard":
        command.extend(
            [
                "cogen_guidance.logit_guidance.enabled=true",
                "cogen_guidance.logit_guidance.gradient_enabled=true",
                f"cogen_guidance.logit_guidance.gradient_scale={variant.gradient_scale}",
                f"cogen_guidance.logit_guidance.motif_weight_all={variant.motif_weight_all}",
                f"cogen_guidance.logit_guidance.motif_weight_many={variant.motif_weight_many}",
                f"cogen_guidance.logit_guidance.motif_top_k={variant.motif_top_k}",
                f"cogen_guidance.logit_guidance.motif_temperature={variant.motif_temperature}",
                f"cogen_guidance.logit_guidance.local_rescue_enabled={str(variant.local_rescue_enabled).lower()}",
                f"cogen_guidance.logit_guidance.local_rescue_top_k={variant.local_rescue_top_k}",
                f"cogen_guidance.logit_guidance.local_rescue_bonus={variant.local_rescue_bonus}",
                f"cogen_guidance.logit_guidance.local_rescue_min_probability={variant.local_rescue_min_probability}",
            ]
        )
    elif variant.mode == "immunogen":
        fragments = config.fixed_fragments or []
        command.extend(
            [
                "cogen_guidance.logit_guidance.enabled=false",
                "cogen_guidance.immunogen.enabled=true",
                f"cogen_guidance.immunogen.motifs={hydra_list(motifs)}",
                f"cogen_guidance.immunogen.fixed_fragments={hydra_fixed_fragments(fragments)}",
                "cogen_guidance.immunogen.protect_fixed_fragments=true",
                "cogen_guidance.immunogen.mutable_logit_guidance.enabled=true",
                "cogen_guidance.immunogen.mutable_logit_guidance.gradient_enabled=true",
                f"cogen_guidance.immunogen.mutable_logit_guidance.gradient_scale={variant.gradient_scale}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.motif_weight_all={variant.motif_weight_all}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.motif_weight_many={variant.motif_weight_many}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.motif_top_k={variant.motif_top_k}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.motif_temperature={variant.motif_temperature}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.local_rescue_enabled={str(variant.local_rescue_enabled).lower()}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.local_rescue_top_k={variant.local_rescue_top_k}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.local_rescue_bonus={variant.local_rescue_bonus}",
                f"cogen_guidance.immunogen.mutable_logit_guidance.local_rescue_min_probability={variant.local_rescue_min_probability}",
                "cogen_guidance.immunogen.mutable_logit_guidance.coverage_guided=true",
                f"cogen_guidance.immunogen.spacing_aware={str(variant.immunogen_spacing_aware).lower()}",
            ]
        )
    elif variant.mode == "alternating":
        list1 = config.alternating_list1 or ["GP", "DK"]
        list2 = config.alternating_list2 or ["DR", "VW"]
        command.extend(
            [
                "cogen_guidance.logit_guidance.enabled=false",
                "cogen_guidance.alternating.enabled=true",
                f"cogen_guidance.alternating.motif_list1={hydra_list(list1)}",
                f"cogen_guidance.alternating.motif_list2={hydra_list(list2)}",
                f"cogen_guidance.alternating.expected_start_group={config.expected_start_group}",
                "cogen_guidance.alternating.combined_weight=1.0",
            ]
        )
        if variant.alternating_guidance_mode == "slot":
            command.extend(
                [
                    "cogen_guidance.alternating.logit_guidance_enabled=true",
                    "cogen_guidance.alternating.transition_logit_guidance_enabled=false",
                    f"cogen_guidance.alternating.logit_gradient_scale={variant.alternating_gradient_scale}",
                    "cogen_guidance.alternating.logit_slot_gap=3.0",
                    "cogen_guidance.alternating.logit_tail_weight=0.25",
                ]
            )
        elif variant.alternating_guidance_mode == "transition":
            command.extend(
                [
                    "cogen_guidance.alternating.logit_guidance_enabled=false",
                    "cogen_guidance.alternating.transition_logit_guidance_enabled=true",
                    f"cogen_guidance.alternating.transition_logit_gradient_scale={variant.alternating_gradient_scale}",
                    "cogen_guidance.alternating.transition_same_group_penalty_weight=0.25",
                    "cogen_guidance.alternating.transition_soft_start_weight=0.20",
                    "cogen_guidance.alternating.transition_soft_end_weight=0.20",
                    "cogen_guidance.alternating.transition_distance_temperature=8.0",
                    "cogen_guidance.alternating.transition_min_separation=1",
                ]
            )
        else:
            raise ValueError(
                f"Unknown alternating_guidance_mode: {variant.alternating_guidance_mode}"
            )
    else:
        raise ValueError(f"Unknown mode: {variant.mode}")
    return command


def build_disco_analysis_command(
    *,
    config: ComparisonPlanConfig,
    variant: VariantPlan,
    run_dir: Path,
) -> list[str]:
    motifs = motifs_for_mode(config, variant.mode)
    command = [
        config.disco_python,
        str(config.disco_root / "runner" / "guidance_analyze_disco.py"),
        "--dump-dir",
        str(run_dir / "disco_output"),
        "--motifs",
        ",".join(motifs),
        "--out-csv",
        str(run_dir / "disco_analysis.csv"),
    ]
    if variant.mode == "immunogen":
        for fragment in config.fixed_fragments or []:
            label = f":{fragment.label}" if fragment.label else ""
            command.extend(
                [
                    "--fixed-fragment",
                    f"{fragment.start_1based}:{fragment.sequence}{label}",
                ]
            )
    if variant.mode == "alternating":
        command.extend(
            [
                "--alternating-list1",
                ",".join(config.alternating_list1 or ["GP", "DK"]),
                "--alternating-list2",
                ",".join(config.alternating_list2 or ["DR", "VW"]),
                "--expected-start-group",
                config.expected_start_group,
            ]
        )
    return command


def build_pg_standard_command(
    *,
    config: ComparisonPlanConfig,
    variant: VariantPlan,
    run_dir: Path,
    replicate: int,
) -> list[str]:
    motifs = motifs_for_mode(config, variant.mode)
    command = [
        config.pg_python,
        str(config.pg_root / "code" / "protein_generator" / "inference.py"),
        "--sequence",
        "X" * length_for_mode(config, variant.mode),
        "--num_designs",
        str(config.designs_per_replicate),
        "--start_num",
        str(replicate * config.designs_per_replicate),
        "--T",
        str(pg_T_for_mode(config, variant.mode)),
        "--n_cycle",
        str(config.n_cycle),
        "--out",
        str(run_dir / "design"),
        "--potentials",
        variant.pg_potential,
        "--potential_scale",
        str(variant.pg_potential_scale),
        "--motifs",
        ",".join(motifs),
        "--motif_weight_all",
        str(variant.motif_weight_all),
        "--motif_weight_many",
        str(variant.motif_weight_many),
        "--motif_top_k",
        str(variant.motif_top_k),
        "--motif_temperature",
        str(variant.motif_temperature),
        "--motif_score_method",
        "gather_mean",
        "--save_args",
        "--trace_timesteps",
    ]
    if config.pg_checkpoint is not None:
        command.extend(["--checkpoint", str(config.pg_checkpoint)])
    if variant.local_rescue_enabled:
        command.extend(
            [
                "--motif_local_search",
                "--motif_search_top_k",
                str(variant.local_rescue_top_k),
                "--motif_search_bonus",
                str(variant.local_rescue_bonus),
                "--motif_search_min_probability",
                str(variant.local_rescue_min_probability),
                "--motif_search_require_score_improvement",
                "--save_best_combined",
                "--combined_plddt_weight",
                "0.2",
                "--combined_motif_weight",
                "0.8",
            ]
        )
    elif config.pg_save_best_motif_only:
        command.extend(
            [
                "--save_best_combined",
                "--combined_plddt_weight",
                "0.0",
                "--combined_motif_weight",
                "1.0",
                "--combined_motif_all_weight",
                "1.0",
                "--combined_motif_many_weight",
                "0.0",
            ]
        )
    return command


def build_pg_immunogen_command(
    *,
    config: ComparisonPlanConfig,
    variant: VariantPlan,
    run_dir: Path,
) -> list[str]:
    if (
        config.pg_seed_pdb is None
        or config.pg_seed_sequence is None
        or config.pg_fixed_fragments_json is None
    ):
        return []
    if config.pg_immunogen_direct:
        seed_trb = config.pg_seed_trb
        if seed_trb is None and config.pg_seed_pdb is not None:
            inferred_seed_trb = config.pg_seed_pdb.with_suffix(".trb")
            if inferred_seed_trb.exists():
                seed_trb = inferred_seed_trb
        command = [
            config.pg_python,
            str(config.pg_root / "code" / "protein_generator" / "inference.py"),
            "--sampler",
            "immunogen_repredict_coverage",
            "--pdb",
            str(config.pg_seed_pdb),
        ]
        if seed_trb is not None:
            command.extend(["--trb", str(seed_trb)])
        command.extend(
            [
            "--fixed_fragments_json",
            str(config.pg_fixed_fragments_json),
            "--num_designs",
            str(config.designs_per_replicate),
            "--T",
            str(config.pg_short_T),
            "--n_cycle",
            str(config.n_cycle),
            "--out",
            str(run_dir / "design"),
            "--potentials",
            variant.pg_potential,
            "--potential_scale",
            str(variant.pg_potential_scale),
            "--motifs",
            ",".join(motifs_for_mode(config, "immunogen")),
            "--motif_weight_all",
            "1.0",
            "--motif_weight_many",
            "0.0",
            "--motif_score_method",
            "gather_mean",
            "--motif_local_search",
            "--motif_search_top_k",
            str(max(variant.local_rescue_top_k, 1)),
            "--motif_search_min_probability",
            "0.0",
            "--motif_search_bonus",
            str(variant.local_rescue_bonus),
            "--save_best_combined",
            "--combined_plddt_weight",
            "0.20",
            "--combined_motif_weight",
            "0.80",
            "--combined_motif_all_weight",
            "1.0",
            "--combined_motif_many_weight",
            "0.0",
            "--immunogen_coverage_guided",
            "--immunogen_repredict",
            "--save_args",
            "--trace_timesteps",
            ]
        )
        if config.pg_checkpoint is not None:
            command.extend(["--checkpoint", str(config.pg_checkpoint)])
        if variant.immunogen_spacing_aware:
            command.extend(["--immunogen_spacing_aware"])
        return command
    command = [
        config.pg_python,
        str(config.pg_root / "code" / "run_immunogen_workflow.py"),
        "--seed-pdb",
        str(config.pg_seed_pdb),
        "--fixed-fragments-json",
        str(config.pg_fixed_fragments_json),
        "--seed-sequence",
        config.pg_seed_sequence,
        "--motifs",
        ",".join(motifs_for_mode(config, "immunogen")),
        "--out-dir",
        str(run_dir),
        "--docker-image",
        config.docker_image,
        "--potential",
        variant.pg_potential,
        "--potential-scale",
        str(variant.pg_potential_scale),
        "--T",
        str(config.pg_short_T),
        "--n-cycle",
        str(config.n_cycle),
        "--num-designs",
        str(config.designs_per_replicate),
        "--motif-search-top-k",
        str(max(variant.local_rescue_top_k, 1)),
        "--motif-search-bonus",
        str(variant.local_rescue_bonus),
    ]
    if config.pg_checkpoint is not None:
        command.extend(["--checkpoint", str(config.pg_checkpoint)])
    if variant.immunogen_spacing_aware:
        command.extend(["--sampler", "immunogen_spacing_aware"])
    return command


def build_pg_alternating_command(
    *,
    config: ComparisonPlanConfig,
    variant: VariantPlan,
    run_dir: Path,
) -> list[str]:
    command = [
        config.pg_python,
        str(config.pg_root / "code" / "protein_generator" / "inference.py"),
        "--sampler",
        "alternating_save",
        "--sequence",
        "X" * config.alternating_length,
        "--num_designs",
        str(config.designs_per_replicate),
        "--T",
        str(config.pg_short_T),
        "--n_cycle",
        str(config.n_cycle),
        "--out",
        str(run_dir / "design"),
        "--potentials",
        variant.pg_potential,
        "--potential_scale",
        str(variant.pg_potential_scale),
        "--motif_list1",
        ",".join(config.alternating_list1 or ["GP", "DK"]),
        "--motif_list2",
        ",".join(config.alternating_list2 or ["DR", "VW"]),
        "--alternating_start_group",
        config.expected_start_group,
        "--motifs",
        ",".join(motifs_for_mode(config, "alternating")),
        "--motif_weight_all",
        "1.0",
        "--motif_weight_many",
        "0.0",
        "--motif_score_method",
        "gather_mean",
        "--save_best_combined",
        "--combined_plddt_weight",
        "0.20",
        "--combined_motif_weight",
        "0.80",
        "--alternating_combined_weight",
        "0.60",
        "--alternating_start_weight",
        "0.30",
        "--alternating_terminal_weight",
        "0.30",
        "--alternating_score_weight",
        "0.35",
        "--alternating_balance_weight",
        "0.15",
        "--alternating_unique_weight",
        "0.20",
        "--save_args",
        "--trace_timesteps",
    ]
    if config.pg_checkpoint is not None:
        command.extend(["--checkpoint", str(config.pg_checkpoint)])
    return command


def build_pg_analysis_command(
    *,
    config: ComparisonPlanConfig,
    variant: VariantPlan,
    run_dir: Path,
) -> list[str]:
    if variant.mode == "standard":
        return [
            config.pg_python,
            str(config.pg_root / "code" / "analyze_motif_outputs.py"),
            "--pdb-dir",
            str(run_dir),
            "--motifs",
            ",".join(motifs_for_mode(config, "standard")),
            "--out",
            str(run_dir / "motif_summary.csv"),
        ]
    if variant.mode == "alternating":
        return command_with_env(
            {"PYTHONPATH": str(config.pg_root / "code")},
            [
            config.pg_python,
            "-m",
            "protein_generator.alternating_motifs.analyze",
            "--pdb-dir",
            str(run_dir),
            "--motif-list1",
            ",".join(config.alternating_list1 or ["GP", "DK"]),
            "--motif-list2",
            ",".join(config.alternating_list2 or ["DR", "VW"]),
            "--out",
            str(run_dir / "alternating_summary.csv"),
            "--expected-start-group",
            config.expected_start_group,
            ],
        )
    if variant.mode == "immunogen" and config.pg_fixed_fragments_json is not None:
        command = [
            config.pg_python,
            str(config.pg_root / "code" / "analyze_immunogen_outputs.py"),
            "--pdb-dir",
            str(run_dir),
            "--fixed-fragments-json",
            str(config.pg_fixed_fragments_json),
            "--motifs",
            ",".join(motifs_for_mode(config, "immunogen")),
            "--out",
            str(run_dir / "summary.csv"),
        ]
        if config.pg_seed_sequence is not None:
            command.extend(["--seed-sequence", config.pg_seed_sequence])
        if config.pg_seed_pdb is not None:
            command.extend(["--seed-pdb", str(config.pg_seed_pdb)])
        return command
    return []


def plan_runs(config: ComparisonPlanConfig) -> list[PlannedRun]:
    """Materialize all requested run directories, command files, and inputs."""

    runs: list[PlannedRun] = []
    variants = [
        variant
        for variant in default_variants(config.variant_set)
        if variant.mode in set(config.modes)
    ]
    for variant in variants:
        for system in config.systems:
            for replicate, seed in enumerate(config.seeds):
                run_dir = (
                    config.out_dir
                    / system.lower()
                    / variant.mode
                    / variant.variant
                    / f"replicate_{replicate:03d}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                input_json_path = None
                command: list[str]
                analysis_command: list[str]
                status = "planned"
                if system == "DISCO":
                    if variant.mode == "immunogen" and not config.fixed_fragments:
                        status = "requires_immunogen_fixed_fragments"
                        command = []
                        analysis_command = []
                    else:
                        input_json_path = write_disco_input_json(
                            path=run_dir / "input.json",
                            length=length_for_mode(config, variant.mode),
                            name=f"{variant.mode}_{variant.variant}_replicate_{replicate:03d}",
                            fixed_fragments=(
                                config.fixed_fragments or []
                            ) if variant.mode == "immunogen" else [],
                        )
                        command = build_disco_command(
                            config=config,
                            variant=variant,
                            run_dir=run_dir,
                            input_json_path=input_json_path,
                            seed=seed,
                        )
                        analysis_command = build_disco_analysis_command(
                            config=config,
                            variant=variant,
                            run_dir=run_dir,
                        )
                    summary_csv_path = run_dir / "disco_analysis.csv"
                elif system == "ProteinGenerator":
                    if variant.mode == "standard":
                        command = build_pg_standard_command(
                            config=config,
                            variant=variant,
                            run_dir=run_dir,
                            replicate=replicate,
                        )
                    elif variant.mode == "immunogen":
                        command = build_pg_immunogen_command(
                            config=config,
                            variant=variant,
                            run_dir=run_dir,
                        )
                        if not command:
                            status = "requires_pg_immunogen_seed_inputs"
                    elif variant.mode == "alternating":
                        command = build_pg_alternating_command(
                            config=config,
                            variant=variant,
                            run_dir=run_dir,
                        )
                    else:
                        raise ValueError(f"Unknown mode: {variant.mode}")
                    analysis_command = build_pg_analysis_command(
                        config=config,
                        variant=variant,
                        run_dir=run_dir,
                    )
                    summary_csv_path = (
                        run_dir / "summary.csv"
                        if variant.mode == "immunogen"
                        else run_dir / (
                            "alternating_summary.csv"
                            if variant.mode == "alternating"
                            else "motif_summary.csv"
                        )
                    )
                else:
                    raise ValueError(f"Unknown system: {system}")

                planned = PlannedRun(
                    system=system,
                    mode=variant.mode,
                    variant=variant.variant,
                    replicate=replicate,
                    seed=seed,
                    run_dir=run_dir,
                    command=command,
                    analysis_command=analysis_command,
                    status=status,
                    input_json_path=input_json_path,
                    summary_csv_path=summary_csv_path,
                    parameters={
                        **asdict(variant),
                        "n_cycle": config.n_cycle,
                        "disco_n_step": config.disco_n_step,
                        "pg_T": pg_T_for_mode(config, variant.mode),
                        "length": length_for_mode(config, variant.mode),
                        "designs_per_replicate": config.designs_per_replicate,
                        "motifs": motifs_for_mode(config, variant.mode),
                    },
                    fairness_notes=common_fairness_notes(config, variant.mode),
                )
                write_planned_run_files(planned)
                runs.append(planned)
    write_plan_outputs(config, runs)
    return runs


def write_planned_run_files(run: PlannedRun) -> None:
    """Write per-run command, analysis command, and manifest files."""

    (run.run_dir / "command.txt").write_text(run.command_text() + "\n")
    (run.run_dir / "analysis_command.txt").write_text(
        run.analysis_command_text() + "\n"
    )
    manifest = run.csv_row()
    manifest["command"] = run.command
    manifest["analysis_command"] = run.analysis_command
    manifest["parameters"] = run.parameters
    manifest["fairness_notes"] = run.fairness_notes
    (run.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_plan_outputs(config: ComparisonPlanConfig, runs: list[PlannedRun]) -> None:
    """Write matrix-level CSV/JSON plus a comparison command scaffold."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run.csv_row() for run in runs]
    fieldnames = list(rows[0].keys()) if rows else []
    with (config.out_dir / "planned_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "run_type": "pg_vs_disco_guidance_plan",
        "variant_set": config.variant_set,
        "systems": config.systems,
        "modes": config.modes,
        "seeds": config.seeds,
        "n_cycle": config.n_cycle,
        "disco_n_step": config.disco_n_step,
        "pg_standard_T": config.pg_standard_T,
        "pg_short_T": config.pg_short_T,
        "num_planned_runs": len(runs),
        "notes": [
            "This planner does not execute inference.",
            "Run command.txt first, then analysis_command.txt for each completed run.",
            "Use runner/guidance_compare.py after summary CSVs exist.",
        ],
    }
    (config.out_dir / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    comparison_sources = [
        f"{run.system}:{run.mode}:{run.variant}:{run.summary_csv_path}"
        for run in runs
    ]
    compare_command = [
        config.disco_python,
        str(config.disco_root / "runner" / "guidance_compare.py"),
    ]
    for source in comparison_sources:
        compare_command.extend(["--source", source])
    compare_command.extend(
        [
            "--out-normalized",
            str(config.out_dir / "normalized_designs.csv"),
            "--out-summary",
            str(config.out_dir / "comparison_summary.csv"),
            "--allow-missing",
        ]
    )
    (config.out_dir / "compare_command.txt").write_text(
        command_to_string(compare_command) + "\n"
    )
