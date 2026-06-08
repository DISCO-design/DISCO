"""Outer-loop evolutionary search helpers for DISCO motif guidance.

ProteinGenerator's Approach 3 was not a sampler or potential. It was an outer
loop over many inference runs: generate parameter sets, launch inference, score
hard outputs, keep elites, mutate, repeat. This module ports that idea to DISCO
without running inference at import time and without touching DISCO's sampler.
"""

from __future__ import annotations

import math
import random
import json
import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")

N_STEP_CHOICES = [50, 100, 150, 200]
N_CYCLE_CHOICES = [2, 4, 6]
GRADIENT_SCALE_CHOICES = [0.5, 1.0, 2.0, 3.0, 5.0]
WEIGHT_CHOICES = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.2, 0.8),
    (0.0, 1.0),
]
TOP_K_CHOICES = [1, 2, 3, 4]
TEMPERATURE_CHOICES = [0.25, 0.5, 0.75]
LOCAL_RESCUE_TOP_K_CHOICES = [2, 3, 4]
LOCAL_RESCUE_BONUS_CHOICES = [0.5, 1.0, 1.5]
LOCAL_RESCUE_MIN_PROBABILITY_CHOICES = [0.02, 0.03, 0.05]
CONFIDENCE_WEIGHT_CHOICES = [0.0, 0.25, 0.5, 0.75]


@dataclass(frozen=True)
class DiscoGuidanceGenome:
    """One DISCO guidance parameter set in the evolutionary search."""

    n_step: int = 200
    n_cycle: int = 4
    gradient_scale: float = 1.0
    motif_weight_all: float = 0.5
    motif_weight_many: float = 0.5
    motif_top_k: int = 3
    motif_temperature: float = 0.5
    local_rescue_enabled: bool = False
    local_rescue_top_k: int = 3
    local_rescue_bonus: float = 1.0
    local_rescue_min_probability: float = 0.03
    save_best_motif_weight: float = 1.0
    save_best_confidence_weight: float = 0.0
    template_sequence: str | None = None

    def label(self) -> str:
        label = (
            f"step{self.n_step}_cycle{self.n_cycle}_gs{self.gradient_scale:g}_"
            f"a{self.motif_weight_all:g}_m{self.motif_weight_many:g}_"
            f"k{self.motif_top_k}_t{self.motif_temperature:g}"
        )
        if self.local_rescue_enabled:
            label += (
                f"_rescue{self.local_rescue_top_k}_"
                f"b{self.local_rescue_bonus:g}_"
                f"p{self.local_rescue_min_probability:g}"
            )
        if self.save_best_confidence_weight > 0:
            label += f"_conf{self.save_best_confidence_weight:g}"
        return label

    def as_overrides(
        self,
        motifs: list[str],
        dump_dir: Path,
        input_json_path: Path,
        seed: int,
        experiment: str = "designable",
    ) -> list[str]:
        """Translate this genome into Hydra overrides for `runner/inference.py`."""

        motif_list = ",".join(motifs)
        overrides = [
            f"experiment={experiment}",
            f"input_json_path={input_json_path}",
            f"dump_dir={dump_dir}",
            f"seeds=[{seed}]",
            f"sample_diffusion.N_step={self.n_step}",
            f"model.N_cycle={self.n_cycle}",
            "cogen_guidance.enabled=true",
            f"cogen_guidance.motifs=[{motif_list}]",
            "cogen_guidance.logit_guidance.enabled=true",
            "cogen_guidance.logit_guidance.gradient_enabled=true",
            f"cogen_guidance.logit_guidance.gradient_scale={self.gradient_scale}",
            f"cogen_guidance.logit_guidance.motif_weight_all={self.motif_weight_all}",
            f"cogen_guidance.logit_guidance.motif_weight_many={self.motif_weight_many}",
            f"cogen_guidance.logit_guidance.motif_top_k={self.motif_top_k}",
            f"cogen_guidance.logit_guidance.motif_temperature={self.motif_temperature}",
            f"cogen_guidance.logit_guidance.local_rescue_enabled={str(self.local_rescue_enabled).lower()}",
            f"cogen_guidance.logit_guidance.local_rescue_top_k={self.local_rescue_top_k}",
            f"cogen_guidance.logit_guidance.local_rescue_bonus={self.local_rescue_bonus}",
            "cogen_guidance.logit_guidance.local_rescue_only_missing=true",
            "cogen_guidance.save_best.enabled=true",
            "cogen_guidance.save_best.select_best_output=true",
            f"cogen_guidance.save_best.motif_score_weight={self.save_best_motif_weight}",
            f"cogen_guidance.save_best.confidence_weight={self.save_best_confidence_weight}",
        ]
        overrides.append(
            "cogen_guidance.logit_guidance.local_rescue_min_probability="
            f"{self.local_rescue_min_probability}"
        )
        return overrides


@dataclass
class DiscoEvaluation:
    """One evaluated individual."""

    generation: int
    individual: int
    genome: DiscoGuidanceGenome
    run_dir: Path
    fitness: float
    confidence: float
    unique_motifs: int
    total_motifs: int
    sequence: str
    counts: dict[str, int]
    command: list[str]


def parse_motifs(motifs: str | Iterable[str]) -> list[str]:
    """Normalize and validate motif strings."""

    if isinstance(motifs, str):
        parsed = [motif.strip().upper() for motif in motifs.split(",") if motif.strip()]
    else:
        parsed = [str(motif).strip().upper() for motif in motifs if str(motif).strip()]
    if not parsed:
        raise ValueError("At least one motif is required")
    for motif in parsed:
        invalid = sorted(set(motif) - AMINO_ACIDS)
        if invalid:
            raise ValueError(f"Motif {motif!r} contains non-amino-acid letters: {invalid}")
    return parsed


def choose_neighbor(value, choices):
    index = choices.index(value)
    candidates = []
    if index > 0:
        candidates.append(choices[index - 1])
    if index < len(choices) - 1:
        candidates.append(choices[index + 1])
    return random.choice(candidates or choices)


def with_local_rescue(genome: DiscoGuidanceGenome) -> DiscoGuidanceGenome:
    return replace(genome, local_rescue_enabled=True)


def initial_population(
    population_size: int,
    include_local_rescue: bool = False,
) -> list[DiscoGuidanceGenome]:
    """Return a conservative DISCO motif-guidance seed population."""

    seeds = [
        DiscoGuidanceGenome(100, 4, 1.0, 1.0, 0.0, 1, 0.25),
        DiscoGuidanceGenome(200, 4, 1.0, 0.0, 1.0, 3, 0.25),
        DiscoGuidanceGenome(200, 4, 2.0, 0.4, 0.6, 3, 0.5),
        DiscoGuidanceGenome(150, 4, 2.0, 0.6, 0.4, 2, 0.5),
        DiscoGuidanceGenome(100, 2, 1.0, 0.8, 0.2, 2, 0.5),
        DiscoGuidanceGenome(200, 6, 1.0, 0.4, 0.6, 3, 0.75),
    ]
    if include_local_rescue:
        seeds = seeds + [with_local_rescue(seed) for seed in seeds[:3]]
    while len(seeds) < population_size:
        seeds.append(
            mutate_genome(
                random.choice(seeds),
                mutation_rate=1.0,
                include_local_rescue=include_local_rescue,
            )
        )
    return seeds[:population_size]


def mutate_genome(
    genome: DiscoGuidanceGenome,
    mutation_rate: float,
    include_local_rescue: bool = False,
    fixed_n_step: int | None = None,
    fixed_n_cycle: int | None = None,
) -> DiscoGuidanceGenome:
    """Return a nearby genome, preserving fixed comparison knobs if requested."""

    child = genome
    if random.random() <= mutation_rate:
        mutation_types = [
            "gradient_scale",
            "weights",
            "top_k",
            "temperature",
            "confidence_weight",
        ]
        if fixed_n_step is None:
            mutation_types.append("n_step")
        if fixed_n_cycle is None:
            mutation_types.append("n_cycle")
        if include_local_rescue:
            mutation_types.append("local_rescue")

        for mutation_type in random.sample(mutation_types, k=random.choice([1, 1, 2])):
            if mutation_type == "n_step":
                child = replace(child, n_step=choose_neighbor(child.n_step, N_STEP_CHOICES))
            elif mutation_type == "n_cycle":
                child = replace(child, n_cycle=choose_neighbor(child.n_cycle, N_CYCLE_CHOICES))
            elif mutation_type == "gradient_scale":
                child = replace(
                    child,
                    gradient_scale=choose_neighbor(
                        child.gradient_scale,
                        GRADIENT_SCALE_CHOICES,
                    ),
                )
            elif mutation_type == "weights":
                weights = choose_neighbor(
                    (child.motif_weight_all, child.motif_weight_many),
                    WEIGHT_CHOICES,
                )
                child = replace(
                    child,
                    motif_weight_all=weights[0],
                    motif_weight_many=weights[1],
                )
            elif mutation_type == "top_k":
                child = replace(
                    child,
                    motif_top_k=choose_neighbor(child.motif_top_k, TOP_K_CHOICES),
                )
            elif mutation_type == "temperature":
                child = replace(
                    child,
                    motif_temperature=choose_neighbor(
                        child.motif_temperature,
                        TEMPERATURE_CHOICES,
                    ),
                )
            elif mutation_type == "confidence_weight":
                confidence = choose_neighbor(
                    child.save_best_confidence_weight,
                    CONFIDENCE_WEIGHT_CHOICES,
                )
                child = replace(
                    child,
                    save_best_confidence_weight=confidence,
                    save_best_motif_weight=1.0,
                )
            elif mutation_type == "local_rescue":
                if not child.local_rescue_enabled:
                    child = replace(child, local_rescue_enabled=True)
                else:
                    child = replace(
                        child,
                        local_rescue_top_k=choose_neighbor(
                            child.local_rescue_top_k,
                            LOCAL_RESCUE_TOP_K_CHOICES,
                        ),
                        local_rescue_bonus=choose_neighbor(
                            child.local_rescue_bonus,
                            LOCAL_RESCUE_BONUS_CHOICES,
                        ),
                        local_rescue_min_probability=choose_neighbor(
                            child.local_rescue_min_probability,
                            LOCAL_RESCUE_MIN_PROBABILITY_CHOICES,
                        ),
                    )

    if fixed_n_step is not None:
        child = replace(child, n_step=fixed_n_step)
    if fixed_n_cycle is not None:
        child = replace(child, n_cycle=fixed_n_cycle)
    return child


def compatible_merge_templates(first: str | None, second: str | None) -> str | None:
    if first is None:
        return second
    if second is None:
        return first
    if len(first) != len(second):
        return first

    merged = []
    for a, b in zip(first, second):
        if a == "X":
            merged.append(b)
        elif b == "X" or a == b:
            merged.append(a)
        else:
            return first
    return "".join(merged)


def crossover(
    first: DiscoGuidanceGenome,
    second: DiscoGuidanceGenome,
) -> DiscoGuidanceGenome:
    all_weight, many_weight = random.choice(
        [
            (first.motif_weight_all, first.motif_weight_many),
            (second.motif_weight_all, second.motif_weight_many),
        ]
    )
    return DiscoGuidanceGenome(
        n_step=random.choice([first.n_step, second.n_step]),
        n_cycle=random.choice([first.n_cycle, second.n_cycle]),
        gradient_scale=random.choice([first.gradient_scale, second.gradient_scale]),
        motif_weight_all=all_weight,
        motif_weight_many=many_weight,
        motif_top_k=random.choice([first.motif_top_k, second.motif_top_k]),
        motif_temperature=random.choice(
            [first.motif_temperature, second.motif_temperature]
        ),
        local_rescue_enabled=random.choice(
            [first.local_rescue_enabled, second.local_rescue_enabled]
        ),
        local_rescue_top_k=random.choice(
            [first.local_rescue_top_k, second.local_rescue_top_k]
        ),
        local_rescue_bonus=random.choice(
            [first.local_rescue_bonus, second.local_rescue_bonus]
        ),
        local_rescue_min_probability=random.choice(
            [
                first.local_rescue_min_probability,
                second.local_rescue_min_probability,
            ]
        ),
        save_best_motif_weight=random.choice(
            [first.save_best_motif_weight, second.save_best_motif_weight]
        ),
        save_best_confidence_weight=random.choice(
            [first.save_best_confidence_weight, second.save_best_confidence_weight]
        ),
        template_sequence=compatible_merge_templates(
            first.template_sequence,
            second.template_sequence,
        ),
    )


def motif_template_from_sequence(sequence: str, motifs: Iterable[str]) -> str:
    """Return a template with motif residues fixed and all other positions as X."""

    template = ["X"] * len(sequence)
    for motif in motifs:
        for start in range(len(sequence) - len(motif) + 1):
            if sequence[start : start + len(motif)] != motif:
                continue
            compatible = True
            for offset, aa in enumerate(motif):
                existing = template[start + offset]
                if existing not in ["X", aa]:
                    compatible = False
                    break
            if not compatible:
                continue
            for offset, aa in enumerate(motif):
                template[start + offset] = aa
    return "".join(template)


def template_to_disco_sequence(template_sequence: str) -> str:
    """Convert PG-style template `X` positions into DISCO mask `-` positions."""

    converted = []
    for raw in template_sequence.upper():
        if raw in {"X", "-"}:
            converted.append("-")
        elif raw in AMINO_ACIDS:
            converted.append(raw)
        else:
            raise ValueError(f"Invalid template residue: {raw!r}")
    return "".join(converted)


def apply_template_to_input_records(
    records: list[dict],
    template_sequence: str,
) -> list[dict]:
    """Return DISCO input records compatible with a motif template.

    DISCO uses `-` for masked designable protein positions. ProteinGenerator's
    evolutionary template convention uses `X` for "do not fix this position".
    This function converts `X` to `-`, replaces the first protein chain in each
    length-compatible record, and drops records whose first protein chain has a
    different length.
    """

    disco_sequence = template_to_disco_sequence(template_sequence)
    updated_records = []
    for record in records:
        candidate = copy.deepcopy(record)
        replaced = False
        for entity in candidate.get("sequences", []):
            protein = entity.get("proteinChain")
            if protein is None:
                continue
            if len(protein.get("sequence", "")) != len(disco_sequence):
                break
            protein["sequence"] = disco_sequence
            candidate["name"] = f"{candidate.get('name', 'sample')}_template"
            replaced = True
            break
        if replaced:
            updated_records.append(candidate)

    if not updated_records:
        raise ValueError(
            "No DISCO input records have a proteinChain length matching "
            f"template length {len(disco_sequence)}"
        )
    return updated_records


def materialize_template_input_json(
    *,
    base_input_json_path: Path,
    output_json_path: Path,
    template_sequence: str | None,
) -> Path:
    """Write a template-modified DISCO input JSON and return its path.

    If `template_sequence` is `None`, return the original input path without
    writing a new file.
    """

    if template_sequence is None:
        return base_input_json_path

    records = json.loads(base_input_json_path.read_text())
    if not isinstance(records, list):
        raise ValueError("DISCO input JSON root must be a list")
    updated_records = apply_template_to_input_records(records, template_sequence)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(updated_records, indent=2) + "\n")
    return output_json_path


def fitness_score(
    *,
    confidence: float,
    unique_motifs: int,
    total_motifs: int,
    required_motifs: int,
    target_total_motifs: int,
    mode: str,
) -> float:
    """Score an evaluated DISCO design from confidence and hard motif hits."""

    unique_fraction = unique_motifs / max(required_motifs, 1)
    total_score = min(total_motifs / max(target_total_motifs, 1), 1.0)
    if mode == "balance":
        return 0.55 * confidence + 0.30 * unique_fraction + 0.15 * total_score
    if mode == "max_motifs":
        return 0.20 * confidence + 0.20 * unique_fraction + 0.60 * total_score
    if mode == "high_confidence":
        return 0.75 * confidence + 0.20 * unique_fraction + 0.05 * total_score
    raise ValueError(f"Unknown fitness mode: {mode}")


def tournament_select(
    evaluations: list[DiscoEvaluation],
    tournament_size: int = 3,
) -> DiscoEvaluation:
    sample_size = min(tournament_size, len(evaluations))
    candidates = random.sample(evaluations, sample_size)
    return max(candidates, key=lambda item: item.fitness)


def next_generation(
    *,
    evaluations: list[DiscoEvaluation],
    population_size: int,
    elite_fraction: float,
    mutation_rate: float,
    motifs: list[str],
    template_from_elites: bool,
    include_local_rescue: bool,
    fixed_n_step: int | None = None,
    fixed_n_cycle: int | None = None,
) -> list[DiscoGuidanceGenome]:
    """Build the next generation from elites, crossover, and mutation."""

    ranked = sorted(evaluations, key=lambda item: item.fitness, reverse=True)
    elite_count = max(1, math.ceil(population_size * elite_fraction))
    elites = ranked[:elite_count]

    next_population = []
    for elite in elites:
        template = None
        if template_from_elites and elite.sequence:
            template = motif_template_from_sequence(elite.sequence, motifs)
        next_population.append(replace(elite.genome, template_sequence=template))

    while len(next_population) < population_size:
        first = tournament_select(ranked)
        second = tournament_select(ranked)
        child = crossover(first.genome, second.genome)
        child = mutate_genome(
            child,
            mutation_rate=mutation_rate,
            include_local_rescue=include_local_rescue,
            fixed_n_step=fixed_n_step,
            fixed_n_cycle=fixed_n_cycle,
        )
        if template_from_elites and first.sequence and random.random() < 0.5:
            child = replace(
                child,
                template_sequence=motif_template_from_sequence(first.sequence, motifs),
            )
        next_population.append(child)

    return next_population[:population_size]


def build_inference_command(
    *,
    python_executable: str,
    disco_root: Path,
    genome: DiscoGuidanceGenome,
    motifs: list[str],
    input_json_path: Path,
    dump_dir: Path,
    seed: int,
    experiment: str = "designable",
) -> list[str]:
    """Build the DISCO inference command for one genome."""

    return [
        python_executable,
        str(disco_root / "runner" / "inference.py"),
        *genome.as_overrides(
            motifs=motifs,
            dump_dir=dump_dir,
            input_json_path=input_json_path,
            seed=seed,
            experiment=experiment,
        ),
    ]


def sequence_from_disco_sequence_text(text: str) -> str:
    """Extract concatenated protein sequence from DISCO sequence output text."""

    residues = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(">") or line.startswith("ligand_smiles"):
            continue
        if line.startswith("CHAIN "):
            parts = dict(
                item.split("=", 1)
                for item in line.split()
                if "=" in item
            )
            if parts.get("type") == "protein" and "seq" in parts:
                residues.append(parts["seq"])
            continue
        if set(line).issubset(AMINO_ACIDS):
            residues.append(line)
    return "".join(residues)


def evaluation_row(evaluation: DiscoEvaluation, motifs: list[str]) -> dict[str, object]:
    """Flatten an evaluation for CSV writing."""

    row = {
        "generation": evaluation.generation,
        "individual": evaluation.individual,
        "fitness": f"{evaluation.fitness:.6f}",
        "confidence": f"{evaluation.confidence:.3f}",
        "unique_motifs": evaluation.unique_motifs,
        "total_motifs": evaluation.total_motifs,
        "sequence": evaluation.sequence,
        "run_dir": str(evaluation.run_dir),
        **asdict(evaluation.genome),
    }
    row.pop("template_sequence", None)
    for motif in motifs:
        row[f"{motif}_count"] = evaluation.counts.get(motif, 0)
    return row
