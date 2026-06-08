"""Immunogen fixed-fragment helpers for DISCO motif experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from disco.model.cogen_inference.guidance.motif import MotifLogitGuidance

AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")


@dataclass(frozen=True)
class FixedFragment:
    start_1based: int
    sequence: str
    label: str = ""

    @property
    def start_0based(self) -> int:
        return self.start_1based - 1

    @property
    def end_0based_exclusive(self) -> int:
        return self.start_0based + len(self.sequence)


@dataclass(frozen=True)
class ImmunogenFixedSpec:
    fixed_fragments: list[FixedFragment]


def _load_fragment(raw: dict[str, Any], index: int) -> FixedFragment:
    if "start_1based" not in raw:
        raise ValueError(f"fixed_fragments[{index}] is missing start_1based")
    if "sequence" not in raw:
        raise ValueError(f"fixed_fragments[{index}] is missing sequence")

    start_1based = int(raw["start_1based"])
    sequence = str(raw["sequence"]).strip().upper()
    label = str(raw.get("label", "")).strip()

    if start_1based < 1:
        raise ValueError(f"fixed_fragments[{index}].start_1based must be >= 1")
    if not sequence:
        raise ValueError(f"fixed_fragments[{index}].sequence must not be empty")
    invalid = sorted(set(sequence) - AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"fixed_fragments[{index}].sequence contains non-standard amino acids: {invalid}"
        )

    return FixedFragment(
        start_1based=start_1based,
        sequence=sequence,
        label=label,
    )


def spec_from_dict(data: dict[str, Any]) -> ImmunogenFixedSpec:
    raw_fragments = data.get("fixed_fragments")
    if isinstance(raw_fragments, ListConfig):
        raw_fragments = OmegaConf.to_container(raw_fragments, resolve=True)
    if not isinstance(raw_fragments, list) or len(raw_fragments) == 0:
        raise ValueError("fixed_fragments must be a non-empty list")
    fragments = []
    for index, raw in enumerate(raw_fragments):
        if isinstance(raw, DictConfig):
            raw = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(f"fixed_fragments[{index}] must be an object")
        fragments.append(_load_fragment(raw, index))
    return ImmunogenFixedSpec(fixed_fragments=fragments)


def build_fixed_sequence_mask(
    sequence: str,
    spec: ImmunogenFixedSpec,
) -> torch.Tensor:
    """Return a bool mask where True means fixed/protected."""

    sequence = sequence.upper()
    mask = torch.zeros(len(sequence), dtype=torch.bool)
    occupied_by: list[str | None] = [None] * len(sequence)

    for fragment in spec.fixed_fragments:
        start = fragment.start_0based
        end = fragment.end_0based_exclusive
        label = fragment.label or fragment.sequence
        if end > len(sequence):
            raise ValueError(
                f"Fixed fragment {label!r} at {fragment.start_1based} extends past "
                f"sequence length {len(sequence)}"
            )
        observed = sequence[start:end]
        if observed != fragment.sequence:
            raise ValueError(
                f"Fixed fragment {label!r} does not match seed sequence at "
                f"{fragment.start_1based}: expected {fragment.sequence}, observed {observed}"
            )
        for offset, aa in enumerate(fragment.sequence):
            position = start + offset
            owner = occupied_by[position]
            if owner is not None and sequence[position] != aa:
                raise ValueError(
                    f"Conflicting fixed fragments at position {position + 1}: "
                    f"{owner!r} and {label!r}"
                )
            occupied_by[position] = label
            mask[position] = True

    return mask


def build_fixed_position_mask(
    length: int,
    spec: ImmunogenFixedSpec,
) -> torch.Tensor:
    """Return a bool fixed-position mask from fragment coordinates only.

    `build_fixed_sequence_mask` validates a fixed-fragment JSON against a known
    seed sequence. During DISCO inference we often only need the protected
    coordinates to block updates or motif windows. This helper intentionally
    does not validate sequence identity; it only checks that the fragments fit.
    """

    mask = torch.zeros(length, dtype=torch.bool)
    occupied_by: list[str | None] = [None] * length
    for fragment in spec.fixed_fragments:
        start = fragment.start_0based
        end = fragment.end_0based_exclusive
        label = fragment.label or fragment.sequence
        if end > length:
            raise ValueError(
                f"Fixed fragment {label!r} at {fragment.start_1based} extends past "
                f"sequence length {length}"
            )
        for position in range(start, end):
            owner = occupied_by[position]
            if owner is not None and owner != label:
                raise ValueError(
                    f"Overlapping fixed fragments at position {position + 1}: "
                    f"{owner!r} and {label!r}"
                )
            occupied_by[position] = label
            mask[position] = True
    return mask


class MutableWindowMotifLogitGuidance(MotifLogitGuidance):
    """Motif logit guidance restricted to fully mutable motif windows.

    This is the DISCO-side equivalent of ProteinGenerator's
    `immunogen_mutable_motifs_bias`. It reuses the generic motif objective, but
    zeros the soft score for any candidate motif start whose window overlaps a
    fixed immunogen fragment.
    """

    def __init__(
        self,
        fixed_spec: ImmunogenFixedSpec,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_spec = fixed_spec

    def _motif_window_scores(self, probs: torch.Tensor, motif: str) -> torch.Tensor:
        scores = super()._motif_window_scores(probs, motif)
        if scores.numel() == 0:
            return scores
        valid = self.valid_start_mask(
            length=probs.shape[0],
            motif_len=len(motif),
            device=probs.device,
        )
        if valid.shape[0] != scores.shape[0]:
            raise ValueError(
                "Mutable-window mask and motif scores have different lengths"
            )
        return torch.where(valid, scores, torch.zeros_like(scores))

    def valid_start_mask(
        self,
        length: int,
        motif_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return True for starts where the full motif window is mutable."""

        max_start = length - motif_len + 1
        if max_start <= 0:
            return torch.zeros(0, device=device, dtype=torch.bool)
        fixed = build_fixed_position_mask(length, self.fixed_spec).to(device=device)
        mutable = ~fixed
        starts = []
        for start in range(max_start):
            starts.append(bool(torch.all(mutable[start : start + motif_len]).item()))
        return torch.tensor(starts, device=device, dtype=torch.bool)


def score_fixed_fragments(sequence: str, spec: ImmunogenFixedSpec) -> dict[str, object]:
    """Report whether all fixed fragments are preserved at their positions."""

    sequence = sequence.upper()
    mismatches = []
    for fragment in spec.fixed_fragments:
        start = fragment.start_0based
        end = fragment.end_0based_exclusive
        observed = sequence[start:end] if start < len(sequence) else ""
        if observed != fragment.sequence:
            mismatches.append(
                {
                    "label": fragment.label,
                    "start_1based": fragment.start_1based,
                    "expected": fragment.sequence,
                    "observed": observed,
                }
            )
    return {
        "fixed_fragments_preserved": len(mismatches) == 0,
        "num_fixed_mismatches": len(mismatches),
        "mismatches": mismatches,
    }


def motif_start_positions(sequence: str, motif: str) -> list[int]:
    """Return all zero-based start positions where `motif` appears."""

    positions = []
    start = 0
    while True:
        index = sequence.find(motif, start)
        if index == -1:
            return positions
        positions.append(index)
        start = index + 1


def pairwise_distances(positions: list[int]) -> list[int]:
    """Return absolute pairwise distances between motif start positions."""

    distances = []
    for left_index, left in enumerate(positions):
        for right in positions[left_index + 1 :]:
            distances.append(abs(right - left))
    return distances


def best_motif_start_set(sequence: str, motifs: list[str]) -> list[int]:
    """Choose one occurrence per motif that is as spread out as possible."""

    position_lists = [motif_start_positions(sequence, motif) for motif in motifs]
    if any(len(positions) == 0 for positions in position_lists):
        return []

    best_positions = []
    best_score = (-1, -1)
    for candidate in product(*position_lists):
        unique_positions = sorted(set(candidate))
        if len(unique_positions) < len(candidate):
            continue
        distances = pairwise_distances(unique_positions)
        min_distance = min(distances) if distances else 0
        span = unique_positions[-1] - unique_positions[0] if unique_positions else 0
        score = (min_distance, span)
        if score > best_score:
            best_score = score
            best_positions = unique_positions
    return best_positions


def spacing_metrics(sequence: str, motifs: list[str]) -> dict[str, float | int | str]:
    """Compute motif-spacing fields ported from PG immunogen analysis."""

    motifs = [motif.strip().upper() for motif in motifs if motif.strip()]
    positions = best_motif_start_set(sequence.upper(), motifs)
    if len(positions) < 2:
        return {
            "motif_spacing_positions_1based": "",
            "motif_min_start_distance": "",
            "motif_mean_pairwise_distance": "",
            "motif_start_span": "",
            "motif_spread_score": "",
        }

    distances = pairwise_distances(positions)
    min_distance = min(distances)
    mean_distance = sum(distances) / len(distances)
    span = positions[-1] - positions[0]
    ideal_min_distance = max(1, len(sequence) / max(len(positions), 1))
    spread_score = min(1.0, min_distance / ideal_min_distance)

    return {
        "motif_spacing_positions_1based": ";".join(
            str(position + 1) for position in positions
        ),
        "motif_min_start_distance": min_distance,
        "motif_mean_pairwise_distance": f"{mean_distance:.3f}",
        "motif_start_span": span,
        "motif_spread_score": f"{spread_score:.3f}",
    }


def spacing_score(
    sequence: str,
    motifs: list[str],
    target_distance: float | None = None,
) -> float:
    """Return a scalar spacing score for optional save-score adjustment."""

    metrics = spacing_metrics(sequence, motifs)
    if metrics["motif_spread_score"] == "":
        return 0.0

    if target_distance is None or target_distance <= 0:
        return float(metrics["motif_spread_score"])

    positions = [
        int(raw) - 1
        for raw in str(metrics["motif_spacing_positions_1based"]).split(";")
    ]
    distances = pairwise_distances(positions)
    if not distances:
        return 0.0

    mean_error = sum(abs(distance - target_distance) for distance in distances)
    mean_error /= len(distances)
    return max(0.0, 1.0 - mean_error / max(len(sequence), 1))
