"""Hard sequence scoring helpers for DISCO motif-guided inference.

These utilities mirror the split used in the ProteinGenerator motif fork:
soft/logit guidance nudges sampling, while hard scoring audits exact sequences.
Keeping this layer discrete is important because final DISCO outputs are exact
amino-acid strings, not probability tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from disco.data.constants import MASK_TOKEN_IDX, PRO_RES_IDX_TO_RESNAME_ONE

DEFAULT_MOTIFS = [
    "AHK",
    "KTTKS",
    "GQP",
    "VW",
    "GHK",
    "KVK",
    "GP",
    "VGP",
    "DR",
    "DK",
    "DRLDR",
    "DRLER",
    "GSPLD",
    "PTSLA",
    "STTLE",
]


def parse_motifs(value: str | Iterable[str] | None) -> list[str]:
    """Normalize motif config input to uppercase non-empty strings."""

    if value is None:
        return list(DEFAULT_MOTIFS)
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = [str(part) for part in value]
    motifs = [part.strip().upper() for part in parts if part.strip()]
    if not motifs:
        raise ValueError("At least one motif is required when motif guidance is enabled.")
    return motifs


def parse_weights(
    value: str | Iterable[float] | None,
    n: int,
) -> list[float]:
    """Normalize optional per-motif weights."""

    if value is None:
        weights = [1.0] * n
    elif isinstance(value, str):
        weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        weights = [float(part) for part in value]

    if len(weights) != n:
        raise ValueError("motif_weights must have the same length as motifs.")
    if sum(weights) <= 0:
        raise ValueError("motif_weights must sum to a positive value.")
    return weights


def count_overlapping(sequence: str, motif: str) -> int:
    """Count exact motif hits, including overlaps."""

    return len(motif_positions(sequence, motif))


def motif_positions(sequence: str, motif: str) -> list[int]:
    """Return zero-based exact motif start positions."""

    positions: list[int] = []
    start = 0
    while True:
        idx = sequence.find(motif, start)
        if idx == -1:
            return positions
        positions.append(idx)
        start = idx + 1


def tokens_to_sequence(tokens: torch.Tensor) -> str:
    """Convert DISCO protein token ids to a readable amino-acid sequence.

    Mask and non-protein tokens are represented as ``-`` so partially generated
    intermediate states can still be scored without pretending they contain a
    valid amino acid.
    """

    flat = tokens.detach().cpu().reshape(-1)
    letters = []
    for token in flat:
        idx = int(token.item())
        if idx == MASK_TOKEN_IDX:
            letters.append("-")
        else:
            letters.append(PRO_RES_IDX_TO_RESNAME_ONE.get(idx, "-"))
    return "".join(letters)


@dataclass(frozen=True)
class MotifScore:
    """Hard exact-sequence motif score and audit details."""

    motif_score: float
    all_score: float
    many_score: float
    num_unique: int
    num_total: int
    counts: dict[str, int]
    positions: dict[str, list[int]]
    present: list[str]
    missing: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "motif_score": self.motif_score,
            "all_score": self.all_score,
            "many_score": self.many_score,
            "num_unique": self.num_unique,
            "num_total": self.num_total,
            "counts": self.counts,
            "positions": self.positions,
            "present": self.present,
            "missing": self.missing,
        }


class HardMotifScorer:
    """Score exact motif occurrences in a concrete amino-acid sequence."""

    def __init__(
        self,
        motifs: str | Iterable[str] | None = None,
        motif_weights: str | Iterable[float] | None = None,
        motif_all_weight: float = 0.8,
        motif_many_weight: float = 0.2,
        motif_many_cap: int = 0,
    ) -> None:
        self.motifs = parse_motifs(motifs)
        self.motif_weights = parse_weights(motif_weights, len(self.motifs))
        self.motif_all_weight = float(motif_all_weight)
        self.motif_many_weight = float(motif_many_weight)
        self.motif_many_cap = int(motif_many_cap)
        self.weight_sum = sum(self.motif_weights)

    def score_sequence(self, sequence: str) -> MotifScore:
        sequence = sequence.upper()
        counts: dict[str, int] = {}
        positions: dict[str, list[int]] = {}
        weighted_present = 0.0
        weighted_many = 0.0
        weighted_many_max = 0.0

        for motif, weight in zip(self.motifs, self.motif_weights, strict=True):
            starts = motif_positions(sequence, motif)
            count = len(starts)
            counts[motif] = count
            positions[motif] = starts

            if count > 0:
                weighted_present += weight

            possible_count = max(len(sequence) - len(motif) + 1, 0)
            if self.motif_many_cap > 0:
                many_max = min(self.motif_many_cap, possible_count)
                many_count = min(count, self.motif_many_cap)
            else:
                many_max = possible_count
                many_count = count

            weighted_many += weight * many_count
            weighted_many_max += weight * many_max

        all_score = weighted_present / self.weight_sum
        many_score = weighted_many / weighted_many_max if weighted_many_max > 0 else 0.0
        motif_score = (
            self.motif_all_weight * all_score
            + self.motif_many_weight * many_score
        )
        present = [motif for motif in self.motifs if counts[motif] > 0]
        missing = [motif for motif in self.motifs if counts[motif] == 0]

        return MotifScore(
            motif_score=float(motif_score),
            all_score=float(all_score),
            many_score=float(many_score),
            num_unique=len(present),
            num_total=sum(counts.values()),
            counts=counts,
            positions=positions,
            present=present,
            missing=missing,
        )
