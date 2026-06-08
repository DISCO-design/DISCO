"""Hard scoring and soft logit guidance for alternating motif designs.

This is a direct DISCO-side equivalent of the ProteinGenerator alternating
hard scorer: it only inspects finished or intermediate amino-acid strings and
does not depend on DISCO model internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from disco.model.cogen_inference.guidance.motif import AA_TO_IDX


def parse_motif_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = [str(part) for part in value]
    return [part.strip().upper() for part in parts if part.strip()]


def motif_positions(sequence: str, motif: str) -> list[int]:
    positions = []
    start = 0
    while True:
        idx = sequence.find(motif, start)
        if idx == -1:
            return positions
        positions.append(idx)
        start = idx + 1


@dataclass(frozen=True)
class MotifHit:
    start: int
    end: int
    motif: str
    group: str


def all_motif_hits(
    sequence: str,
    list1: Iterable[str],
    list2: Iterable[str],
) -> list[MotifHit]:
    hits: list[MotifHit] = []
    for group, motifs in [("list1", list1), ("list2", list2)]:
        for motif in motifs:
            for start in motif_positions(sequence, motif):
                hits.append(
                    MotifHit(
                        start=start,
                        end=start + len(motif),
                        motif=motif,
                        group=group,
                    )
                )
    return sorted(hits, key=lambda hit: (hit.start, hit.end, hit.group, hit.motif))


def non_overlapping_hits(hits: Iterable[MotifHit]) -> list[MotifHit]:
    """Pick a deterministic non-overlapping hit path."""

    ordered = sorted(hits, key=lambda hit: (hit.end, hit.start, hit.motif))
    if not ordered:
        return []

    previous = []
    for i, hit in enumerate(ordered):
        j = i - 1
        while j >= 0 and ordered[j].end > hit.start:
            j -= 1
        previous.append(j)

    # Optimize hit count first, then covered motif residues.
    dp: list[tuple[int, int, tuple[int, ...]]] = [
        (0, 0, tuple()) for _ in range(len(ordered) + 1)
    ]
    for i, hit in enumerate(ordered, start=1):
        skip = dp[i - 1]
        base = dp[previous[i - 1] + 1]
        take = (base[0] + 1, base[1] + hit.end - hit.start, base[2] + (i - 1,))
        dp[i] = max(skip, take, key=lambda item: (item[0], item[1], tuple(-x for x in item[2])))

    return sorted([ordered[i] for i in dp[-1][2]], key=lambda hit: (hit.start, hit.end))


def longest_homopolymer(sequence: str) -> int:
    if not sequence:
        return 0
    longest = 1
    current = 1
    for prev, aa in zip(sequence, sequence[1:]):
        if aa == prev:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def score_alternation(
    sequence: str,
    list1: Iterable[str],
    list2: Iterable[str],
    expected_start_group: str = "list1",
) -> dict[str, object]:
    motifs1 = parse_motif_list(list1)
    motifs2 = parse_motif_list(list2)
    hits = non_overlapping_hits(all_motif_hits(sequence.upper(), motifs1, motifs2))
    groups = [hit.group for hit in hits]

    transitions = max(0, len(groups) - 1)
    alternating_transitions = sum(
        1 for left, right in zip(groups, groups[1:]) if left != right
    )
    same_group_transitions = transitions - alternating_transitions
    alternation_score = alternating_transitions / transitions if transitions else 1.0

    list1_count = sum(1 for group in groups if group == "list1")
    list2_count = sum(1 for group in groups if group == "list2")
    motif_count = len(hits)
    balance_score = (
        1.0 - abs(list1_count - list2_count) / motif_count if motif_count else 0.0
    )

    if hits:
        last = hits[-1]
        tail_length = len(sequence) - last.end
        end_state = last.group
        start_state = hits[0].group
    else:
        tail_length = len(sequence)
        end_state = "none"
        start_state = "none"

    bad_tail = end_state == "list1"
    tail_penalty_raw = tail_length if bad_tail else 0
    tail_penalty_norm = tail_penalty_raw / len(sequence) if sequence else 0.0
    terminal_score = 1.0 - tail_penalty_norm
    start_state_score = 1.0 if start_state == expected_start_group else 0.0
    unique_motifs = sorted({hit.motif for hit in hits})

    return {
        "alternation_score": alternation_score,
        "transition_consistency": alternation_score,
        "same_group_transitions": same_group_transitions,
        "alternating_transitions": alternating_transitions,
        "motif_balance_score": balance_score,
        "list1_count": list1_count,
        "list2_count": list2_count,
        "num_motif_hits": motif_count,
        "num_unique_motifs": len(unique_motifs),
        "unique_motifs": ",".join(unique_motifs),
        "motif_pattern": "->".join("1" if group == "list1" else "2" for group in groups),
        "motif_hit_summary": ";".join(
            f"{hit.motif}@{hit.start + 1}:{'1' if hit.group == 'list1' else '2'}"
            for hit in hits
        ),
        "end_state": end_state,
        "start_state": start_state,
        "expected_start_group": expected_start_group,
        "start_state_score": start_state_score,
        "tail_length": tail_length,
        "bad_tail_after_list1": bad_tail,
        "tail_penalty_raw": tail_penalty_raw,
        "tail_penalty_norm": tail_penalty_norm,
        "terminal_score": terminal_score,
        "longest_homopolymer": longest_homopolymer(sequence),
    }


def combined_alternating_score(
    sequence: str,
    list1: Iterable[str],
    list2: Iterable[str],
    alternation_weight: float = 0.45,
    balance_weight: float = 0.20,
    coverage_weight: float = 0.20,
    terminal_weight: float = 0.15,
    start_weight: float = 0.15,
    expected_start_group: str = "list1",
) -> float:
    total_weight = (
        alternation_weight
        + balance_weight
        + coverage_weight
        + terminal_weight
        + start_weight
    )
    if total_weight <= 0:
        return 0.0

    metrics = score_alternation(
        sequence,
        list1,
        list2,
        expected_start_group=expected_start_group,
    )
    expected_unique = len(set(parse_motif_list(list1) + parse_motif_list(list2)))
    coverage = float(metrics["num_unique_motifs"]) / expected_unique if expected_unique else 0.0
    return (
        alternation_weight * float(metrics["alternation_score"])
        + balance_weight * float(metrics["motif_balance_score"])
        + coverage_weight * coverage
        + terminal_weight * float(metrics["terminal_score"])
        + start_weight * float(metrics["start_state_score"])
    ) / total_weight


@dataclass
class AlternatingLogitGuidanceResult:
    """Diagnostic summary for one alternating logit-guidance call."""

    gradient_applied: bool = False
    gradient_norm: float = 0.0
    objective: float = 0.0
    transition: float = 0.0
    coverage: float = 0.0
    tail: float = 0.0
    num_slots: int = 0
    same_group_penalty: float = 0.0
    start: float = 0.0
    end: float = 0.0

    def as_dict(self, prefix: str = "guidance_alternating_logit") -> dict[str, object]:
        data: dict[str, object] = {
            f"{prefix}_applied": self.gradient_applied,
            f"{prefix}_gradient_norm": self.gradient_norm,
            f"{prefix}_objective": self.objective,
            f"{prefix}_transition": self.transition,
            f"{prefix}_coverage": self.coverage,
            f"{prefix}_tail": self.tail,
            f"{prefix}_slots": self.num_slots,
        }
        if self.same_group_penalty or self.start or self.end:
            data.update(
                {
                    f"{prefix}_same_group_penalty": self.same_group_penalty,
                    f"{prefix}_start": self.start,
                    f"{prefix}_end": self.end,
                }
            )
        return data


class AlternatingLogitGuidance:
    """Slot-based ProteinGenerator alternating potential adapted to DISCO logits.

    ProteinGenerator's ``AlternatingMotifsBias`` optimized a differentiable
    ``soft_seq`` tensor. DISCO exposes sequence decoder logits before its
    path-planning sampler, so this class builds the same kind of objective from
    ``softmax(logits / temperature)`` and adds a normalized gradient back to the
    logits. The sampler remains responsible for turning those logits into the
    next discrete sequence state.
    """

    def __init__(
        self,
        motif_list1: str | Iterable[str] | None,
        motif_list2: str | Iterable[str] | None,
        gradient_scale: float = 1.0,
        temperature: float = 0.5,
        max_slots: int = 0,
        slot_gap: float = 3.0,
        slot_width: float = 1.35,
        tail_weight: float = 0.25,
        coverage_weight: float = 0.35,
        transition_weight: float = 0.40,
        start_group: str = "list1",
    ) -> None:
        self.list1 = parse_motif_list(motif_list1)
        self.list2 = parse_motif_list(motif_list2)
        self.gradient_scale = float(gradient_scale)
        self.temperature = float(temperature)
        self.max_slots = int(max_slots)
        self.slot_gap = float(slot_gap)
        self.slot_width = float(slot_width)
        self.tail_weight = float(tail_weight)
        self.coverage_weight = float(coverage_weight)
        self.transition_weight = float(transition_weight)
        self.start_group = str(start_group)
        if self.start_group not in {"list1", "list2"}:
            raise ValueError("alternating start_group must be 'list1' or 'list2'")
        if not self.list1 or not self.list2:
            raise ValueError("motif_list1 and motif_list2 are required")
        for motif in self.list1 + self.list2:
            for aa in motif:
                if aa not in AA_TO_IDX:
                    raise ValueError(f"Unknown amino acid in motif {motif!r}: {aa!r}")
        self.last_result = AlternatingLogitGuidanceResult()

    def apply(
        self,
        logits: torch.Tensor,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits with an alternating-motif gradient added."""

        del xt_seq
        if logits.ndim == 2:
            guided, result = self._apply_single(logits, can_change_idx)
        elif logits.ndim == 3:
            outputs = []
            results = []
            for batch_idx in range(logits.shape[0]):
                change_idx = (
                    can_change_idx[batch_idx]
                    if can_change_idx.ndim == 2
                    else can_change_idx
                )
                guided_b, result_b = self._apply_single(logits[batch_idx], change_idx)
                outputs.append(guided_b)
                results.append(result_b)
            guided = torch.stack(outputs, dim=0)
            result = self._merge_batch_results(results)
        else:
            raise ValueError(
                f"Expected logits with 2 or 3 dims, got {tuple(logits.shape)}"
            )

        self.last_result = result
        return guided

    def objective_from_logits(self, logits: torch.Tensor) -> tuple[float, dict[str, float]]:
        """Compute the current soft objective for tests and diagnostics."""

        with torch.no_grad():
            probs = torch.softmax(logits.float() / max(self.temperature, 1e-6), dim=-1)
            objective, details = self.compute_objective(probs)
        return float(objective.detach().cpu().item()), details

    def compute_objective(
        self,
        probs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the slot/coverage/tail objective from amino-acid probabilities."""

        length = probs.shape[0]
        if length == 0:
            zero = probs.new_tensor(0.0)
            return zero, {
                "objective": 0.0,
                "transition": 0.0,
                "coverage": 0.0,
                "tail": 0.0,
                "num_slots": 0.0,
            }

        list1_scores = self.group_start_scores(probs, self.list1)
        list2_scores = self.group_start_scores(probs, self.list2)
        groups = self.slot_plan(length)

        slot_terms = []
        for slot_index, group in enumerate(groups):
            scores = list1_scores if group == "list1" else list2_scores
            slot_terms.append(
                torch.max(
                    scores * self.slot_mask(length, slot_index, len(groups), probs)
                )
            )
        transition_objective = (
            torch.mean(torch.stack(slot_terms))
            if slot_terms
            else probs.new_tensor(0.0)
        )
        coverage_objective = 0.5 * (torch.max(list1_scores) + torch.max(list2_scores))

        tail_objective = probs.new_tensor(1.0)
        if groups and groups[-1] == "list1":
            positions = self._positions(length, probs)
            tail_norm = (length - positions) / max(1, length)
            probabilities = torch.softmax(
                list1_scores / max(self.temperature, 1e-6),
                dim=0,
            )
            expected_tail = torch.sum(probabilities * tail_norm)
            tail_objective = 1.0 - expected_tail

        objective = (
            self.transition_weight * transition_objective
            + self.coverage_weight * coverage_objective
            + self.tail_weight * tail_objective
        )
        details = {
            "objective": float(objective.detach().cpu().item()),
            "transition": float(transition_objective.detach().cpu().item()),
            "coverage": float(coverage_objective.detach().cpu().item()),
            "tail": float(tail_objective.detach().cpu().item()),
            "num_slots": float(len(groups)),
        }
        return objective, details

    def group_start_scores(self, probs: torch.Tensor, motifs: list[str]) -> torch.Tensor:
        """Return best per-position soft start score for one motif list."""

        length = probs.shape[0]
        scores = probs.new_zeros((length,))
        for motif in motifs:
            motif_score = self.motif_window_scores(probs, motif)
            if motif_score.numel() == 0:
                continue
            padded = probs.new_zeros((length,))
            padded[: motif_score.shape[0]] = motif_score
            scores = torch.maximum(scores, padded)
        return scores

    def motif_window_scores(self, probs: torch.Tensor, motif: str) -> torch.Tensor:
        """Score how likely ``motif`` starts at every valid position."""

        motif_len = len(motif)
        if motif_len > probs.shape[0]:
            return probs.new_zeros((0,))

        motif_filter = probs.new_zeros((1, probs.shape[-1], motif_len))
        for offset, aa in enumerate(motif):
            motif_filter[0, AA_TO_IDX[aa], offset] = 1.0

        x = probs.unsqueeze(0).permute(0, 2, 1)
        return F.conv1d(x, motif_filter).squeeze(0).squeeze(0) / motif_len

    def slot_plan(self, length: int) -> list[str]:
        """Create the broad list1/list2/list1/... slot sequence."""

        longest = max(len(motif) for motif in self.list1 + self.list2)
        auto_slots = max(2, int(length // max(1.0, longest + self.slot_gap)))
        num_slots = self.max_slots if self.max_slots > 0 else auto_slots
        groups = []
        current = self.start_group
        for _ in range(num_slots):
            groups.append(current)
            current = "list2" if current == "list1" else "list1"
        return groups

    def slot_mask(
        self,
        length: int,
        slot_index: int,
        num_slots: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Return a broad Gaussian-like mask for one planned slot."""

        positions = self._positions(length, like)
        center = (slot_index + 0.5) * length / num_slots
        sigma = max(1.0, self.slot_width * length / num_slots)
        return torch.exp(-0.5 * ((positions - center) / sigma) ** 2)

    def _apply_single(
        self,
        logits: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, AlternatingLogitGuidanceResult]:
        if self.gradient_scale == 0:
            return logits, AlternatingLogitGuidanceResult()

        with torch.enable_grad():
            work = logits.detach().clone().float().requires_grad_(True)
            probs = torch.softmax(work / max(self.temperature, 1e-6), dim=-1)
            objective, details = self.compute_objective(probs)
            if float(objective.detach().cpu().item()) == 0.0:
                return logits, AlternatingLogitGuidanceResult()
            if not objective.requires_grad:
                return logits, AlternatingLogitGuidanceResult(
                    objective=float(details["objective"]),
                    transition=float(details["transition"]),
                    coverage=float(details["coverage"]),
                    tail=float(details["tail"]),
                    num_slots=int(details["num_slots"]),
                )
            objective.backward()
            grad = work.grad

        if grad is None:
            return logits, AlternatingLogitGuidanceResult()

        change_mask = can_change_idx.to(device=logits.device, dtype=torch.bool)
        while change_mask.ndim < grad.ndim:
            change_mask = change_mask.unsqueeze(-1)
        grad = torch.where(change_mask, grad, torch.zeros_like(grad))
        grad_norm = float(torch.norm(grad).detach().cpu().item())
        if grad_norm == 0.0:
            return logits, AlternatingLogitGuidanceResult()

        delta = grad / grad_norm * self.gradient_scale
        result = AlternatingLogitGuidanceResult(
            gradient_applied=True,
            gradient_norm=grad_norm,
            objective=float(details["objective"]),
            transition=float(details["transition"]),
            coverage=float(details["coverage"]),
            tail=float(details["tail"]),
            num_slots=int(details["num_slots"]),
        )
        return logits + delta.to(dtype=logits.dtype), result

    @staticmethod
    def _positions(length: int, like: torch.Tensor) -> torch.Tensor:
        return torch.arange(length, device=like.device, dtype=torch.float32)

    @staticmethod
    def _merge_batch_results(
        results: list[AlternatingLogitGuidanceResult],
    ) -> AlternatingLogitGuidanceResult:
        if not results:
            return AlternatingLogitGuidanceResult()
        return AlternatingLogitGuidanceResult(
            gradient_applied=any(result.gradient_applied for result in results),
            gradient_norm=max(result.gradient_norm for result in results),
            objective=max(result.objective for result in results),
            transition=max(result.transition for result in results),
            coverage=max(result.coverage for result in results),
            tail=max(result.tail for result in results),
            num_slots=max(result.num_slots for result in results),
            same_group_penalty=max(
                result.same_group_penalty for result in results
            ),
            start=max(result.start for result in results),
            end=max(result.end for result in results),
        )


class SoftTransitionAlternatingLogitGuidance(AlternatingLogitGuidance):
    """Slot-free ProteinGenerator transition potential adapted to DISCO logits.

    ProteinGenerator's ``soft_transition_alternating_motifs_bias`` does not plan
    broad list1/list2 slots.  Instead it scores all ordered pairs of soft motif
    start events: list1->list2 and list2->list1 are rewarded, list1->list1 and
    list2->list2 are penalized, and small start/end/tail stabilizers reduce
    common bad endings.  DISCO receives the same idea as a logit gradient before
    its existing path-planning sampler runs.
    """

    def __init__(
        self,
        motif_list1: str | Iterable[str] | None,
        motif_list2: str | Iterable[str] | None,
        gradient_scale: float = 1.0,
        temperature: float = 0.5,
        tail_weight: float = 0.25,
        coverage_weight: float = 0.35,
        transition_weight: float = 0.40,
        same_group_penalty_weight: float = 0.25,
        start_weight: float = 0.20,
        end_weight: float = 0.20,
        direction_temperature: float = 8.0,
        min_separation: int = 1,
        start_group: str = "list1",
    ) -> None:
        super().__init__(
            motif_list1=motif_list1,
            motif_list2=motif_list2,
            gradient_scale=gradient_scale,
            temperature=temperature,
            max_slots=0,
            slot_gap=3.0,
            slot_width=1.35,
            tail_weight=tail_weight,
            coverage_weight=coverage_weight,
            transition_weight=transition_weight,
            start_group=start_group,
        )
        self.same_group_penalty_weight = float(same_group_penalty_weight)
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.direction_temperature = float(direction_temperature)
        self.min_separation = int(min_separation)
        self._pair_weight_cache: dict[tuple[int, str], torch.Tensor] = {}

    def ordered_pair_score(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Score soft ordered transitions from one event distribution to another."""

        length = left.shape[0]
        weight = self._pair_weight(length, left)
        pair_mass = left[:, None] * right[None, :] * weight
        return torch.sum(pair_mass) / max(1, length)

    def _pair_weight(self, length: int, like: torch.Tensor) -> torch.Tensor:
        """Return cached ordered-pair weights for one length/device pair."""

        cache_key = (length, str(like.device))
        cached = self._pair_weight_cache.get(cache_key)
        if cached is None:
            starts = torch.arange(length, device=like.device)
            distance = starts[None, :] - starts[:, None]
            valid = distance >= self.min_separation
            decay = torch.exp(
                -distance.float().clamp(min=0)
                / max(self.direction_temperature, 1e-6)
            )
            cached = torch.where(valid, decay, torch.zeros_like(decay))
            self._pair_weight_cache[cache_key] = cached
        return cached.to(device=like.device, dtype=like.dtype)

    def expected_tail_after_list1(self, list1_scores: torch.Tensor) -> torch.Tensor:
        """Estimate the expected normalized C-terminal tail after a list1 event."""

        length = list1_scores.shape[0]
        starts = self._positions(length, list1_scores)
        probabilities = torch.softmax(
            list1_scores / max(self.temperature, 1e-6),
            dim=0,
        )
        tail_norm = (length - starts) / max(1, length)
        return torch.sum(probabilities * tail_norm)

    def expected_late_list2(self, list2_scores: torch.Tensor) -> torch.Tensor:
        """Reward list2 motif probability late in the sequence."""

        length = list2_scores.shape[0]
        starts = self._positions(length, list2_scores)
        probabilities = torch.softmax(
            list2_scores / max(self.temperature, 1e-6),
            dim=0,
        )
        late_norm = starts / max(1, length - 1)
        return torch.sum(probabilities * late_norm)

    def compute_objective(
        self,
        probs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the slot-free transition objective from amino-acid probabilities."""

        length = probs.shape[0]
        if length == 0:
            zero = probs.new_tensor(0.0)
            return zero, {
                "objective": 0.0,
                "transition": 0.0,
                "coverage": 0.0,
                "tail": 0.0,
                "same_group_penalty": 0.0,
                "start": 0.0,
                "end": 0.0,
                "num_slots": 0.0,
            }

        list1_scores = self.group_start_scores(probs, self.list1)
        list2_scores = self.group_start_scores(probs, self.list2)

        transition_objective = 0.5 * (
            self.ordered_pair_score(list1_scores, list2_scores)
            + self.ordered_pair_score(list2_scores, list1_scores)
        )
        same_group_penalty = 0.5 * (
            self.ordered_pair_score(list1_scores, list1_scores)
            + self.ordered_pair_score(list2_scores, list2_scores)
        )
        coverage_objective = 0.5 * (
            torch.max(list1_scores) + torch.max(list2_scores)
        )
        start_scores = list1_scores if self.start_group == "list1" else list2_scores
        start_objective = start_scores[0]
        tail_objective = 1.0 - self.expected_tail_after_list1(list1_scores)
        end_objective = self.expected_late_list2(list2_scores)

        objective = (
            self.transition_weight * transition_objective
            + self.coverage_weight * coverage_objective
            + self.tail_weight * tail_objective
            + self.start_weight * start_objective
            + self.end_weight * end_objective
            - self.same_group_penalty_weight * same_group_penalty
        )
        details = {
            "objective": float(objective.detach().cpu().item()),
            "transition": float(transition_objective.detach().cpu().item()),
            "coverage": float(coverage_objective.detach().cpu().item()),
            "tail": float(tail_objective.detach().cpu().item()),
            "same_group_penalty": float(
                same_group_penalty.detach().cpu().item()
            ),
            "start": float(start_objective.detach().cpu().item()),
            "end": float(end_objective.detach().cpu().item()),
            "num_slots": 0.0,
        }
        return objective, details

    def _apply_single(
        self,
        logits: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, AlternatingLogitGuidanceResult]:
        guided, result = super()._apply_single(logits, can_change_idx)
        if result.gradient_applied or result.objective:
            with torch.no_grad():
                probs = torch.softmax(
                    logits.float() / max(self.temperature, 1e-6),
                    dim=-1,
                )
                _, details = self.compute_objective(probs)
            result.same_group_penalty = float(details["same_group_penalty"])
            result.start = float(details["start"])
            result.end = float(details["end"])
        return guided, result
