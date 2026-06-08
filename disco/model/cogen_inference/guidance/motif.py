"""Motif logit guidance for DISCO sequence co-generation.

The classes here operate only on sequence decoder logits.  They do not alter
DISCO's Pairformer, diffusion module, or path-planning sampler.  This keeps the
first port close to the ProteinGenerator motif approaches while respecting
DISCO's existing discrete sampling machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from disco.data.constants import PRO_STD_RESIDUES
from disco.model.cogen_inference.guidance.scoring import (
    HardMotifScorer,
    parse_motifs,
    parse_weights,
    tokens_to_sequence,
)


def three_to_one(resname: str) -> str:
    mapping = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    return mapping[resname]


AA_TO_IDX = {
    one_letter: idx
    for three_letter, idx in PRO_STD_RESIDUES.items()
    if three_letter != "UNK"
    for one_letter in [three_to_one(three_letter)]
}


@dataclass
class MotifGuidanceResult:
    """Small diagnostic summary for the last logit guidance call."""

    gradient_applied: bool = False
    local_rescue_applied: bool = False
    gradient_norm: float = 0.0
    local_rescue_candidates: int = 0
    local_rescue_selected: int = 0
    selected_summary: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "guidance_gradient_applied": self.gradient_applied,
            "guidance_local_rescue_applied": self.local_rescue_applied,
            "guidance_gradient_norm": self.gradient_norm,
            "guidance_local_rescue_candidates": self.local_rescue_candidates,
            "guidance_local_rescue_selected": self.local_rescue_selected,
            "guidance_local_rescue_summary": ";".join(self.selected_summary),
        }


class MotifLogitGuidance:
    """Apply PG-style motif pressure to DISCO sequence decoder logits."""

    def __init__(
        self,
        motifs: str | Iterable[str] | None,
        motif_weights: str | Iterable[float] | None = None,
        gradient_enabled: bool = True,
        gradient_scale: float = 1.0,
        motif_weight_all: float = 0.5,
        motif_weight_many: float = 0.5,
        motif_top_k: int = 3,
        temperature: float = 0.5,
        local_rescue_enabled: bool = False,
        local_rescue_bonus: float = 1.0,
        local_rescue_top_k: int = 3,
        local_rescue_min_probability: float = 0.03,
        local_rescue_max_motifs_per_step: int = 4,
        local_rescue_max_changes_per_step: int = 8,
        local_rescue_only_missing: bool = True,
    ) -> None:
        self.motifs = parse_motifs(motifs)
        self.motif_weights = parse_weights(motif_weights, len(self.motifs))
        self.gradient_enabled = bool(gradient_enabled)
        self.gradient_scale = float(gradient_scale)
        self.motif_weight_all = float(motif_weight_all)
        self.motif_weight_many = float(motif_weight_many)
        self.motif_top_k = int(motif_top_k)
        self.temperature = float(temperature)
        self.local_rescue_enabled = bool(local_rescue_enabled)
        self.local_rescue_bonus = float(local_rescue_bonus)
        self.local_rescue_top_k = int(local_rescue_top_k)
        self.local_rescue_min_probability = float(local_rescue_min_probability)
        self.local_rescue_max_motifs_per_step = int(local_rescue_max_motifs_per_step)
        self.local_rescue_max_changes_per_step = int(local_rescue_max_changes_per_step)
        self.local_rescue_only_missing = bool(local_rescue_only_missing)
        self.weight_sum = sum(self.motif_weights)
        self.hard_scorer = HardMotifScorer(motifs=self.motifs, motif_weights=self.motif_weights)
        self.last_result = MotifGuidanceResult()

        for motif in self.motifs:
            for aa in motif:
                if aa not in AA_TO_IDX:
                    raise ValueError(f"Unknown amino acid in motif {motif!r}: {aa!r}")

    def apply(
        self,
        logits: torch.Tensor,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits modified by the configured motif guidance methods."""

        result = MotifGuidanceResult()
        guided = logits
        if guided.ndim == 2:
            guided, result = self._apply_single(guided, xt_seq, can_change_idx)
        elif guided.ndim == 3:
            outputs = []
            batch_results = []
            for b in range(guided.shape[0]):
                seq_b = xt_seq[b] if xt_seq.ndim == 2 else xt_seq
                change_b = can_change_idx[b] if can_change_idx.ndim == 2 else can_change_idx
                out_b, res_b = self._apply_single(guided[b], seq_b, change_b)
                outputs.append(out_b)
                batch_results.append(res_b)
            guided = torch.stack(outputs, dim=0)
            result = self._merge_batch_results(batch_results)
        else:
            raise ValueError(f"Expected logits with 2 or 3 dims, got shape {tuple(logits.shape)}")

        self.last_result = result
        return guided

    def _apply_single(
        self,
        logits: torch.Tensor,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, MotifGuidanceResult]:
        result = MotifGuidanceResult()
        guided = logits

        if self.gradient_enabled and self.gradient_scale != 0:
            guided, grad_norm = self._apply_gradient_guidance(guided, can_change_idx)
            result.gradient_applied = grad_norm > 0
            result.gradient_norm = grad_norm

        if self.local_rescue_enabled and self.local_rescue_bonus != 0:
            guided, candidates, selected, summary = self._apply_local_rescue(
                guided,
                xt_seq,
                can_change_idx,
            )
            result.local_rescue_applied = selected > 0
            result.local_rescue_candidates = candidates
            result.local_rescue_selected = selected
            result.selected_summary = summary

        return guided, result

    def _motif_window_scores(self, probs: torch.Tensor, motif: str) -> torch.Tensor:
        motif_len = len(motif)
        if motif_len > probs.shape[0]:
            return probs.new_zeros((0,))

        motif_filter = probs.new_zeros((1, probs.shape[-1], motif_len))
        for offset, aa in enumerate(motif):
            motif_filter[0, AA_TO_IDX[aa], offset] = 1.0

        x = probs.unsqueeze(0).permute(0, 2, 1)
        return F.conv1d(x, motif_filter).squeeze(0).squeeze(0) / motif_len

    def _motif_objective(self, probs: torch.Tensor) -> torch.Tensor:
        all_scores = []
        many_scores = []
        for motif, weight in zip(self.motifs, self.motif_weights, strict=True):
            scores = self._motif_window_scores(probs, motif)
            if scores.numel() == 0:
                continue
            best_score = scores.max()
            k = min(self.motif_top_k, scores.shape[-1])
            topk_scores = torch.topk(scores, k=k).values
            all_scores.append(weight * best_score)
            many_scores.append(weight * topk_scores.mean())

        if not all_scores:
            return probs.new_tensor(0.0)

        all_score = torch.stack(all_scores).sum() / self.weight_sum
        many_score = torch.stack(many_scores).sum() / self.weight_sum
        return self.motif_weight_all * all_score + self.motif_weight_many * many_score

    def _apply_gradient_guidance(
        self,
        logits: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        with torch.enable_grad():
            work = logits.detach().clone().float().requires_grad_(True)
            probs = torch.softmax(work / self.temperature, dim=-1)
            objective = self._motif_objective(probs)
            if objective.item() == 0.0:
                return logits, 0.0
            objective.backward()
            grad = work.grad

        if grad is None:
            return logits, 0.0

        change_mask = can_change_idx.to(device=logits.device, dtype=torch.bool)
        while change_mask.ndim < grad.ndim:
            change_mask = change_mask.unsqueeze(-1)
        grad = torch.where(change_mask, grad, torch.zeros_like(grad))
        grad_norm = float(torch.norm(grad).detach().cpu().item())
        if grad_norm == 0.0:
            return logits, 0.0

        delta = grad / grad_norm * self.gradient_scale
        return logits + delta.to(dtype=logits.dtype), grad_norm

    def _apply_local_rescue(
        self,
        logits: torch.Tensor,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, list[str]]:
        probs = torch.softmax(logits / self.temperature, dim=-1).detach()
        hard_tokens = torch.argmax(logits, dim=-1)
        hard_sequence = tokens_to_sequence(hard_tokens)
        hard_details = self.hard_scorer.score_sequence(hard_sequence)
        candidates = self._find_local_rescue_candidates(
            probs,
            hard_tokens,
            hard_details.missing,
            can_change_idx,
        )
        selected = self._select_local_rescue_candidates(candidates, hard_tokens)
        if not selected:
            return logits, len(candidates), 0, []

        guided = logits.clone()
        selected_summary = []
        for candidate in selected:
            selected_summary.append(f"{candidate['motif']}@{candidate['start'] + 1}")
            for position, aa_idx in candidate["assignments"]:
                guided[position, aa_idx] += self.local_rescue_bonus * candidate["weight"]
        return guided, len(candidates), len(selected), selected_summary

    def _find_local_rescue_candidates(
        self,
        probs: torch.Tensor,
        hard_tokens: torch.Tensor,
        missing_motifs: list[str],
        can_change_idx: torch.Tensor,
    ) -> list[dict[str, object]]:
        k = min(self.local_rescue_top_k, probs.shape[-1])
        topk_idx = torch.topk(probs, k=k, dim=-1).indices.detach().cpu().tolist()
        can_change = can_change_idx.detach().cpu().bool().tolist()
        hard_idx = hard_tokens.detach().cpu().tolist()
        candidates: list[dict[str, object]] = []
        length = probs.shape[0]

        for motif, weight in zip(self.motifs, self.motif_weights, strict=True):
            if self.local_rescue_only_missing and motif not in missing_motifs:
                continue
            motif_indices = [AA_TO_IDX[aa] for aa in motif]
            for start in range(length - len(motif) + 1):
                required_probs = []
                assignments = []
                argmax_changes = 0
                possible = True
                for offset, aa_idx in enumerate(motif_indices):
                    position = start + offset
                    if not can_change[position]:
                        possible = False
                        break
                    if aa_idx not in topk_idx[position]:
                        possible = False
                        break
                    probability = float(probs[position, aa_idx].detach().cpu().item())
                    if probability < self.local_rescue_min_probability:
                        possible = False
                        break
                    required_probs.append(probability)
                    assignments.append((position, aa_idx))
                    if hard_idx[position] != aa_idx:
                        argmax_changes += 1
                if not possible:
                    continue
                geom_mean = float(np.exp(np.mean(np.log(np.array(required_probs) + 1e-8))))
                score = weight * geom_mean + (1.0 if motif in missing_motifs else 0.0)
                score -= 0.05 * argmax_changes
                candidates.append(
                    {
                        "motif": motif,
                        "start": start,
                        "score": score,
                        "assignments": assignments,
                        "argmax_changes": argmax_changes,
                        "weight": weight,
                    }
                )

        return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)

    def _select_local_rescue_candidates(
        self,
        candidates: list[dict[str, object]],
        hard_tokens: torch.Tensor,
    ) -> list[dict[str, object]]:
        hard_idx = hard_tokens.detach().cpu().tolist()
        selected = []
        claimed: dict[int, int] = {}
        changes = 0

        for candidate in candidates:
            if len(selected) >= self.local_rescue_max_motifs_per_step:
                break
            new_changes = 0
            conflict = False
            for position, aa_idx in candidate["assignments"]:
                if position in claimed and claimed[position] != aa_idx:
                    conflict = True
                    break
                if position not in claimed and hard_idx[position] != aa_idx:
                    new_changes += 1
            if conflict:
                continue
            if changes + new_changes > self.local_rescue_max_changes_per_step:
                continue
            selected.append(candidate)
            changes += new_changes
            for position, aa_idx in candidate["assignments"]:
                claimed[position] = aa_idx

        return selected

    @staticmethod
    def _merge_batch_results(results: list[MotifGuidanceResult]) -> MotifGuidanceResult:
        merged = MotifGuidanceResult()
        merged.gradient_applied = any(result.gradient_applied for result in results)
        merged.local_rescue_applied = any(result.local_rescue_applied for result in results)
        merged.gradient_norm = max((result.gradient_norm for result in results), default=0.0)
        merged.local_rescue_candidates = sum(result.local_rescue_candidates for result in results)
        merged.local_rescue_selected = sum(result.local_rescue_selected for result in results)
        merged.selected_summary = [
            item for result in results for item in result.selected_summary
        ]
        return merged
