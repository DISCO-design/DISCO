"""High-level guidance controller for DISCO co-generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from disco.data.constants import MASK_TOKEN_IDX
from disco.model.cogen_inference.guidance.alternating import (
    AlternatingLogitGuidance,
    SoftTransitionAlternatingLogitGuidance,
    combined_alternating_score,
    parse_motif_list,
    score_alternation,
)
from disco.model.cogen_inference.guidance.immunogen import (
    ImmunogenFixedSpec,
    MutableWindowMotifLogitGuidance,
    build_fixed_position_mask,
    score_fixed_fragments,
    spacing_metrics,
    spacing_score,
    spec_from_dict,
)
from disco.model.cogen_inference.guidance.motif import MotifLogitGuidance
from disco.model.cogen_inference.guidance.motif import AA_TO_IDX
from disco.model.cogen_inference.guidance.scoring import (
    HardMotifScorer,
    parse_motifs,
    tokens_to_sequence,
)


@dataclass
class BestGuidanceState:
    """Best observed sequence/structure state under the configured score."""

    score: float
    step: int
    sequence_tokens: torch.Tensor
    structure: torch.Tensor
    details: dict[str, Any]
    sample_index: int = 0


class GuidanceController:
    """Orchestrates optional logit guidance, hard scoring, and save-best tracking."""

    def __init__(self, config: DictConfig | dict[str, Any]) -> None:
        self.config = config
        motifs = config.get("motifs", None)
        motif_weights = config.get("motif_weights", None)

        scorer_cfg = config.get("scoring", {})
        self.hard_scorer = HardMotifScorer(
            motifs=motifs,
            motif_weights=motif_weights,
            motif_all_weight=scorer_cfg.get("motif_all_weight", 0.8),
            motif_many_weight=scorer_cfg.get("motif_many_weight", 0.2),
            motif_many_cap=scorer_cfg.get("motif_many_cap", 0),
        )

        logit_cfg = config.get("logit_guidance", {})
        self.logit_guidance = None
        if logit_cfg.get("enabled", False):
            self.logit_guidance = MotifLogitGuidance(
                motifs=motifs,
                motif_weights=motif_weights,
                gradient_enabled=logit_cfg.get("gradient_enabled", True),
                gradient_scale=logit_cfg.get("gradient_scale", 1.0),
                motif_weight_all=logit_cfg.get("motif_weight_all", 0.5),
                motif_weight_many=logit_cfg.get("motif_weight_many", 0.5),
                motif_top_k=logit_cfg.get("motif_top_k", 3),
                temperature=logit_cfg.get("motif_temperature", 0.5),
                local_rescue_enabled=logit_cfg.get("local_rescue_enabled", False),
                local_rescue_bonus=logit_cfg.get("local_rescue_bonus", 1.0),
                local_rescue_top_k=logit_cfg.get("local_rescue_top_k", 3),
                local_rescue_min_probability=logit_cfg.get(
                    "local_rescue_min_probability", 0.03
                ),
                local_rescue_max_motifs_per_step=logit_cfg.get(
                    "local_rescue_max_motifs_per_step", 4
                ),
                local_rescue_max_changes_per_step=logit_cfg.get(
                    "local_rescue_max_changes_per_step", 8
                ),
                local_rescue_only_missing=logit_cfg.get(
                    "local_rescue_only_missing", True
                ),
            )
        self.hard_commit_enabled = bool(logit_cfg.get("hard_commit_enabled", False))
        self.hard_commit_max_motifs_per_step = int(
            logit_cfg.get("hard_commit_max_motifs_per_step", 4)
        )
        self.hard_commit_max_changes_per_step = int(
            logit_cfg.get("hard_commit_max_changes_per_step", 16)
        )
        self.hard_commit_spread = bool(logit_cfg.get("hard_commit_spread", True))
        self.hard_commit_position_offset = int(
            logit_cfg.get("hard_commit_position_offset", 0)
        )
        self.hard_commit_all_motifs = bool(
            logit_cfg.get("hard_commit_all_motifs", False)
        )
        self.last_hard_commit_details: dict[str, object] = {}

        save_cfg = config.get("save_best", {})
        self.save_best_enabled = bool(save_cfg.get("enabled", False))
        self.select_best_output = bool(save_cfg.get("select_best_output", False))
        self.motif_score_weight = float(save_cfg.get("motif_score_weight", 1.0))
        self.confidence_weight = float(save_cfg.get("confidence_weight", 0.0))
        alternating_cfg = config.get("alternating", {})
        self.alternating_enabled = bool(alternating_cfg.get("enabled", False))
        self.alternating_list1 = parse_motif_list(
            alternating_cfg.get("motif_list1", None)
        )
        self.alternating_list2 = parse_motif_list(
            alternating_cfg.get("motif_list2", None)
        )
        self.alternating_expected_start = alternating_cfg.get(
            "expected_start_group", "list1"
        )
        self.alternating_combined_weight = float(
            alternating_cfg.get("combined_weight", 0.0)
        )
        self.alternating_weights = {
            "alternation_weight": alternating_cfg.get("alternation_weight", 0.45),
            "balance_weight": alternating_cfg.get("balance_weight", 0.20),
            "coverage_weight": alternating_cfg.get("coverage_weight", 0.20),
            "terminal_weight": alternating_cfg.get("terminal_weight", 0.15),
            "start_weight": alternating_cfg.get("start_weight", 0.15),
        }
        self.alternating_logit_guidance = None
        if self.alternating_enabled and alternating_cfg.get(
            "logit_guidance_enabled",
            False,
        ):
            self.alternating_logit_guidance = AlternatingLogitGuidance(
                motif_list1=self.alternating_list1,
                motif_list2=self.alternating_list2,
                gradient_scale=alternating_cfg.get("logit_gradient_scale", 1.0),
                temperature=alternating_cfg.get("logit_temperature", 0.5),
                max_slots=alternating_cfg.get("logit_max_slots", 0),
                slot_gap=alternating_cfg.get("logit_slot_gap", 3.0),
                slot_width=alternating_cfg.get("logit_slot_width", 1.35),
                tail_weight=alternating_cfg.get("logit_tail_weight", 0.25),
                coverage_weight=alternating_cfg.get("logit_coverage_weight", 0.35),
                transition_weight=alternating_cfg.get(
                    "logit_transition_weight",
                    0.40,
                ),
                start_group=self.alternating_expected_start,
        )
        self.alternating_transition_logit_guidance = None
        if self.alternating_enabled and alternating_cfg.get(
            "transition_logit_guidance_enabled",
            False,
        ):
            self.alternating_transition_logit_guidance = (
                SoftTransitionAlternatingLogitGuidance(
                    motif_list1=self.alternating_list1,
                    motif_list2=self.alternating_list2,
                    gradient_scale=alternating_cfg.get(
                        "transition_logit_gradient_scale",
                        alternating_cfg.get("logit_gradient_scale", 1.0),
                    ),
                    temperature=alternating_cfg.get(
                        "transition_logit_temperature",
                        alternating_cfg.get("logit_temperature", 0.5),
                    ),
                    tail_weight=alternating_cfg.get(
                        "transition_logit_tail_weight",
                        alternating_cfg.get("logit_tail_weight", 0.25),
                    ),
                    coverage_weight=alternating_cfg.get(
                        "transition_logit_coverage_weight",
                        alternating_cfg.get("logit_coverage_weight", 0.35),
                    ),
                    transition_weight=alternating_cfg.get(
                        "transition_logit_transition_weight",
                        alternating_cfg.get("logit_transition_weight", 0.40),
                    ),
                    same_group_penalty_weight=alternating_cfg.get(
                        "transition_same_group_penalty_weight",
                        0.25,
                    ),
                    start_weight=alternating_cfg.get(
                        "transition_soft_start_weight",
                        0.20,
                    ),
                    end_weight=alternating_cfg.get(
                        "transition_soft_end_weight",
                        0.20,
                    ),
                    direction_temperature=alternating_cfg.get(
                        "transition_distance_temperature",
                        8.0,
                    ),
                    min_separation=alternating_cfg.get(
                        "transition_min_separation",
                        1,
                    ),
                    start_group=self.alternating_expected_start,
                )
            )
        immunogen_cfg = config.get("immunogen", {})
        self.immunogen_enabled = bool(immunogen_cfg.get("enabled", False))
        self.immunogen_combined_weight = float(
            immunogen_cfg.get("combined_weight", 0.0)
        )
        self.immunogen_protect_fixed = bool(
            immunogen_cfg.get("protect_fixed_fragments", True)
        )
        self.immunogen_spacing_aware = bool(
            immunogen_cfg.get("spacing_aware", False)
        )
        self.immunogen_spacing_weight = float(
            immunogen_cfg.get("spacing_weight", 0.05)
        )
        self.immunogen_spacing_target_distance = float(
            immunogen_cfg.get("spacing_target_distance", 0.0)
        )
        self.immunogen_motifs = immunogen_cfg.get("motifs", motifs)
        self.immunogen_motif_weights = immunogen_cfg.get(
            "motif_weights",
            motif_weights,
        )
        self.immunogen_spec: ImmunogenFixedSpec | None = None
        if self.immunogen_enabled:
            self.immunogen_spec = spec_from_dict(
                {"fixed_fragments": immunogen_cfg.get("fixed_fragments", [])}
            )
        self.immunogen_mutable_guidance = None
        mutable_cfg = immunogen_cfg.get("mutable_logit_guidance", {})
        if (
            self.immunogen_enabled
            and self.immunogen_spec is not None
            and mutable_cfg.get("enabled", False)
        ):
            local_rescue_max_motifs = mutable_cfg.get(
                "local_rescue_max_motifs_per_step",
                4,
            )
            local_rescue_max_changes = mutable_cfg.get(
                "local_rescue_max_changes_per_step",
                8,
            )
            if mutable_cfg.get("coverage_guided", False):
                motif_count = len(parse_motifs(self.immunogen_motifs))
                local_rescue_max_motifs = max(1, motif_count)
                local_rescue_max_changes = max(2, 2 * motif_count)
            self.immunogen_mutable_guidance = MutableWindowMotifLogitGuidance(
                fixed_spec=self.immunogen_spec,
                motifs=self.immunogen_motifs,
                motif_weights=self.immunogen_motif_weights,
                gradient_enabled=mutable_cfg.get("gradient_enabled", True),
                gradient_scale=mutable_cfg.get("gradient_scale", 1.0),
                motif_weight_all=mutable_cfg.get("motif_weight_all", 0.5),
                motif_weight_many=mutable_cfg.get("motif_weight_many", 0.5),
                motif_top_k=mutable_cfg.get("motif_top_k", 3),
                temperature=mutable_cfg.get("motif_temperature", 0.5),
                local_rescue_enabled=mutable_cfg.get("local_rescue_enabled", False),
                local_rescue_bonus=mutable_cfg.get("local_rescue_bonus", 1.0),
                local_rescue_top_k=mutable_cfg.get("local_rescue_top_k", 3),
                local_rescue_min_probability=mutable_cfg.get(
                    "local_rescue_min_probability",
                    0.03,
                ),
                local_rescue_max_motifs_per_step=local_rescue_max_motifs,
                local_rescue_max_changes_per_step=local_rescue_max_changes,
                local_rescue_only_missing=mutable_cfg.get(
                    "local_rescue_only_missing",
                    True,
                ),
            )
        self.best_state: BestGuidanceState | None = None
        self.best_states_by_sample: dict[int, BestGuidanceState] = {}
        self.last_modify_details: dict[str, object] = {}
        self.last_change_mask_details: dict[str, object] = {}

    def modify_change_mask(
        self,
        can_change_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Block sampler updates at configured fixed immunogen fragments."""

        self.last_change_mask_details = {}
        if (
            not self.immunogen_enabled
            or self.immunogen_spec is None
            or not self.immunogen_protect_fixed
        ):
            return can_change_idx

        length = can_change_idx.shape[-1]
        fixed_mask = build_fixed_position_mask(length, self.immunogen_spec).to(
            device=can_change_idx.device,
            dtype=torch.bool,
        )
        protected_count = int(fixed_mask.sum().detach().cpu().item())
        if protected_count == 0:
            return can_change_idx

        updated = can_change_idx.clone()
        if updated.ndim == 1:
            updated[fixed_mask] = False
        else:
            updated[..., fixed_mask] = False

        self.last_change_mask_details = {
            "guidance_immunogen_fixed_positions_blocked": protected_count,
            "guidance_immunogen_fixed_positions_1based": ",".join(
                str(index + 1)
                for index in fixed_mask.nonzero(as_tuple=False).flatten().tolist()
            ),
        }
        return updated

    def modify_logits(
        self,
        logits: torch.Tensor,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
        step: int,
        x_denoised: torch.Tensor | None = None,
        x_l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply configured logit-level guidance before DISCO samples sequence."""

        del step, x_denoised, x_l
        guided = logits
        details: dict[str, object] = {}

        if self.logit_guidance is not None:
            guided = self.logit_guidance.apply(guided, xt_seq, can_change_idx)
            details.update(self.logit_guidance.last_result.as_dict())

        if self.alternating_logit_guidance is not None:
            guided = self.alternating_logit_guidance.apply(
                guided,
                xt_seq,
                can_change_idx,
            )
            details.update(self.alternating_logit_guidance.last_result.as_dict())

        if self.alternating_transition_logit_guidance is not None:
            guided = self.alternating_transition_logit_guidance.apply(
                guided,
                xt_seq,
                can_change_idx,
            )
            details.update(
                self.alternating_transition_logit_guidance.last_result.as_dict(
                    prefix="guidance_alternating_transition_logit"
                )
            )

        if self.immunogen_mutable_guidance is not None:
            guided = self.immunogen_mutable_guidance.apply(
                guided,
                xt_seq,
                can_change_idx,
            )
            details.update(self._immunogen_mutable_guidance_details())

        self.last_modify_details = details
        return guided

    def modify_sampled_sequence(
        self,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally commit missing motifs into the sampled hard sequence.

        This is intentionally separate from logit guidance: it edits the concrete
        sampled sequence after DISCO's sampler has chosen tokens.  It is disabled
        by default and meant for experiments where exact motif presence is the
        primary constraint.
        """

        self.last_hard_commit_details = {}
        if not self.hard_commit_enabled:
            return xt_seq

        if xt_seq.ndim == 1:
            updated, details = self._hard_commit_single(xt_seq, can_change_idx)
            self.last_hard_commit_details = details
            return updated

        outputs = []
        selected_total = 0
        changed_total = 0
        summaries: list[str] = []
        for sample_index in range(xt_seq.shape[0]):
            change_b = (
                can_change_idx[sample_index]
                if can_change_idx.ndim == 2
                else can_change_idx
            )
            updated_b, details_b = self._hard_commit_single(
                xt_seq[sample_index],
                change_b,
            )
            outputs.append(updated_b)
            selected_total += int(details_b.get("guidance_hard_commit_selected", 0))
            changed_total += int(details_b.get("guidance_hard_commit_changes", 0))
            summary = str(details_b.get("guidance_hard_commit_summary", ""))
            if summary:
                summaries.append(f"{sample_index}:{summary}")

        self.last_hard_commit_details = {
            "guidance_hard_commit_enabled": True,
            "guidance_hard_commit_selected": selected_total,
            "guidance_hard_commit_changes": changed_total,
            "guidance_hard_commit_summary": ";".join(summaries),
        }
        return torch.stack(outputs, dim=0)

    def _hard_commit_single(
        self,
        xt_seq: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        sequence = tokens_to_sequence(xt_seq)
        score = self.hard_scorer.score_sequence(sequence)
        if self.hard_commit_all_motifs:
            motifs_to_commit = [
                (motif_index, motif)
                for motif_index, motif in enumerate(self.hard_scorer.motifs)
            ]
        else:
            motifs_to_commit = [
                (self.hard_scorer.motifs.index(motif), motif)
                for motif in score.missing
            ]

        if not motifs_to_commit:
            return xt_seq, {
                "guidance_hard_commit_enabled": True,
                "guidance_hard_commit_selected": 0,
                "guidance_hard_commit_changes": 0,
                "guidance_hard_commit_summary": "",
            }

        can_change = can_change_idx.detach().cpu().bool().reshape(-1).tolist()
        current = xt_seq.detach().cpu().reshape(-1).tolist()
        updated = xt_seq.clone()
        claimed: dict[int, int] = {}
        selected: list[str] = []
        changes = 0

        total_motifs = len(self.hard_scorer.motifs)
        for motif_index, motif in motifs_to_commit:
            if len(selected) >= self.hard_commit_max_motifs_per_step:
                break
            motif_indices = [AA_TO_IDX[aa] for aa in motif]
            if self.hard_commit_spread:
                target_start = round(
                    ((motif_index + 1) * len(current) / (total_motifs + 1))
                    - (len(motif_indices) / 2)
                )
                target_start += self.hard_commit_position_offset
                target_start = max(0, min(target_start, len(current) - len(motif_indices)))
            else:
                target_start = 0
            candidates: list[tuple[int, int, int, int]] = []
            for start in range(0, len(current) - len(motif_indices) + 1):
                possible = True
                change_count = 0
                masked_count = 0
                for offset, aa_idx in enumerate(motif_indices):
                    position = start + offset
                    if not can_change[position]:
                        possible = False
                        break
                    if position in claimed and claimed[position] != aa_idx:
                        possible = False
                        break
                    if current[position] == MASK_TOKEN_IDX:
                        masked_count += 1
                    if current[position] != aa_idx:
                        change_count += 1
                if possible:
                    target_distance = abs(start - target_start)
                    candidates.append(
                        (target_distance, change_count, -masked_count, start)
                    )
            candidates.sort()
            for _target_distance, change_count, _masked_bias, start in candidates:
                if changes + change_count > self.hard_commit_max_changes_per_step:
                    continue
                for offset, aa_idx in enumerate(motif_indices):
                    position = start + offset
                    updated[position] = aa_idx
                    claimed[position] = aa_idx
                changes += change_count
                selected.append(f"{motif}@{start + 1}")
                current = updated.detach().cpu().reshape(-1).tolist()
                break

        return updated, {
            "guidance_hard_commit_enabled": True,
            "guidance_hard_commit_selected": len(selected),
            "guidance_hard_commit_changes": changes,
            "guidance_hard_commit_summary": ";".join(selected),
        }

    def observe_step(
        self,
        xt_seq: torch.Tensor,
        structure: torch.Tensor,
        logits: torch.Tensor,
        can_change_idx: torch.Tensor,
        step: int,
    ) -> dict[str, object]:
        """Score the current step and optionally update best-state bookkeeping."""

        batch_size = self._batch_size(xt_seq)
        sample_records: list[dict[str, object]] = []
        primary_details: dict[str, object] | None = None
        primary_best_updated = False
        global_best_updated = False

        for sample_index in range(batch_size):
            score_seq = self._sample_tensor(xt_seq, sample_index, batch_size)
            score_logits = self._sample_tensor(logits, sample_index, batch_size)
            score_change_idx = self._sample_tensor(
                can_change_idx,
                sample_index,
                batch_size,
            )
            score_structure = self._sample_tensor(
                structure,
                sample_index,
                batch_size,
            )
            scored = self._score_sample(
                score_seq=score_seq,
                score_logits=score_logits,
                score_change_idx=score_change_idx,
            )
            details = scored["details"]
            motif_score = scored["motif_score"]
            combined = float(scored["combined"])

            sample_best_updated = False
            if self.save_best_enabled:
                state = BestGuidanceState(
                    score=combined,
                    step=int(step),
                    sequence_tokens=score_seq.detach().cpu().clone(),
                    structure=score_structure.detach().cpu().clone(),
                    details={
                        **details,
                        "motif_counts": motif_score.counts,
                        "motif_positions": motif_score.positions,
                    },
                    sample_index=sample_index,
                )
                previous_sample_best = self.best_states_by_sample.get(sample_index)
                if self._is_better_best_state(state, previous_sample_best):
                    self.best_states_by_sample[sample_index] = state
                    sample_best_updated = True

                if self._is_better_best_state(state, self.best_state):
                    self.best_state = state
                    global_best_updated = True

            best_for_sample = self.best_states_by_sample.get(sample_index)
            if sample_index == 0:
                primary_details = dict(details)
                primary_best_updated = sample_best_updated
                if best_for_sample is not None:
                    primary_details["guidance_best_updated"] = sample_best_updated
                    primary_details["guidance_best_step"] = best_for_sample.step
                    primary_details["guidance_best_score"] = best_for_sample.score

            sample_records.append(
                {
                    "sample_index": sample_index,
                    "sequence": details["guidance_sequence"],
                    "combined_score": combined,
                    "motif_score": details["guidance_motif_score"],
                    "sequence_confidence": details["guidance_sequence_confidence"],
                    "num_unique": details["guidance_num_unique"],
                    "num_total": details["guidance_num_total"],
                    "present": details["guidance_present"],
                    "missing": details["guidance_missing"],
                    "best_updated": sample_best_updated,
                    "best_step": ""
                    if best_for_sample is None
                    else best_for_sample.step,
                    "best_score": ""
                    if best_for_sample is None
                    else best_for_sample.score,
                }
            )

        details = primary_details or {}
        details.update(
            {
                "guidance_sample_index": 0,
                "guidance_batch_size": batch_size,
                "guidance_batch_sample_scores": sample_records,
                "guidance_global_best_updated": global_best_updated,
            }
        )
        if self.best_state is not None:
            details.update(
                {
                    "guidance_global_best_sample_index": self.best_state.sample_index,
                    "guidance_global_best_step": self.best_state.step,
                    "guidance_global_best_score": self.best_state.score,
                    "guidance_global_best_sequence": self.best_state.details.get(
                        "guidance_sequence",
                        "",
                    ),
                }
            )
            details.setdefault("guidance_best_updated", primary_best_updated)
            details.setdefault(
                "guidance_best_step",
                self.best_states_by_sample.get(0, self.best_state).step,
            )
            details.setdefault(
                "guidance_best_score",
                self.best_states_by_sample.get(0, self.best_state).score,
            )

        return details

    @staticmethod
    def _state_mask_count(state: BestGuidanceState) -> int:
        """Return the number of unresolved mask characters in a saved state."""

        sequence = str(state.details.get("guidance_sequence", ""))
        return sequence.count("-")

    @classmethod
    def _is_better_best_state(
        cls,
        candidate: BestGuidanceState,
        current: BestGuidanceState | None,
    ) -> bool:
        """Compare best-state candidates with deterministic tie-breaking.

        Motif-focused runs can produce high-scoring intermediate states before
        every position has been decoded.  Those are useful diagnostics, but they
        are not valid final proteins.  Prefer the more decoded sequence before
        comparing scores; within the same decoded/masked status, compare score
        and then prefer the later step for deterministic tie-breaking.
        """

        if current is None:
            return True
        candidate_masks = cls._state_mask_count(candidate)
        current_masks = cls._state_mask_count(current)
        if candidate_masks < current_masks:
            return True
        if candidate_masks > current_masks:
            return False
        if candidate.score > current.score:
            return True
        if candidate.score < current.score:
            return False
        return candidate.step > current.step

    def _score_sample(
        self,
        *,
        score_seq: torch.Tensor,
        score_logits: torch.Tensor,
        score_change_idx: torch.Tensor,
    ) -> dict[str, Any]:
        """Score one concrete sample from a possibly batched DISCO step."""

        sequence = tokens_to_sequence(score_seq)
        motif_score = self.hard_scorer.score_sequence(sequence)
        confidence = self._sequence_confidence(
            score_logits,
            score_seq,
            score_change_idx,
        )
        combined = (
            self.motif_score_weight * motif_score.motif_score
            + self.confidence_weight * confidence
        )
        alternating_details = self._alternating_details(sequence)
        if alternating_details is not None:
            combined += self.alternating_combined_weight * float(
                alternating_details["guidance_alternating_combined_score"]
            )
        immunogen_details = self._immunogen_details(sequence)
        if immunogen_details is not None:
            combined += self.immunogen_combined_weight * float(
                immunogen_details["guidance_immunogen_fixed_score"]
            )
            combined += self.immunogen_spacing_weight * float(
                immunogen_details.get("guidance_immunogen_spacing_score", 0.0)
            )

        details: dict[str, object] = {
            "guidance_sequence": sequence,
            "guidance_motif_score": motif_score.motif_score,
            "guidance_all_score": motif_score.all_score,
            "guidance_many_score": motif_score.many_score,
            "guidance_num_unique": motif_score.num_unique,
            "guidance_num_total": motif_score.num_total,
            "guidance_missing": ",".join(motif_score.missing),
            "guidance_present": ",".join(motif_score.present),
            "guidance_sequence_confidence": confidence,
            "guidance_combined_score": float(combined),
            **self.last_change_mask_details,
            **self.last_modify_details,
            **self.last_hard_commit_details,
        }
        if alternating_details is not None:
            details.update(alternating_details)
        if immunogen_details is not None:
            details.update(immunogen_details)

        return {
            "combined": float(combined),
            "details": details,
            "motif_score": motif_score,
        }

    def _immunogen_details(self, sequence: str) -> dict[str, object] | None:
        if not self.immunogen_enabled or self.immunogen_spec is None:
            return None
        score = score_fixed_fragments(sequence, self.immunogen_spec)
        fixed_score = 1.0 if score["fixed_fragments_preserved"] else 0.0
        details = {
            "guidance_immunogen_enabled": True,
            "guidance_immunogen_fixed_score": fixed_score,
            "guidance_immunogen_fixed_preserved": score["fixed_fragments_preserved"],
            "guidance_immunogen_fixed_mismatches": score["num_fixed_mismatches"],
            "guidance_immunogen_mismatch_summary": ";".join(
                f"{m['label']}@{m['start_1based']}:{m['expected']}!={m['observed']}"
                for m in score["mismatches"]
            ),
        }
        if self.immunogen_spacing_aware:
            motifs = parse_motifs(self.immunogen_motifs)
            target = (
                self.immunogen_spacing_target_distance
                if self.immunogen_spacing_target_distance > 0
                else None
            )
            spacing = spacing_score(sequence, motifs, target_distance=target)
            metrics = spacing_metrics(sequence, motifs)
            details.update(
                {
                    "guidance_immunogen_spacing_score": spacing,
                    "guidance_immunogen_spacing_positions": metrics[
                        "motif_spacing_positions_1based"
                    ],
                    "guidance_immunogen_spacing_min_distance": metrics[
                        "motif_min_start_distance"
                    ],
                    "guidance_immunogen_spacing_mean_distance": metrics[
                        "motif_mean_pairwise_distance"
                    ],
                    "guidance_immunogen_spacing_span": metrics["motif_start_span"],
                    "guidance_immunogen_spacing_spread_score": metrics[
                        "motif_spread_score"
                    ],
                }
            )
        return details

    def _immunogen_mutable_guidance_details(self) -> dict[str, object]:
        if self.immunogen_mutable_guidance is None:
            return {}
        result = self.immunogen_mutable_guidance.last_result
        return {
            "guidance_immunogen_mutable_gradient_applied": result.gradient_applied,
            "guidance_immunogen_mutable_local_rescue_applied": (
                result.local_rescue_applied
            ),
            "guidance_immunogen_mutable_gradient_norm": result.gradient_norm,
            "guidance_immunogen_mutable_local_rescue_candidates": (
                result.local_rescue_candidates
            ),
            "guidance_immunogen_mutable_local_rescue_selected": (
                result.local_rescue_selected
            ),
            "guidance_immunogen_mutable_local_rescue_summary": ";".join(
                result.selected_summary
            ),
        }

    def _alternating_details(self, sequence: str) -> dict[str, object] | None:
        if not self.alternating_enabled:
            return None
        if not self.alternating_list1 or not self.alternating_list2:
            return {
                "guidance_alternating_enabled": True,
                "guidance_alternating_error": "motif_list1 and motif_list2 are required",
                "guidance_alternating_combined_score": 0.0,
            }

        metrics = score_alternation(
            sequence,
            self.alternating_list1,
            self.alternating_list2,
            expected_start_group=self.alternating_expected_start,
        )
        score = combined_alternating_score(
            sequence,
            self.alternating_list1,
            self.alternating_list2,
            expected_start_group=self.alternating_expected_start,
            **self.alternating_weights,
        )
        return {
            "guidance_alternating_enabled": True,
            "guidance_alternating_combined_score": score,
            "guidance_alternating_pattern": metrics["motif_pattern"],
            "guidance_alternating_hits": metrics["num_motif_hits"],
            "guidance_alternating_unique": metrics["num_unique_motifs"],
            "guidance_alternating_score": metrics["alternation_score"],
            "guidance_alternating_balance": metrics["motif_balance_score"],
            "guidance_alternating_terminal": metrics["terminal_score"],
            "guidance_alternating_start": metrics["start_state_score"],
            "guidance_alternating_summary": metrics["motif_hit_summary"],
        }

    def get_best_state(self) -> BestGuidanceState | None:
        return self.best_state

    def get_best_states_by_sample(self) -> dict[int, BestGuidanceState]:
        """Return the best observed state for each batch sample."""

        return dict(self.best_states_by_sample)

    def select_best_outputs(
        self,
        xt_seq: torch.Tensor,
        structure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]] | None:
        """Build output tensors with each sample replaced by its own best state."""

        if not self.select_best_output or not self.best_states_by_sample:
            return None

        batch_size = self._batch_size(xt_seq)
        selected_seq = xt_seq.clone()
        selected_structure = structure.clone()
        selected_samples: list[dict[str, object]] = []

        for sample_index, state in sorted(self.best_states_by_sample.items()):
            if sample_index >= batch_size:
                continue
            self._assign_sample_tensor(
                selected_seq,
                state.sequence_tokens.to(device=xt_seq.device, dtype=xt_seq.dtype),
                sample_index,
                batch_size,
            )
            self._assign_sample_tensor(
                selected_structure,
                state.structure.to(device=structure.device, dtype=structure.dtype),
                sample_index,
                batch_size,
            )
            selected_samples.append(
                {
                    "sample_index": sample_index,
                    "best_step": state.step,
                    "best_score": state.score,
                    "best_sequence": state.details.get("guidance_sequence", ""),
                }
            )

        if not selected_samples:
            return None

        global_state = self.best_state
        details: dict[str, object] = {
            "guidance_best_output_selected": True,
            "guidance_best_output_batch_size": batch_size,
            "guidance_best_output_num_samples": len(selected_samples),
            "guidance_best_output_samples": selected_samples,
        }
        if global_state is not None:
            details.update(
                {
                    "guidance_best_sample_index": global_state.sample_index,
                    "guidance_best_step": global_state.step,
                    "guidance_best_score": global_state.score,
                    "guidance_best_details": global_state.details,
                }
            )
        return selected_seq, selected_structure, details

    @staticmethod
    def _batch_size(xt_seq: torch.Tensor) -> int:
        return int(xt_seq.shape[0]) if xt_seq.ndim >= 2 else 1

    @staticmethod
    def _sample_tensor(
        tensor: torch.Tensor,
        sample_index: int,
        batch_size: int,
    ) -> torch.Tensor:
        if tensor.ndim > 0 and tensor.shape[0] == batch_size:
            return tensor[sample_index]
        return tensor

    @staticmethod
    def _assign_sample_tensor(
        target: torch.Tensor,
        value: torch.Tensor,
        sample_index: int,
        batch_size: int,
    ) -> None:
        if target.ndim > 0 and target.shape[0] == batch_size:
            if target[sample_index].shape == value.shape:
                target[sample_index] = value
            else:
                target[sample_index] = value.reshape_as(target[sample_index])
        elif target.shape == value.shape:
            target.copy_(value)

    @staticmethod
    def _sequence_confidence(
        logits: torch.Tensor,
        sequence_tokens: torch.Tensor,
        can_change_idx: torch.Tensor,
    ) -> float:
        if logits.numel() == 0:
            return 0.0
        valid = (sequence_tokens != MASK_TOKEN_IDX) & can_change_idx.to(
            device=sequence_tokens.device,
            dtype=torch.bool,
        )
        if not bool(valid.any().item()):
            return 0.0
        probs = torch.softmax(logits, dim=-1)
        token_idx = sequence_tokens.to(device=logits.device).long().clamp(min=0)
        selected = probs.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)
        return float(selected[valid.to(device=selected.device)].mean().detach().cpu().item())


def build_guidance_controller(
    config: DictConfig | dict[str, Any] | None,
) -> GuidanceController | None:
    """Create a controller only when guidance is explicitly enabled."""

    if config is None or not config.get("enabled", False):
        return None
    return GuidanceController(config)
