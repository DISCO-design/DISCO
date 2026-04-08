# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from disco.data.constants import MASK_TOKEN_IDX, PRO_RES_IDX_TO_RESNAME_ONE
from disco.model.cogen_inference.sequence_inference_noise_scheduler import (
    SequenceInferenceNoiseScheduler,
)
from disco.model.modules.plm import esm_tokens_to_sequence, MODEL_REGISTRY

logger = logging.getLogger(__name__)

INF = 1e7
LOG_ZERO = -1e5
EPSILON = 1e-7
ONE_TENSOR = torch.tensor(1.0)


def stochastic_sample_from_categorical(logits, temperature=1.0, noise_scale=1.0):
    """Samples from a categorical distribution with temperature scaling.

    Applies Gumbel noise and temperature scaling to logits, then returns
    the argmax token and its log-softmax score.

    Args:
        logits: Unnormalized log-probabilities of shape (..., num_classes).
        temperature: Temperature for scaling logits. 0.0 gives greedy decoding.
        noise_scale: Scale factor for the Gumbel noise.

    Returns:
        Tuple of (tokens, scores) where tokens are sampled indices and scores
        are the corresponding log-softmax values.
    """
    logits = logits.double()
    if temperature != 0.0:
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = logits / temperature + noise_scale * gumbel

    scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def topk_lowest_masking(scores, cutoff_len):
    """Creates a boolean mask for the top-k lowest scoring positions.

    Positions with scores below the cutoff_len-th lowest value are marked True.

    Args:
        scores: Score tensor of shape (..., seq_len).
        cutoff_len: Number of lowest-scoring positions to select.

    Returns:
        Boolean tensor of the same shape as scores, True for selected positions.
    """
    if cutoff_len == scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)

    sorted_scores, _ = scores.sort(dim=-1)

    if scores.ndim == 2:
        cutoff_len = cutoff_len.unsqueeze(0).repeat(len(scores)).unsqueeze(1)

    threshold = sorted_scores.gather(dim=-1, index=cutoff_len)
    return scores < threshold


def get_seq_t(seq_noise_scheduler, t_or_iter_num: float | int) -> float:
    btwn_zero_one = lambda x: 0.0 <= x and x <= 1.0

    type_cond = isinstance(t_or_iter_num, float) or (
        torch.is_tensor(t_or_iter_num) and torch.is_floating_point(t_or_iter_num)
    )

    if type_cond and btwn_zero_one(t_or_iter_num):
        return t_or_iter_num
    else:
        return seq_noise_scheduler.get_t(t_or_iter_num)


class BaseSequenceSamplingStrategy(ABC):
    def __init__(
        self,
        sequence_noise_scheduler: SequenceInferenceNoiseScheduler,
    ):
        self.sequence_noise_scheduler = sequence_noise_scheduler
        self.should_ensure_unmasked_stay = True

    def start(self, num_structure_inference_steps: int) -> None:
        self.sequence_noise_scheduler.start(num_structure_inference_steps)

    def reset(self, num_structure_inference_steps: int) -> None:
        self.sequence_noise_scheduler.start(num_structure_inference_steps)

    @abstractmethod
    def step(
        self,
        decoder_logits: torch.Tensor,
        xt_struct: torch.Tensor,
        xt_seq: torch.Tensor,
        struct_inf_iter_num: int,
        last_struct_inf_iter_num: int,
        can_change_idx: torch.Tensor,
        x_denoised: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Takes in the current xt structure and sequence along with the computed
        decoder logits and step to compute the new sequence. Note that this method
        does not have to actually change xt_seq at all and can return the same
        xt_seq as was given as input.

        Args:
            decoder_logits (torch.Tensor): Sequence decoder logits, computed (in some way)
                                           by the model
            xt_struct (torch.Tensor): The all-atom structure at time t
            xt_seq (torch.Tensor): The partially masked sequence at time t
            iter_num (int): The iteration number for structure inference
            x_denoised (torch.Tensor, optional): The model's predicted clean structure (x0).
                                                  Better geometry than xt_struct for scoring.

        Returns:
            (torch.Tensor): The new sequence from taking an inference step given
                            xt_seq and the decoder logits
        """
        pass


class VanillaMDLMSamplingStrategy(BaseSequenceSamplingStrategy):
    """Vanilla masked discrete language model (MDLM) sampling strategy.

    Implements MDLM inference with a log-linear noise schedule. At each step,
    computes transition probabilities from the current noise level to the next
    and samples new tokens for masked positions.
    """

    def __init__(
        self,
        sequence_noise_scheduler: SequenceInferenceNoiseScheduler,
        logits_temp: float = 1.0,
        switch_temp: bool = False,
    ):
        super().__init__(sequence_noise_scheduler)
        self.logits_temp = logits_temp
        self.switch_temp = switch_temp

    def step(
        self,
        decoder_logits: torch.Tensor,
        xt_struct: torch.Tensor,
        xt_seq: torch.Tensor,
        struct_inf_iter_num: int,
        last_struct_inf_iter_num: int,
        can_change_idx: torch.Tensor,
        x_denoised: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Assumes that we're doing MDLM inference with log linear noise schedule
        """
        t_next = get_seq_t(self.sequence_noise_scheduler, struct_inf_iter_num)
        t_last = get_seq_t(self.sequence_noise_scheduler, last_struct_inf_iter_num)

        if t_next < 0.0:
            return xt_seq

        dt = t_last - t_next

        logits_temp = self.logits_temp
        if self.switch_temp:
            logits_temp = self.logits_temp if t_next > 0.2 else 0.1

        decoder_logits = decoder_logits / logits_temp
        decoder_logits = decoder_logits - decoder_logits.logsumexp(
            dim=-1, keepdims=True
        )

        # Multiply the indices besides mask by dt
        log_dt = dt.log() if torch.is_tensor(dt) else np.log(dt)
        log_t_next = t_next.log() if torch.is_tensor(t_next) else np.log(t_next)
        log_t_last = t_last.log() if torch.is_tensor(t_last) else np.log(t_last)

        decoder_logits = decoder_logits + log_dt

        mask_log_like = log_t_next if t_next > 0.0 else LOG_ZERO
        decoder_logits[..., MASK_TOKEN_IDX] = mask_log_like

        decoder_logits = decoder_logits - log_t_last

        global ONE_TENSOR
        if ONE_TENSOR.device != decoder_logits.device:
            ONE_TENSOR = ONE_TENSOR.to(device=decoder_logits.device)

        assert torch.isclose(decoder_logits.exp().sum(dim=-1), ONE_TENSOR).all()

        new_x_pre, _ = stochastic_sample_from_categorical(decoder_logits)
        copy_flag = (xt_seq != MASK_TOKEN_IDX) * 1

        return (copy_flag * xt_seq) + ((1 - copy_flag) * new_x_pre)


class PathPlanningSamplingStrategy(BaseSequenceSamplingStrategy):
    """Path planning based sequence sampling strategy.

    Uses a score-based approach to decide which positions to unmask at each step.
    Supports multiple scoring methods (confidence, random, noisy confidence) and
    optional entropy-adaptive temperature scaling for diversity. Can optionally
    use an external PLM planner for re-scoring unmasked positions.
    """

    def __init__(
        self,
        sequence_noise_scheduler: SequenceInferenceNoiseScheduler,
        score_type: str,
        planner_name: str = None,
        logits_temp: float = 1.0,
        score_temp: float = 1.0,
        mask_stochasticity_strength: float = 1.0,
        should_ensure_unmasked_stay: bool = True,
        switch_temp: bool = False,
        allow_remasking: bool = False,
        entropy_adaptive_temp: bool = False,
        entropy_adaptive_beta: float = 1.0,
        entropy_adaptive_anneal_power: float = 0.0,
    ):
        super().__init__(sequence_noise_scheduler)

        self.planner, self.batch_converter, self.planner_idx_mapper = None, None, None
        if planner_name is not None:
            planner_name = MODEL_REGISTRY[planner_name]
            self.planner = AutoModelForMaskedLM.from_pretrained(planner_name)
            self.batch_converter = AutoTokenizer.from_pretrained(planner_name)

            vocab = self.batch_converter.get_vocab()
            self.planner_idx_mapper = torch.tensor(
                [
                    vocab[PRO_RES_IDX_TO_RESNAME_ONE[i]]
                    for i in range(len(PRO_RES_IDX_TO_RESNAME_ONE))
                ]
            )

        self.switch_temp = switch_temp

        self.allow_remasking = allow_remasking
        self.logits_temp = logits_temp
        self.score_temp = score_temp
        self.mask_stochasticity_strength = mask_stochasticity_strength
        self.should_ensure_unmasked_stay = should_ensure_unmasked_stay

        self.unmask_likelihood_cache = None

        self.score_type = score_type
        assert self.score_type in ["confidence", "random"]

        # Entropy-adaptive diversity mechanisms
        self.entropy_adaptive_temp = entropy_adaptive_temp
        self.entropy_adaptive_beta = entropy_adaptive_beta
        self.entropy_adaptive_anneal_power = entropy_adaptive_anneal_power

        # Structured diagnostic data for the last step (read by InferenceLoopImpl)
        self.last_step_diagnostics = {}

    def get_planner_logits(self, xt_seq: torch.Tensor) -> torch.Tensor:
        sequence_data = esm_tokens_to_sequence(xt_seq)

        batch_tokens = self.batch_converter(sequence_data, return_tensors="pt")
        batch_tokens = {k: v.to(xt_seq.device) for k, v in batch_tokens.items()}

        if self.planner.device != xt_seq.device:
            self.planner = self.planner.to(device=xt_seq.device)

        # Do the indexing to remove BOS/EOS tokens
        logits = self.planner(**batch_tokens).logits[:, 1:-1]
        if logits.shape[0] == 1:
            logits = logits.squeeze(dim=0)

        # Reorder the logits to get the logit indices to match the indexing our code uses
        return logits[..., self.planner_idx_mapper]

    def step(
        self,
        decoder_logits: torch.Tensor,
        xt_struct: torch.Tensor,
        xt_seq: torch.Tensor,
        struct_inf_iter_num: int,
        last_struct_inf_iter_num: int,
        can_change_idx: torch.Tensor,
        x_denoised: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_next = get_seq_t(self.sequence_noise_scheduler, struct_inf_iter_num)

        if t_next < 0.0:
            return xt_seq

        last_mask = xt_seq == MASK_TOKEN_IDX
        unmask_candidates = ~last_mask

        logits_temp = self.logits_temp
        if self.switch_temp:
            logits_temp = self.logits_temp if t_next > 0.2 else 0.1

        # Entropy-adaptive per-position temperature: increase temp at confident positions
        # to promote diversity where the model would otherwise always pick the same token.
        # Entropy is computed from raw logits (before temperature) so it reflects inherent
        # model confidence, not the switch_temp-adjusted distribution.
        if self.entropy_adaptive_temp:
            with torch.no_grad():
                probs = decoder_logits.softmax(dim=-1)
                H = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # per-position entropy
                H_max = math.log(20)  # max entropy for 20 amino acids
                # Anneal beta: strong early (t_next≈1), zero late (t_next≈0)
                effective_beta = self.entropy_adaptive_beta * (
                    t_next**self.entropy_adaptive_anneal_power
                )
                temp_multiplier = 1.0 + effective_beta * (H_max - H) / H_max
                per_pos_temp = logits_temp * temp_multiplier  # [..., seq_len]
            # Pre-divide logits by per-position temp, then sample with temp=1.0
            scaled_logits = decoder_logits / per_pos_temp.unsqueeze(-1)
            x0, logp = stochastic_sample_from_categorical(
                scaled_logits, temperature=1.0
            )
        else:
            x0, logp = stochastic_sample_from_categorical(
                decoder_logits, temperature=logits_temp
            )

        unmask_cache_cond = (
            self.should_ensure_unmasked_stay and self.score_type == "confidence"
        )

        if unmask_cache_cond and unmask_candidates.any():
            logp[unmask_candidates] = self.unmask_likelihood_cache[unmask_candidates]

        if self.planner is not None:
            planner_logits = self.get_planner_logits(xt_seq)
            planner_logp = (
                planner_logits.log_softmax(dim=-1)
                .gather(-1, x0.unsqueeze(-1))
                .squeeze(-1)
            )

            decoder_logits[unmask_candidates] = planner_logits[unmask_candidates]
            logp[unmask_candidates] = planner_logp[unmask_candidates]

        match self.score_type:
            case "confidence":
                score = logp

            case "random":
                score = torch.rand_like(logp).log()

        score[unmask_candidates] *= self.mask_stochasticity_strength

        mult_val = t_next
        score[~can_change_idx] = INF

        num_total_changeable_pre = can_change_idx.sum(dim=-1)
        num_total_changeable = (
            num_total_changeable_pre.item()
            if num_total_changeable_pre.ndim == 0
            else num_total_changeable_pre[0]
        )

        num_to_mask = math.floor(num_total_changeable * mult_val)
        mask = topk_lowest_masking(
            score, torch.tensor(num_to_mask, device=score.device)
        )
        xt_seq[mask] = MASK_TOKEN_IDX

        mask_to_x0 = last_mask & ~mask
        xt_seq[mask_to_x0] = x0[mask_to_x0]

        # --- Diagnostic data collection ---
        num_masked = (xt_seq == MASK_TOKEN_IDX).sum().item()
        num_unmasked = (xt_seq != MASK_TOKEN_IDX).sum().item()
        num_changed = mask_to_x0.sum().item()
        num_remasked_this_step = (
            (last_mask & ~mask)
            .logical_not_()
            .logical_and_(mask & ~last_mask)
            .sum()
            .item()
            if self.allow_remasking
            else 0
        )

        # Compute entropy of decoder logits (measures model uncertainty)
        with torch.no_grad():
            probs = decoder_logits.softmax(dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            mean_entropy = entropy.mean().item()
            std_entropy = entropy.std().item()
            min_entropy = entropy.min().item()
            max_entropy = entropy.max().item()

        self.last_step_diagnostics = {
            "t_next": float(t_next) if not torch.is_tensor(t_next) else t_next.item(),
            "logits_temp": float(logits_temp),
            "num_masked": num_masked,
            "num_unmasked": num_unmasked,
            "num_newly_unmasked": num_changed,
            "entropy_mean": mean_entropy,
            "entropy_std": std_entropy,
            "entropy_min": min_entropy,
            "entropy_max": max_entropy,
            "score_type": self.score_type,
            "switch_temp": str(self.switch_temp),
        }

        logger.debug(
            f"t_next={t_next:.4f} logits_temp={logits_temp:.4f} "
            f"masked={num_masked} unmasked={num_unmasked} "
            f"newly_unmasked={num_changed} "
            f"entropy(mean={mean_entropy:.3f} std={std_entropy:.3f} "
            f"min={min_entropy:.3f} max={max_entropy:.3f})"
        )

        return xt_seq
