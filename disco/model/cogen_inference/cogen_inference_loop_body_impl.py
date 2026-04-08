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

import copy
import logging
import math

import numpy as np
import torch
from omegaconf import DictConfig

from disco.data.constants import MASK_TOKEN_IDX, PRO_RES_IDX_TO_RESNAME_ONE
from disco.data.infer_data_pipeline import COLLATE_KEYS
from disco.data.json_to_feature import SampleDictToFeatures
from disco.model.cogen_inference.feature_dict_updater import FeatureDictUpdater
from disco.model.cogen_inference.sequence_sampling_strategy import (
    BaseSequenceSamplingStrategy,
    get_seq_t,
)
from disco.model.utils import centre_random_augmentation, InferenceNoiseScheduler

logger = logging.getLogger(__name__)


def get_pairformer_kwargs(
    model,
    input_feature_dict,
    N_cycle,
    inplace_safe,
    chunk_size,
    xt_noised_struct,
    sigma,
    sigma_seq,
    task_info,
):
    """Constructs kwargs for the PairformerStack forward pass.

    Assembles a dictionary of arguments needed by the pairformer, handling
    batch dimension expansion of sigma and sigma_seq when needed.

    Args:
        model: The DISCO model instance.
        input_feature_dict: Feature dictionary for the current step.
        N_cycle: Number of recycling iterations.
        inplace_safe: Whether in-place operations are safe.
        chunk_size: Chunk size for memory-efficient attention.
        xt_noised_struct: Noised structure coordinates.
        sigma: Structure noise level tensor.
        sigma_seq: Sequence noise level tensor.
        task_info: Task metadata dictionary.

    Returns:
        Dictionary of keyword arguments for the pairformer forward pass.
    """
    if xt_noised_struct.ndim == 3 and len(xt_noised_struct) > 1:
        if len(sigma) != len(xt_noised_struct):
            sigma = sigma.repeat(len(xt_noised_struct))
        if len(sigma_seq) != len(xt_noised_struct):
            sigma_seq = sigma_seq.repeat(len(xt_noised_struct))

    base_dict = {
        "input_feature_dict": input_feature_dict,
        "N_cycle": N_cycle,
        "inplace_safe": inplace_safe,
        "chunk_size": chunk_size,
        "xt_noised_struct": xt_noised_struct,
        "sigma": sigma,
        "task_info": task_info,
    }

    return {"sigma_seq": sigma_seq, **base_dict}


def get_diff_model_kwargs(
    model,
    x_noisy,
    t_hat_noise_level_struct,
    t_hat_noise_level_seq,
    input_feature_dict,
    s_inputs,
    s_trunk,
    z_trunk,
    s_skip,
    z_skip,
    chunk_size,
    inplace_safe,
):
    """Constructs kwargs for the diffusion model forward pass.

    Assembles a dictionary of arguments needed by the diffusion module,
    including noisy coordinates, noise levels, trunk representations,
    and skip connections.

    Args:
        model: The DISCO model instance.
        x_noisy: Noised all-atom coordinates.
        t_hat_noise_level_struct: Structure noise level for each sample.
        t_hat_noise_level_seq: Sequence noise level for each sample.
        input_feature_dict: Feature dictionary for the current step.
        s_inputs: Single representation inputs from the pairformer.
        s_trunk: Single representation trunk from the pairformer.
        z_trunk: Pair representation trunk from the pairformer.
        s_skip: Single representation skip connection (may be None).
        z_skip: Pair representation skip connection (may be None).
        chunk_size: Chunk size for memory-efficient attention.
        inplace_safe: Whether in-place operations are safe.

    Returns:
        Dictionary of keyword arguments for the diffusion module.
    """
    base_dict = {
        "x_noisy": x_noisy,
        "t_hat_noise_level_struct": t_hat_noise_level_struct,
        "input_feature_dict": input_feature_dict,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "chunk_size": chunk_size,
        "inplace_safe": inplace_safe,
    }

    return {
        "t_hat_noise_level_seq": t_hat_noise_level_seq,
        "s_skip": s_skip,
        "z_skip": z_skip,
        **base_dict,
    }


def get_inference_seq_decoder_call_kwargs(
    model,
    xt_struct,
    x0_struct,
    xt_seq,
    t_hat_noise_level_struct,
    t_hat_noise_level_seq,
    input_feature_dict,
    s_inputs,
    s_trunk,
    z_trunk,
    s_skip,
    z_skip,
    chunk_size,
    enforce_unmask_stay,
    inplace_safe,
    x_seq_logits=None,
):
    """Constructs kwargs for the sequence decoder during inference.

    Assembles a dictionary of arguments needed by the inference sequence
    decoder, including structure coordinates, sequence state, noise levels,
    and trunk representations.

    Args:
        model: The DISCO model instance.
        xt_struct: Noised structure at time t.
        x0_struct: Predicted clean structure (x0).
        xt_seq: Current masked/unmasked sequence at time t.
        t_hat_noise_level_struct: Structure noise level.
        t_hat_noise_level_seq: Sequence noise level.
        input_feature_dict: Feature dictionary for the current step.
        s_inputs: Single representation inputs from the pairformer.
        s_trunk: Single representation trunk from the pairformer.
        z_trunk: Pair representation trunk from the pairformer.
        s_skip: Single representation skip connection (may be None).
        z_skip: Pair representation skip connection (may be None).
        chunk_size: Chunk size for memory-efficient attention.
        enforce_unmask_stay: Whether to prevent re-masking of unmasked tokens.
        inplace_safe: Whether in-place operations are safe.
        x_seq_logits: Pre-computed sequence logits from diffusion module, if any.

    Returns:
        Dictionary of keyword arguments for the sequence decoder.
    """
    base_dict = {
        "xt_seq": xt_seq,
        "xt_struct": xt_struct,
        "input_feature_dict": input_feature_dict,
        "s_trunk": s_trunk,
        "enforce_unmask_stay": enforce_unmask_stay,
    }

    return {
        "logits": x_seq_logits,
        "t_hat_noise_level_struct": t_hat_noise_level_struct,
        "t_hat_noise_level_seq": t_hat_noise_level_seq,
        "s_inputs": s_inputs,
        "z_trunk": z_trunk,
        "s_skip": s_skip,
        "z_skip": z_skip,
        **base_dict,
    }


class InferenceLoopImpl:
    """Executor for a single step of the diffusion inference loop body.

    Manages the predictor-corrector denoising step, including pairformer
    evaluation, diffusion module denoising, sequence decoding, side chain
    updates, and optional noisy guidance.
    """

    def __init__(
        self,
        model: "DISCO",
        sequence_sampling_strategy: BaseSequenceSamplingStrategy,
        ftr_dict_updater: FeatureDictUpdater,
        gamma0: float,
        gamma_min: float,
        dtype: torch.dtype,
        N_cycle: int,
        inplace_safe: bool,
        task_dict: dict,
        pairformer_chunk_size: int,
        attn_chunk_size: int,
        x1_denoised: list,
        bb_only: bool,
        sample2feat: SampleDictToFeatures,
        random_transform_ref_pos: bool,
        random_transform_msk_res: bool,
        ref_pos_augment: bool,
        is_inverse_folding: bool,
        use_same_structure_for_all_seqs: bool,
        batch_size: int,
        batch_shape: tuple,
        inverse_folding_struct: torch.Tensor,
        inverse_folding_t_hat: float,
        noise_scheduler: InferenceNoiseScheduler,
        noise_scale_lambda: float,
        step_scale_eta: float,
        seq_backbone_noise_sigma: float = 0.0,
        seq_backbone_noise_sigma_min: float = 0.0,
        seq_backbone_noise_anneal: bool = False,
        seq_backbone_noise_start_t: float = 0.2,
        integrator: str = "euler",
        gamma_anneal: str = "none",
        noisy_guidance: DictConfig | None = None,
    ):
        self.model = model
        self.sequence_sampling_strategy = sequence_sampling_strategy
        self.ftr_dict_updater = ftr_dict_updater
        self.gamma0 = gamma0
        self.gamma_min = gamma_min
        self.dtype = dtype
        self.N_cycle = N_cycle
        self.inplace_safe = inplace_safe
        self.task_dict = task_dict
        self.pairformer_chunk_size = pairformer_chunk_size
        self.attn_chunk_size = attn_chunk_size
        self.x1_denoised = x1_denoised
        self.bb_only = bb_only
        self.sample2feat = sample2feat
        self.random_transform_ref_pos = random_transform_ref_pos
        self.random_transform_msk_res = random_transform_msk_res
        self.ref_pos_augment = ref_pos_augment
        self.is_inverse_folding = is_inverse_folding
        self.use_same_structure_for_all_seqs = use_same_structure_for_all_seqs
        self.batch_size = batch_size
        self.batch_shape = batch_shape
        self.inverse_folding_struct = inverse_folding_struct
        self.inverse_folding_t_hat = inverse_folding_t_hat
        self.noise_scheduler = noise_scheduler
        self.noise_scale_lambda = noise_scale_lambda
        self.step_scale_eta = step_scale_eta
        self.seq_backbone_noise_sigma = seq_backbone_noise_sigma
        self.seq_backbone_noise_sigma_min = seq_backbone_noise_sigma_min
        self.seq_backbone_noise_anneal = seq_backbone_noise_anneal
        self.seq_backbone_noise_start_t = seq_backbone_noise_start_t

        assert integrator in ("euler", "heun"), f"Unknown integrator: {integrator}"
        self.integrator = integrator
        assert gamma_anneal in (
            "none",
            "linear",
            "cosine",
        ), f"Unknown gamma_anneal: {gamma_anneal}"
        self.gamma_anneal = gamma_anneal

        # Noisy guidance config
        self.noisy_guidance_enabled = noisy_guidance is not None and noisy_guidance.get(
            "enabled", False
        )
        if self.noisy_guidance_enabled:
            self.guide_struct = noisy_guidance.get("guide_struct", True)
            self.omega_struct = noisy_guidance.get("omega_struct", 1.0)
            self.uncond_seq_time_mode = noisy_guidance.get(
                "uncond_seq_time_mode", "fixed"
            )
            self.uncond_seq_time = noisy_guidance.get("uncond_seq_time", 0.8)
            self.guide_seq = noisy_guidance.get("guide_seq", True)
            self.omega_seq = noisy_guidance.get("omega_seq", 1.0)
            self.uncond_struct_time_mode = noisy_guidance.get(
                "uncond_struct_time_mode", "fixed"
            )
            self.uncond_struct_time = noisy_guidance.get("uncond_struct_time", 0.8)
            self.sigma_max = noise_scheduler.time_to_noise_lvl(torch.tensor(0.0)).item()
            start_frac = noisy_guidance.get("guidance_start_frac", 0.0)
            end_frac = noisy_guidance.get("guidance_end_frac", 1.0)
            self.guidance_noise_upper = noise_scheduler.time_to_noise_lvl(
                torch.tensor(start_frac)
            ).item()
            self.guidance_noise_lower = (
                noise_scheduler.time_to_noise_lvl(torch.tensor(end_frac)).item()
                if end_frac < 1.0
                else 0.0
            )
            self.rescale_phi = noisy_guidance.get("rescale_phi", 0.7)
            self.omega_schedule = noisy_guidance.get("omega_schedule", "constant")

        self.trajectory_noise_levels = []
        self.xt_trajectory = []
        self.diagnostic_records = []

        self.s_inputs = None
        self.s_trunk = None
        self.z_trunk = None
        self.s_skip = None
        self.z_skip = None

    # ==================== Noisy Guidance Helpers ====================

    def _should_apply_guidance(self, c_tau_last_val: float) -> bool:
        """Check if noisy guidance should be applied at the current noise level."""
        if not self.noisy_guidance_enabled:
            return False
        if self.is_inverse_folding:
            return False
        has_struct_guidance = self.guide_struct and self.omega_struct != 1.0
        has_seq_guidance = self.guide_seq and self.omega_seq != 1.0
        if not has_struct_guidance and not has_seq_guidance:
            return False
        return self.guidance_noise_lower <= c_tau_last_val <= self.guidance_noise_upper

    def _get_uncond_sigma(
        self,
        current_sigma: torch.Tensor,
        time_mode: str,
        uncond_time: float,
        device: torch.device,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        """Compute the unconditional noise level sigma from the config.

        Args:
            current_sigma: the current noise level (sigma) for the modality.
                May include a (gamma+1) factor from the predictor-corrector sampler.
            time_mode: "fixed" or "offset".
            uncond_time: the time value (absolute or offset). This is a noise
                fraction where 0=clean, 1=fully noised.
            device: torch device.
            gamma: the gamma factor so we can strip (gamma+1) before inverting
                in offset mode, then re-apply it.

        Returns:
            The unconditional sigma, guaranteed >= current_sigma.
        """
        if time_mode == "fixed":
            # time_to_noise_lvl maps scheduler-time: 0 -> sigma_max, 1 -> sigma_min
            # uncond_time is a noise fraction (0=clean, 1=fully noised)
            # So scheduler-time = 1 - uncond_time (e.g., uncond_time=0.8 -> scheduler_time=0.2 -> high sigma)
            uncond_sigma = self.noise_scheduler.time_to_noise_lvl(
                torch.tensor(1.0 - uncond_time, device=device)
            )
            # Re-apply gamma factor to match the current sigma convention
            uncond_sigma = uncond_sigma * (gamma + 1)
        elif time_mode == "offset":
            # Strip gamma factor to recover the base sigma on the noise schedule
            base_sigma = current_sigma.item() / (gamma + 1)
            # Invert time_to_noise_lvl to get current scheduler time:
            #   sigma = sigma_data * (s_max^(1/rho) + t * (s_min^(1/rho) - s_max^(1/rho)))^rho
            #   t = ((sigma/sigma_data)^(1/rho) - s_max^(1/rho)) / (s_min^(1/rho) - s_max^(1/rho))
            ns = self.noise_scheduler
            current_sched_time = (
                (base_sigma / ns.sigma_data) ** (1.0 / ns.rho)
                - ns.s_max ** (1.0 / ns.rho)
            ) / (ns.s_min ** (1.0 / ns.rho) - ns.s_max ** (1.0 / ns.rho))
            # Go back toward noisier (lower scheduler time = higher sigma)
            target_sched_time = max(0.0, current_sched_time - uncond_time)
            uncond_sigma = self.noise_scheduler.time_to_noise_lvl(
                torch.tensor(target_sched_time, device=device)
            )
            # Re-apply gamma factor
            uncond_sigma = uncond_sigma * (gamma + 1)
        elif time_mode == "proportional":
            # uncond_time is a multiplier (>1.0): scale current sigma directly.
            # At high noise, this adds a large absolute noise offset (strong signal).
            # At low noise, only a small offset (gentle signal).
            uncond_sigma = current_sigma * uncond_time
        else:
            raise ValueError(f"Unknown uncond time mode: {time_mode}")
        # Ensure uncond sigma >= current sigma (more noised) and <= sigma_max
        return torch.clamp(uncond_sigma, min=current_sigma.item(), max=self.sigma_max)

    def _remask_sequence_to_time(
        self,
        xt_seq: torch.Tensor,
        prot_mask: torch.Tensor,
        target_mask_frac: float,
        can_change_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Re-mask the sequence to achieve approximately target_mask_frac masked
        among protein residues.

        Args:
            xt_seq: current sequence tensor [..., N_token], may be batched.
            prot_mask: boolean mask of protein residues [..., N_token].
            target_mask_frac: fraction of protein tokens that should be masked
                (0=none masked, 1=all masked).
            can_change_idx: boolean mask of designable positions. If provided,
                only these positions may be re-masked. Originally-fixed residues
                are never touched.

        Returns:
            Re-masked sequence tensor (cloned).
        """
        xt_seq_remasked = xt_seq.clone()

        if xt_seq.ndim == 1:
            # Unbatched
            prot_indices = prot_mask.nonzero(as_tuple=True)[0]
            n_prot = len(prot_indices)
            n_target_masked = int(target_mask_frac * n_prot)

            # Currently unmasked protein positions that are also designable
            currently_unmasked = xt_seq_remasked[prot_indices] != MASK_TOKEN_IDX
            if can_change_idx is not None:
                currently_unmasked = currently_unmasked & can_change_idx[prot_indices]
            unmasked_prot_indices = prot_indices[currently_unmasked]
            n_currently_unmasked = len(unmasked_prot_indices)

            # Currently masked count
            n_currently_masked = n_prot - n_currently_unmasked

            # How many more to mask
            n_to_mask = max(0, n_target_masked - n_currently_masked)
            if n_to_mask > 0 and n_currently_unmasked > 0:
                n_to_mask = min(n_to_mask, n_currently_unmasked)
                perm = torch.randperm(n_currently_unmasked, device=xt_seq.device)[
                    :n_to_mask
                ]
                xt_seq_remasked[unmasked_prot_indices[perm]] = MASK_TOKEN_IDX
        else:
            # Batched
            for b in range(xt_seq.shape[0]):
                prot_idx_b = prot_mask[b].nonzero(as_tuple=True)[0]
                n_prot = len(prot_idx_b)
                n_target_masked = int(target_mask_frac * n_prot)

                currently_unmasked = xt_seq_remasked[b][prot_idx_b] != MASK_TOKEN_IDX
                if can_change_idx is not None:
                    can_change_b = (
                        can_change_idx[b] if can_change_idx.ndim > 1 else can_change_idx
                    )
                    currently_unmasked = currently_unmasked & can_change_b[prot_idx_b]
                unmasked_idx = prot_idx_b[currently_unmasked]
                n_currently_masked = n_prot - len(unmasked_idx)

                n_to_mask = max(0, n_target_masked - n_currently_masked)
                if n_to_mask > 0 and len(unmasked_idx) > 0:
                    n_to_mask = min(n_to_mask, len(unmasked_idx))
                    perm = torch.randperm(len(unmasked_idx), device=xt_seq.device)[
                        :n_to_mask
                    ]
                    xt_seq_remasked[b][unmasked_idx[perm]] = MASK_TOKEN_IDX

        return xt_seq_remasked

    def _prepare_expanded_batch(
        self,
        input_feature_dict: dict,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor,
        sigma_seq: torch.Tensor,
        xt_seq: torch.Tensor,
        gamma: float,
        can_change_idx: torch.Tensor | None = None,
    ) -> tuple[
        dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """Construct expanded batch with cond + uncond copies for noisy guidance.

        Returns:
            input_ftr_expanded: input_feature_dict with batch dim expanded
            x_noisy_expanded: [expanded_B, N_atom, 3]
            sigma_expanded: [expanded_B] per-element struct sigma for pairformer
            sigma_seq_expanded: [expanded_B] per-element seq sigma for pairformer
            t_hat_expanded: [expanded_B] per-element struct sigma for diffusion
            sigma_seq_diff_expanded: [expanded_B] per-element seq sigma for diffusion
            expanded_batch_size: the new batch size
        """
        B = self.batch_size
        device = x_noisy.device

        # Start with lists of per-copy data
        x_noisy_copies = [x_noisy]
        sigma_copies = [t_hat.reshape(1).expand(B)]
        sigma_seq_copies = [sigma_seq.reshape(1).expand(B)]
        ftr_dict_copies = [input_feature_dict]

        has_struct_guidance = self.guide_struct and self.omega_struct != 1.0
        has_seq_guidance = self.guide_seq and self.omega_seq != 1.0

        # --- Structure guidance uncond copy (more-noised sequence) ---
        if has_struct_guidance:
            uncond_sigma_seq = self._get_uncond_sigma(
                sigma_seq,
                self.uncond_seq_time_mode,
                self.uncond_seq_time,
                device,
                gamma=gamma,
            )

            # Deep copy input_feature_dict COLLATE_KEYS for uncond sequence
            uncond_struct_ftr = {}
            for key in input_feature_dict:
                if key in COLLATE_KEYS:
                    val = input_feature_dict[key]
                    if torch.is_tensor(val):
                        uncond_struct_ftr[key] = val.clone()
                    elif isinstance(val, list):
                        uncond_struct_ftr[key] = copy.deepcopy(val)
                    elif isinstance(val, np.ndarray):
                        uncond_struct_ftr[key] = val.copy()
                    else:
                        uncond_struct_ftr[key] = copy.deepcopy(val)
                else:
                    uncond_struct_ftr[key] = input_feature_dict[key]

            # Re-mask protein tokens to the target uncond time
            prot_mask = uncond_struct_ftr["prot_residue_mask"]
            if self.uncond_seq_time_mode == "proportional":
                # Proportional mode: multiply current mask fraction by the configured
                # multiplier. This naturally scales guidance signal strength.
                flat_prot_mask = (
                    prot_mask.reshape(-1) if prot_mask.ndim > 1 else prot_mask
                )
                flat_xt_seq = xt_seq.reshape(-1) if xt_seq.ndim > 1 else xt_seq
                n_prot = flat_prot_mask.sum().item()
                n_currently_masked = (
                    (flat_xt_seq[flat_prot_mask] == MASK_TOKEN_IDX).sum().item()
                )
                current_mask_frac = n_currently_masked / max(n_prot, 1)
                target_mask_frac = min(1.0, current_mask_frac * self.uncond_seq_time)
            else:
                target_mask_frac = self.uncond_seq_time
            xt_seq_remasked = self._remask_sequence_to_time(
                xt_seq,
                prot_mask,
                target_mask_frac=target_mask_frac,
                can_change_idx=can_change_idx,
            )
            uncond_struct_ftr["masked_prot_restype"] = xt_seq_remasked

            # Determine which positions changed for FeatureDictUpdater
            xt_seq_changed_idx = xt_seq_remasked != xt_seq

            # Update features (ref conformers, etc.) for the changed positions
            if xt_seq_changed_idx.any():
                uncond_struct_ftr = self.ftr_dict_updater.update(
                    uncond_struct_ftr,
                    self.sample2feat,
                    xt_seq_remasked,
                    xt_seq_changed_idx,
                    self.bb_only,
                    self.random_transform_ref_pos,
                    self.random_transform_msk_res,
                    self.ref_pos_augment,
                )

            x_noisy_copies.append(x_noisy)  # same structure
            sigma_copies.append(t_hat.reshape(1).expand(B))  # same struct sigma
            sigma_seq_copies.append(
                uncond_sigma_seq.reshape(1).expand(B)
            )  # higher seq sigma
            ftr_dict_copies.append(uncond_struct_ftr)

        # --- Sequence guidance uncond copy (more-noised structure) ---
        if has_seq_guidance:
            uncond_sigma_struct = self._get_uncond_sigma(
                t_hat,
                self.uncond_struct_time_mode,
                self.uncond_struct_time,
                device,
                gamma=gamma,
            )

            # Forward-diffuse structure to higher noise level
            delta_sigma = torch.sqrt(
                torch.clamp(uncond_sigma_struct**2 - t_hat**2, min=0.0)
            )
            x_noisy_seq_uncond = x_noisy + delta_sigma * torch.randn_like(x_noisy)

            x_noisy_copies.append(x_noisy_seq_uncond)
            sigma_copies.append(
                uncond_sigma_struct.reshape(1).expand(B)
            )  # higher struct sigma
            sigma_seq_copies.append(sigma_seq.reshape(1).expand(B))  # same seq sigma
            ftr_dict_copies.append(input_feature_dict)  # same features

        # Combine copies along batch dimension.
        # x_noisy already has a batch dim (e.g. [B, N_atom, 3] or [1, N_atom, 3]),
        # so always cat along dim=0.
        x_noisy_expanded = torch.cat(x_noisy_copies, dim=0)
        sigma_expanded = torch.cat(sigma_copies, dim=0)
        sigma_seq_expanded = torch.cat(sigma_seq_copies, dim=0)

        # Feature dict tensors may NOT have a leading batch dim when B==1
        # (e.g. ref_pos is [N_atom, 3] not [1, N_atom, 3]). In that case we
        # must stack (adds new dim) rather than cat (which would concatenate
        # along the token/atom dim and corrupt the structure).
        ftr_needs_stack = B == 1 and input_feature_dict["ref_pos"].ndim == 2

        # Build expanded input_feature_dict
        input_ftr_expanded = {}
        for key in input_feature_dict:
            if key in COLLATE_KEYS:
                vals = [ftr[key] for ftr in ftr_dict_copies]
                if torch.is_tensor(vals[0]):
                    if ftr_needs_stack:
                        input_ftr_expanded[key] = torch.stack(vals, dim=0)
                    else:
                        input_ftr_expanded[key] = torch.cat(vals, dim=0)
                elif isinstance(vals[0], list):
                    merged = []
                    for v in vals:
                        merged.extend(v)
                    input_ftr_expanded[key] = merged
                elif isinstance(vals[0], np.ndarray):
                    if ftr_needs_stack:
                        input_ftr_expanded[key] = np.stack(vals, axis=0)
                    else:
                        input_ftr_expanded[key] = np.concatenate(vals, axis=0)
                else:
                    input_ftr_expanded[key] = vals[0]
            else:
                input_ftr_expanded[key] = input_feature_dict[key]

        expanded_batch_size = x_noisy_expanded.shape[0]
        t_hat_expanded = sigma_expanded
        sigma_seq_diff_expanded = sigma_seq_expanded

        return (
            input_ftr_expanded,
            x_noisy_expanded,
            sigma_expanded,
            sigma_seq_expanded,
            t_hat_expanded,
            sigma_seq_diff_expanded,
            expanded_batch_size,
        )

    def _get_effective_omega(self, omega_config: float, c_tau_last_val: float) -> float:
        """Compute effective omega, optionally decayed by schedule.

        Args:
            omega_config: the configured omega value (e.g. 3.0).
            c_tau_last_val: current noise level (sigma).

        Returns:
            Effective omega for this step.
        """
        if self.omega_schedule == "constant":
            return omega_config
        elif self.omega_schedule == "linear_decay":
            # omega decays linearly from omega_config at the noisy end of the
            # guidance interval to 1.0 at the clean end.
            span = self.guidance_noise_upper - self.guidance_noise_lower + 1e-8
            frac = (c_tau_last_val - self.guidance_noise_lower) / span
            frac = max(0.0, min(1.0, frac))
            return 1.0 + (omega_config - 1.0) * frac
        else:
            raise ValueError(f"Unknown omega_schedule: {self.omega_schedule}")

    def _split_and_interpolate_struct(
        self,
        x_denoised: torch.Tensor,
        x_noisy_cond: torch.Tensor,
        t_hat_cond: torch.Tensor,
        x_noisy_expanded: torch.Tensor,
        t_hat_expanded: torch.Tensor,
        c_tau_last_val: float = 0.0,
    ) -> torch.Tensor:
        """Split batched denoised output and apply structure guidance interpolation.

        For structure guidance (Alg 1): cond and uncond share same x_noisy and t_hat,
        so score interpolation is equivalent to interpolating x_denoised directly.

        Returns:
            x_denoised_guided: [B, N_atom, 3] guided denoised output
        """
        B = self.batch_size
        x_denoised_cond = x_denoised[:B]
        x_denoised_guided = x_denoised_cond
        idx = B

        has_struct_guidance = self.guide_struct and self.omega_struct != 1.0

        # Structure guidance (Alg 1): same t_hat, interpolate x_denoised directly
        if has_struct_guidance:
            omega_s = self._get_effective_omega(self.omega_struct, c_tau_last_val)
            x_denoised_struct_uncond = x_denoised[idx : idx + B]
            x_denoised_guided = (
                omega_s * x_denoised_guided + (1 - omega_s) * x_denoised_struct_uncond
            )
            idx += B

        # CFG rescale (Lin et al.): renormalize guided x0 to match std of
        # conditional x0, then interpolate with phi.
        if has_struct_guidance and self.rescale_phi > 0:
            # Compute per-sample std over atom and coordinate dims
            std_cond = x_denoised_cond.std(
                dim=list(range(1, x_denoised_cond.ndim)), keepdim=True
            )
            std_guided = x_denoised_guided.std(
                dim=list(range(1, x_denoised_guided.ndim)), keepdim=True
            )
            x_denoised_rescaled = x_denoised_guided * (std_cond / (std_guided + 1e-8))
            x_denoised_guided = (
                self.rescale_phi * x_denoised_rescaled
                + (1 - self.rescale_phi) * x_denoised_guided
            )

        return x_denoised_guided

    def _split_and_interpolate_seq_logits(
        self,
        x_seq_logits: torch.Tensor | None,
        c_tau_last_val: float = 0.0,
    ) -> torch.Tensor | None:
        """Split batched seq logits and apply sequence guidance interpolation (Alg 2).

        Arithmetic average of logits (per paper).
        """
        if x_seq_logits is None:
            return None

        B = self.batch_size
        x_seq_logits_cond = x_seq_logits[:B]
        idx = B

        has_struct_guidance = self.guide_struct and self.omega_struct != 1.0
        has_seq_guidance = self.guide_seq and self.omega_seq != 1.0

        # Skip struct guidance copy (it doesn't contribute to seq logits guidance)
        if has_struct_guidance:
            idx += B

        if has_seq_guidance:
            omega_q = self._get_effective_omega(self.omega_seq, c_tau_last_val)
            x_seq_logits_uncond = x_seq_logits[idx : idx + B]
            return omega_q * x_seq_logits_cond + (1 - omega_q) * x_seq_logits_uncond

        return x_seq_logits_cond

    # ==================== End Noisy Guidance Helpers ====================

    def execute(
        self,
        x_l: torch.Tensor,
        xt_seq: torch.Tensor,
        c_tau: torch.Tensor,
        c_tau_last: torch.Tensor,
        i: int,
        last_i: int,
        input_feature_dict: dict,
        xt_seq_changed: bool,
        can_change_idx: torch.Tensor,
        freeze_struct: bool = False,
        freeze_seq: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_inverse_folding:
            x_l = self.inverse_folding_struct

        # [..., N_sample, N_atom, 3]
        x_l = (
            centre_random_augmentation(x_input_coords=x_l, N_sample=1)
            .squeeze(dim=-3)
            .to(self.dtype)
        )

        # Denoise with a predictor-corrector sampler
        # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
        if self.gamma_anneal == "none":
            gamma = float(self.gamma0) if c_tau > self.gamma_min else 0
        else:
            # Anneal gamma based on noise level: more noise early, less late
            if c_tau > self.gamma_min:
                # c_tau decreases over time; normalize to [0, 1] using the noise scheduler range
                sigma_max = self.noise_scheduler.time_to_noise_lvl(
                    torch.tensor(0.0)
                ).item()
                frac = float(c_tau) / sigma_max  # ~1 at start, ~0 at end
                if self.gamma_anneal == "linear":
                    gamma = float(self.gamma0) * frac
                elif self.gamma_anneal == "cosine":
                    gamma = float(self.gamma0) * math.cos(math.pi * (1 - frac) / 2)
            else:
                gamma = 0

        if self.is_inverse_folding:
            x_noisy = x_l
            t_hat = self.inverse_folding_t_hat
        else:
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + self.noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=x_l.device, dtype=self.dtype
            )

        if self.use_same_structure_for_all_seqs:
            assert self.batch_size > 1
            x_noisy[:] = x_noisy[0]

        seq_time_pre = get_seq_t(
            self.sequence_sampling_strategy.sequence_noise_scheduler, i
        )

        seq_time = 1 - seq_time_pre
        if not torch.is_tensor(seq_time):
            seq_time = torch.tensor(seq_time, device=xt_seq.device)

        sigma_seq = self.noise_scheduler.time_to_noise_lvl(seq_time) * (gamma + 1)

        # Noisy guidance: check if we should apply it this step
        c_tau_last_val = (
            float(c_tau_last.item())
            if torch.is_tensor(c_tau_last)
            else float(c_tau_last)
        )
        apply_guidance = self._should_apply_guidance(c_tau_last_val)

        # Prepare expanded batch if guidance is active
        # guidance_stacked tracks whether we added a new batch dim (unbatched B=1 case)
        guidance_stacked = False
        if apply_guidance:
            (
                input_ftr_expanded,
                x_noisy_expanded,
                sigma_expanded,
                sigma_seq_expanded,
                t_hat_expanded,
                sigma_seq_diff_expanded,
                expanded_batch_size,
            ) = self._prepare_expanded_batch(
                input_feature_dict,
                x_noisy,
                t_hat,
                sigma_seq,
                xt_seq,
                gamma=gamma,
                can_change_idx=can_change_idx,
            )
            # If we added a batch dim to feature dict tensors (stack), remember so we
            # can squeeze it back later. Check feature dict, not x_noisy, since
            # x_noisy may have a synthetic [1,...] dim while features are unbatched.
            guidance_stacked = (
                self.batch_size == 1 and input_feature_dict["ref_pos"].ndim == 2
            )
        else:
            input_ftr_expanded = input_feature_dict
            x_noisy_expanded = x_noisy
            sigma_expanded = t_hat.unsqueeze(dim=0)
            sigma_seq_expanded = sigma_seq.unsqueeze(dim=0)

        # Re-run pairformer when sequence changed, or always when guidance is active
        # (guidance creates an expanded batch that needs fresh pairformer output)
        if xt_seq_changed or apply_guidance:
            outs = self.model.get_pairformer_output(
                **get_pairformer_kwargs(
                    model=self.model,
                    input_feature_dict=input_ftr_expanded,
                    N_cycle=self.N_cycle,
                    inplace_safe=self.inplace_safe,
                    chunk_size=self.pairformer_chunk_size,
                    xt_noised_struct=x_noisy_expanded,
                    sigma=sigma_expanded,
                    sigma_seq=sigma_seq_expanded,
                    task_info=self.task_dict,
                )
            )

            match len(outs):
                case 7:
                    (
                        self.s_inputs,
                        self.s_trunk,
                        self.z_trunk,
                        self.s_skip,
                        self.z_skip,
                        _,
                        _,
                    ) = outs
                case 6:
                    (
                        self.s_inputs,
                        self.s_trunk,
                        self.z_trunk,
                        self.s_skip,
                        self.z_skip,
                        _,
                    ) = outs
                case 5:
                    self.s_inputs, self.s_trunk, self.z_trunk, _, _ = outs
                case _:
                    raise ValueError(f"Unexpected pairformer output length {len(outs)}")

        # 2. Denoise from x_{t_hat} to x_{c_tau}
        if apply_guidance:
            t_hat_for_diff = t_hat_expanded
            sigma_seq_for_diff = sigma_seq_diff_expanded
        else:
            t_hat_for_diff = t_hat.unsqueeze(0).repeat(
                *self.batch_shape if len(self.batch_shape) > 0 else 1
            )
            sigma_seq_for_diff = sigma_seq.unsqueeze(0).repeat(
                *self.batch_shape if len(self.batch_shape) > 0 else 1
            )

        diff_model_kwargs = get_diff_model_kwargs(
            model=self.model,
            x_noisy=x_noisy_expanded,
            t_hat_noise_level_struct=t_hat_for_diff,
            t_hat_noise_level_seq=sigma_seq_for_diff,
            input_feature_dict=input_ftr_expanded,
            s_inputs=self.s_inputs,
            s_trunk=self.s_trunk,
            z_trunk=self.z_trunk,
            s_skip=self.s_skip,
            z_skip=self.z_skip,
            chunk_size=self.attn_chunk_size,
            inplace_safe=self.inplace_safe,
        )

        x_seq_logits = None
        denoiser_outputs = self.model.diffusion_module(**diff_model_kwargs)
        if isinstance(denoiser_outputs, tuple):
            x_denoised, x_seq_logits = denoiser_outputs
        else:
            x_denoised = denoiser_outputs

        # Split and interpolate guided outputs back to original batch size.
        # x_denoised and x_seq_logits come from x_noisy_expanded which was cat'd
        # (always has batch dim), so [:B] gives the correct [B, ...] shape matching
        # the original x_noisy / x_denoised shape. No squeeze needed.
        if apply_guidance:
            x_denoised = self._split_and_interpolate_struct(
                x_denoised,
                x_noisy,
                t_hat,
                x_noisy_expanded,
                t_hat_expanded,
                c_tau_last_val=c_tau_last_val,
            )
            x_seq_logits = self._split_and_interpolate_seq_logits(
                x_seq_logits,
                c_tau_last_val=c_tau_last_val,
            )

        # When guidance was active, restore pairformer outputs to cond-only batch size
        # so downstream code (Heun, seq decoder, etc.) sees original batch dimension.
        # Invariant: conditional copy is always first in the expanded batch, so [:B]
        # gives the correct conditional pairformer outputs. These are cached as
        # self.s_trunk etc. and reused on subsequent steps where xt_seq_changed=False
        # and apply_guidance=False.
        if apply_guidance:
            B = self.batch_size
            self.s_inputs = self.s_inputs[:B]
            self.s_trunk = self.s_trunk[:B]
            self.z_trunk = self.z_trunk[:B]
            if self.s_skip is not None:
                self.s_skip = self.s_skip[:B]
            if self.z_skip is not None:
                self.z_skip = self.z_skip[:B]
            # If we stacked (unbatched B=1 case), squeeze the batch dim back out
            if guidance_stacked:
                self.s_inputs = self.s_inputs.squeeze(0)
                self.s_trunk = self.s_trunk.squeeze(0)
                self.z_trunk = self.z_trunk.squeeze(0)
                if self.s_skip is not None:
                    self.s_skip = self.s_skip.squeeze(0)
                if self.z_skip is not None:
                    self.z_skip = self.z_skip.squeeze(0)

        # Restore original batch-sized t_hat and sigma_seq for the Euler step
        t_hat = t_hat.unsqueeze(0).repeat(
            *self.batch_shape if len(self.batch_shape) > 0 else 1
        )
        sigma_seq = sigma_seq.unsqueeze(0).repeat(
            *self.batch_shape if len(self.batch_shape) > 0 else 1
        )

        d_cur = (x_noisy - x_denoised) / t_hat[
            ..., None, None
        ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
        dt = c_tau - t_hat
        if not freeze_struct:
            if self.integrator == "heun" and c_tau > 0:
                # Heun's 2nd-order method (EDM Algorithm 2):
                # 1) Euler prediction to get x_euler at c_tau
                x_euler = x_noisy + dt[..., None, None] * d_cur
                # 2) Evaluate denoiser at the Euler-predicted point
                c_tau_batch = c_tau.unsqueeze(0).repeat(
                    *self.batch_shape if len(self.batch_shape) > 0 else 1
                )
                heun_diff_kwargs = get_diff_model_kwargs(
                    model=self.model,
                    x_noisy=x_euler,
                    t_hat_noise_level_struct=c_tau_batch,
                    t_hat_noise_level_seq=sigma_seq,
                    input_feature_dict=input_feature_dict,
                    s_inputs=self.s_inputs,
                    s_trunk=self.s_trunk,
                    z_trunk=self.z_trunk,
                    s_skip=self.s_skip,
                    z_skip=self.z_skip,
                    chunk_size=self.attn_chunk_size,
                    inplace_safe=self.inplace_safe,
                )
                heun_outputs = self.model.diffusion_module(**heun_diff_kwargs)
                x_denoised_2 = (
                    heun_outputs[0] if isinstance(heun_outputs, tuple) else heun_outputs
                )
                d_prime = (x_euler - x_denoised_2) / c_tau_batch[..., None, None]
                # 3) Average the two gradients
                x_l = x_noisy + dt[..., None, None] * (0.5 * d_cur + 0.5 * d_prime)
            else:
                # Euler step
                x_l = x_noisy + self.step_scale_eta * dt[..., None, None] * d_cur

        self.x1_denoised.append(x_denoised.detach().cpu())
        self.xt_trajectory.append(x_l.detach().cpu())
        self.trajectory_noise_levels.append(
            c_tau_last.detach().cpu().item()
            if torch.is_tensor(c_tau_last)
            else float(c_tau_last)
        )
        xt_seq = input_feature_dict["masked_prot_restype"]

        prot_mask = input_feature_dict["prot_residue_mask"]

        # Inject backbone noise for sequence decoder (late-stage only)
        # This keeps structural input in the beneficial noise regime for inverse folding
        x_denoised_for_seq = x_denoised
        x_noisy_for_seq = x_noisy

        seq_logit_inf_input = get_inference_seq_decoder_call_kwargs(
            model=self.model,
            xt_struct=x_noisy_for_seq,
            x0_struct=x_denoised_for_seq,
            xt_seq=xt_seq,
            t_hat_noise_level_struct=t_hat,
            t_hat_noise_level_seq=sigma_seq,
            input_feature_dict=input_feature_dict,
            s_inputs=self.s_inputs,
            s_trunk=self.s_trunk,
            z_trunk=self.z_trunk,
            s_skip=self.s_skip,
            z_skip=self.z_skip,
            chunk_size=self.attn_chunk_size,
            enforce_unmask_stay=self.sequence_sampling_strategy.should_ensure_unmasked_stay,
            inplace_safe=self.inplace_safe,
            x_seq_logits=x_seq_logits,
        )

        seq_decoder_logits = self.model.get_inference_seq_decoder_logits(
            **seq_logit_inf_input
        )

        xt_seq_new = self.sequence_sampling_strategy.step(
            seq_decoder_logits,
            x_l,
            xt_seq.clone(),
            i,
            last_i,
            can_change_idx,
            x_denoised=x_denoised,
        )

        xt_seq_new[~can_change_idx] = xt_seq[~can_change_idx]

        if freeze_seq:
            xt_seq_new = xt_seq

        xt_seq_changed_idx = xt_seq_new != xt_seq
        xt_seq_changed = xt_seq_changed_idx.any()

        logger.debug(f"Had {xt_seq_changed_idx.sum()} changes")

        if xt_seq_changed:
            logger.debug(
                f"New residues are {xt_seq_new[xt_seq_changed_idx]}, old was {xt_seq[xt_seq_changed_idx]}"
                f"New residue names are {''.join([PRO_RES_IDX_TO_RESNAME_ONE[i.item()] for i in xt_seq_new[xt_seq_changed_idx]])}"
            )

            xt_seq = xt_seq_new

            input_feature_dict = self.ftr_dict_updater.update(
                input_feature_dict,
                self.sample2feat,
                xt_seq,
                xt_seq_changed_idx,
                self.bb_only,
                self.random_transform_ref_pos,
                self.random_transform_msk_res,
                self.ref_pos_augment,
            )

        # Collect structured diagnostic data
        step_diag = {
            "step": i,
            "c_tau": float(c_tau.item()) if torch.is_tensor(c_tau) else float(c_tau),
            "c_tau_last": float(c_tau_last.item())
            if torch.is_tensor(c_tau_last)
            else float(c_tau_last),
            "freeze_struct": freeze_struct,
            "freeze_seq": freeze_seq,
            "num_seq_changes": int(xt_seq_changed_idx.sum().item()),
            "noisy_guidance_active": apply_guidance,
        }
        if apply_guidance:
            step_diag["noisy_guidance_omega_struct"] = self.omega_struct
            step_diag["noisy_guidance_omega_seq"] = self.omega_seq
        if hasattr(self.sequence_sampling_strategy, "last_step_diagnostics"):
            step_diag.update(self.sequence_sampling_strategy.last_step_diagnostics)
        self.diagnostic_records.append(step_diag)

        return x_l, xt_seq, input_feature_dict, xt_seq_changed
