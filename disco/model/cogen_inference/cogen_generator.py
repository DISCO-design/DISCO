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
from typing import Any

import torch
from biotite.structure import AtomArray
from omegaconf import DictConfig

from disco.data.json_to_feature import SampleDictToFeatures
from disco.model.cogen_inference.cogen_inference_loop_body_impl import InferenceLoopImpl
from disco.model.cogen_inference.cogen_inference_loop_strategies import (
    InferenceLoopStrategyFactory,
)
from disco.model.cogen_inference.feature_dict_updater import FeatureDictUpdater
from disco.model.cogen_inference.sequence_sampling_strategy import (
    BaseSequenceSamplingStrategy,
)
from disco.model.utils import InferenceNoiseScheduler

logger = logging.getLogger(__name__)


def sample_diffusion_cogen(
    model: "DISCO",
    input_feature_dict: dict[str, Any],
    noise_schedule: torch.Tensor,
    noise_scheduler: InferenceNoiseScheduler,
    sequence_sampling_strategy: BaseSequenceSamplingStrategy,
    sample2feat: SampleDictToFeatures,
    random_transform_ref_pos: bool,
    random_transform_msk_res: bool,
    ref_pos_augment: bool,
    N_cycle: int = 4,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    pairformer_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    inplace_safe: bool = False,
    attn_chunk_size: int | None = None,
    task_dict: dict[str, Any] = None,
    x1_pred: bool = False,  # if output x1 each step
    bb_only: bool = False,
    x_structure_inverse_folding: torch.Tensor | None = None,
    sigma_inverse_folding: torch.Tensor | None = 0.2,
    use_same_structure_for_all_seqs: bool = False,
    inference_loop_strategy_name: str = "FullSeqResetInferenceLoop",
    looped_reset_inf_settings: DictConfig | None = None,
    backtracking_inf_settings: DictConfig | None = None,
    post_generation_refinement: DictConfig | None = None,
    seq_backbone_noise_sigma: float = 0.0,
    seq_backbone_noise_sigma_min: float = 0.0,
    seq_backbone_noise_anneal: bool = False,
    seq_backbone_noise_start_t: float = 0.2,
    integrator: str = "euler",
    gamma_anneal: str = "none",
    noisy_guidance: DictConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, AtomArray]:
    """Implements Algorithm 18 in AF3.
    It performs denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        model (DISCO): the entire DISCO model.
        input_feature_dict (dict[str, Any]): input meta feature dict
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        sequence_sampling_strategy (BaseSequenceSamplingStrategy): The sequence strategy which, given
            decoder logits, takes the next sequence step.
        sample2feat (SampleDictToFeatures): An already initialized object to help featurize given an
            updated xt sequence.
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, AtomArray]
            torch.Tensor: the denoised coordinates of x in inference stage
                [..., N_sample, N_atom, 3]

            torch.Tensor: the denoised protein sequence
                [..., N_sample, N_residues]

            AtomArray: the atom array updated with the denoised protein sequence

    """
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    N_atom = atom_to_token_idx.size(-1)
    batch_size = 1 if atom_to_token_idx.ndim == 1 else atom_to_token_idx.shape[0]
    batch_shape = (batch_size,)

    device = noise_schedule.device
    dtype = noise_schedule.dtype

    # make sure same device
    input_feature = {
        "ref_pos": 3,
        "ref_charge": 1,
        "ref_mask": 1,
        "ref_element": 128,
        "ref_atom_name_chars": 4 * 64,
        "asym_id": 1,
        "residue_index": 1,
        "entity_id": 1,
        "sym_id": 1,
        "token_index": 1,
        "token_bonds": 1,
    }
    for key, __ in input_feature.items():
        input_feature_dict[key] = input_feature_dict[key].to(device)

    is_inverse_folding = "x_structure_inverse_folding" in input_feature_dict

    # init noise
    # [..., N_sample, N_atom, 3]
    # x_l shape:
    inverse_folding_struct = inverse_folding_t_hat = None
    if is_inverse_folding:
        inverse_folding_struct = (
            input_feature_dict["x_structure_inverse_folding"]
            .to(dtype=dtype)
            .unsqueeze(dim=0)
            .repeat(batch_size, 1, 1)
        )

        x_l = inverse_folding_struct
        inverse_folding_t_hat = torch.tensor(sigma_inverse_folding, device=x_l.device)

    else:
        x_l = (
            noise_schedule[0]
            * torch.randn(size=(*batch_shape, N_atom, 3), device=device, dtype=dtype)
            * noise_scale_lambda
        )

    if use_same_structure_for_all_seqs:
        assert batch_size > 1
        x_l[:] = x_l[0]

    x1_denoised = []
    xt_seq = input_feature_dict["masked_prot_restype"]

    sequence_sampling_strategy.start(len(noise_schedule) - 1)

    with FeatureDictUpdater(batch_size) as ftr_dict_updater:
        loop_executor = InferenceLoopImpl(
            model=model,
            sequence_sampling_strategy=sequence_sampling_strategy,
            ftr_dict_updater=ftr_dict_updater,
            gamma0=gamma0,
            gamma_min=gamma_min,
            dtype=dtype,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            task_dict=task_dict,
            pairformer_chunk_size=pairformer_chunk_size,
            attn_chunk_size=attn_chunk_size,
            x1_denoised=x1_denoised,
            bb_only=bb_only,
            sample2feat=sample2feat,
            random_transform_ref_pos=random_transform_ref_pos,
            random_transform_msk_res=random_transform_msk_res,
            ref_pos_augment=ref_pos_augment,
            is_inverse_folding=is_inverse_folding,
            use_same_structure_for_all_seqs=use_same_structure_for_all_seqs,
            batch_size=batch_size,
            batch_shape=batch_shape,
            inverse_folding_struct=inverse_folding_struct,
            inverse_folding_t_hat=inverse_folding_t_hat,
            noise_scheduler=noise_scheduler,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
            seq_backbone_noise_sigma=seq_backbone_noise_sigma,
            seq_backbone_noise_sigma_min=seq_backbone_noise_sigma_min,
            seq_backbone_noise_anneal=seq_backbone_noise_anneal,
            seq_backbone_noise_start_t=seq_backbone_noise_start_t,
            integrator=integrator,
            gamma_anneal=gamma_anneal,
            noisy_guidance=noisy_guidance,
        )

        inf_loop_strat = InferenceLoopStrategyFactory.create_loop_strategy(
            inference_loop_strategy_name,
            noise_schedule=noise_schedule,
            noise_scheduler=noise_scheduler,
            ftr_dict_updater=ftr_dict_updater,
            loop_executor=loop_executor,
            looped_reset_inf_settings=looped_reset_inf_settings,
            backtracking_inf_settings=backtracking_inf_settings,
            post_generation_refinement=post_generation_refinement,
        )

        outs = inf_loop_strat.run_loop(x_l, xt_seq, input_feature_dict)

    return outs
