# Copyright 2024 ByteDance and/or its affiliates.
# This file was modified in 2026 by Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn
from openfold.model.primitives import LayerNorm

from disco.data.constants import MASK_TOKEN_IDX, PRO_STD_RESIDUES, UNK_TOKEN_IDX
from disco.data.json_to_feature import SampleDictToFeatures
from disco.model.cogen_inference.cogen_generator import sample_diffusion_cogen

# external models
from disco.model.modules.lmpnn import LigandMPNN
from disco.model.modules.plm import LMWrapper
from disco.model.utils import (
    InferenceNoiseScheduler,
    simple_merge_dict_list,
    stochastic_sample_from_categorical,
)
from disco.utils.geometry import DistOneHotCalculator
from disco.utils.logger import get_logger
from disco.utils.torch_utils import autocasting_disable_decorator

from .modules.diffusion import JointDiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from .modules.head import DistogramHead
from .modules.pairformer import PairformerStack
from .modules.primitives import LinearNoBias

logger = get_logger(__name__)

_EPS = 1e-6
_NEG_INFINITY = -1000000.0
_MAX_PROTEIN_TOKEN_IDX = max(PRO_STD_RESIDUES.values())


def asdict_shallow(dataclass_instance):
    """
    Same as dataclasses.asdict but only returns the shallow attributes
    instead of deepcopying nested attributes.
    """
    return {
        field.name: getattr(dataclass_instance, field.name)
        for field in fields(dataclass_instance)
    }


class DISCO(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs, structure_encoder, sequence_sampling_strategy) -> None:
        super().__init__()
        self.configs = configs

        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        assert self.N_cycle > 0, f"N_cycle must be > 0, got {self.N_cycle}"

        # Diffusion scheduler
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.should_structure_encode_x0 = self.configs.should_structure_encode_x0
        self.structure_encoder_use_x0_input = (
            self.configs.structure_encoder_use_x0_input
        )

        self.diffusion_sequence_cycle = getattr(
            self.configs, "diffusion_sequence_cycle", 1
        )
        self.sequence_sampling_method = getattr(
            self.configs, "sequence_sampling_method", "uniform"
        )

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )

        self.lm_module = LMWrapper(
            configs.lm_name,
            output_dim=configs.model.pairformer.c_z,
            single_rep_output_dim=configs.c_s_inputs,
            freeze_lm=configs.freeze_lm,
        )

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )

        self.pairformer_stack = PairformerStack(**configs.model.pairformer)

        self.distance_calculator = DistOneHotCalculator(
            **configs.recycling_distance_calculator
        )

        self.dist_projector = LinearNoBias(
            in_features=self.distance_calculator.num_bins, out_features=self.c_z
        )

        self.use_joint_diff_module = configs.model.use_joint_diffusion_module
        self.sequence_sampling_strategy = sequence_sampling_strategy

        self.should_seq_struct_encode_after_pairformer = (
            self.configs.should_seq_struct_encode_after_pairformer
        )

        self.structure_encoder = structure_encoder
        self.linear_no_bias_s_cycle_x0_struct = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )

        self.layernorm_s_cycle_x0_struct = LayerNorm(self.c_s)

        self.linear_no_bias_s_cycle_xt_struct = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )

        self.layernorm_s_cycle_xt_struct = LayerNorm(self.c_s)

        self.diffusion_module = JointDiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)

        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        self.linear_no_bias_s_cycle_x0_seq = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )

        self.linear_no_bias_s_cycle_xt_seq = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )

        self.layernorm_s_cycle_x0_seq = LayerNorm(self.c_s)
        self.layernorm_s_cycle_xt_seq = LayerNorm(self.c_s)

        self.lm_encoding_dense_final_x0_seq_layer_norm = nn.LayerNorm(
            self.lm_module.lm_model.embed_dim
        )

        self.lm_encoding_dense_final_x0_seq_linear = nn.Linear(
            self.lm_module.lm_model.embed_dim, self.c_s
        )

        self.lm_encoding_dense_final_xt_seq_layer_norm = nn.LayerNorm(
            self.lm_module.lm_model.embed_dim
        )

        self.lm_encoding_dense_final_xt_seq_linear = nn.Linear(
            self.lm_module.lm_model.embed_dim, self.c_s
        )

    def get_inference_seq_decoder_logits(
        self,
        xt_seq: torch.Tensor,
        xt_struct: torch.Tensor,
        t_hat_noise_level_struct: torch.Tensor,
        t_hat_noise_level_seq: torch.Tensor,
        input_feature_dict: dict,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_skip: torch.Tensor,
        z_skip: torch.Tensor,
        enforce_unmask_stay: bool,
        logits: torch.Tensor | None = None,
    ):
        """Computes sequence decoder logits for inference.

        Applies SUBS parameterization to the provided logits, excluding unknown
        tokens and optionally enforcing that already-unmasked positions remain
        unchanged.

        Args:
            xt_seq: Noised sequence token indices.
            xt_struct: Noised structure coordinates.
            t_hat_noise_level_struct: Structure noise level.
            t_hat_noise_level_seq: Sequence noise level.
            input_feature_dict: Dictionary of input features.
            s_inputs: Single representation inputs.
            s_trunk: Single representation from trunk.
            z_trunk: Pair representation from trunk.
            s_skip: Single representation skip connection.
            z_skip: Pair representation skip connection.
            enforce_unmask_stay: If True, forces unmasked positions to keep
                their current token.
            logits: Pre-computed logits. If None, only SUBS parameterization
                is applied.

        Returns:
            Normalized log-probability logits with SUBS masking constraints
            applied.
        """
        return self.apply_subs_parameterization(
            logits, xt_seq, exclude_unk=True, enforce_unmask_stay=enforce_unmask_stay
        )

    def apply_subs_parameterization(
        self,
        logits_pre,
        xt_seq,
        exclude_unk: bool = False,
        enforce_unmask_stay: bool = True,
    ):
        """Applies SUBS parameterization to decoder logits with masking constraints.

        Sets mask token and nucleotide logits to negative infinity, normalizes
        via log-softmax, and optionally forces unmasked positions to retain
        their current token identity.

        Args:
            logits_pre: Raw decoder logits of shape ``(..., vocab_size)``.
            xt_seq: Current noised sequence token indices.
            exclude_unk: If True, also sets the unknown token logit to negative
                infinity.
            enforce_unmask_stay: If True, forces already-unmasked positions to
                keep their current token by zeroing their logit and setting all
                others to negative infinity.

        Returns:
            Normalized log-probability logits with masking constraints applied.
        """
        # SUBS parameterization
        logits_pre[..., MASK_TOKEN_IDX] = _NEG_INFINITY

        # NOTE: Setting here that we will never unmask to a nucleotide.
        #       When we want to do seq gen for DNA/RNA, take out this line.
        logits_pre[..., _MAX_PROTEIN_TOKEN_IDX + 1 :] = _NEG_INFINITY

        if exclude_unk:
            logits_pre[..., UNK_TOKEN_IDX] = _NEG_INFINITY

        logits = logits_pre - logits_pre.logsumexp(dim=-1, keepdim=True)

        if xt_seq.ndim < logits.ndim - 1 and logits.shape[0] > 1:
            xt_seq = xt_seq.unsqueeze(0)

        if len(xt_seq) == 1 and logits.ndim == 3 and len(logits) > 1:
            xt_seq = xt_seq.repeat(len(logits), 1)

        unmasked_indices = xt_seq != MASK_TOKEN_IDX
        logits = logits.squeeze()

        if enforce_unmask_stay:
            logits[unmasked_indices] = _NEG_INFINITY
            logits[unmasked_indices, xt_seq[unmasked_indices]] = 0

        return logits

    def get_pair_representation(
        self,
        input_feature_dict,
        z,
        s_inputs,
        config,
        inplace_safe,
        chunk_size,
        deepspeed_evo_attention_condition_satisfy,
    ):
        """Gets pair representation from the language model wrapper.

        Passes input features through the language model to update the pair
        representation and single representation inputs.

        Args:
            input_feature_dict: Dictionary of input features.
            z: Pair representation tensor.
            s_inputs: Single representation inputs tensor.
            config: Model configuration object.
            inplace_safe: Whether it is safe to use inplace operations.
            chunk_size: Chunk size for memory-efficient operations.
            deepspeed_evo_attention_condition_satisfy: Whether DeepSpeed evo
                attention conditions are met (requires token count > 16).

        Returns:
            Tuple of (updated z, updated s_inputs, lm_logits, None).
        """
        z, s_inputs, lm_logits = self.lm_module(input_feature_dict, z, s_inputs)
        return z, s_inputs, lm_logits

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        task_info: dict[str, Any],
        sigma_seq: torch.Tensor = None,
        xt_noised_struct: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
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
            input_feature_dict[key] = input_feature_dict[key].to("cuda")

        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 451]
        input_feature_dict["token_bonds"] = input_feature_dict["token_bonds"].to("cuda")
        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict)
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )

        check_args = [sigma, sigma_seq, xt_noised_struct]
        can_seq_struct_recycle = all(map(lambda x: x is not None, check_args))

        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        lm_logits, s_orig, z_orig = None, None, None
        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
            if cycle_no > 0 and can_seq_struct_recycle:
                s, z, _ = self._update_reps_with_seq_struct_encode(
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=z,
                    s_skip=s_orig,
                    z_skip=z_orig,
                    xt_seq=input_feature_dict["masked_prot_restype"],
                    xt_struct=xt_noised_struct,
                    sigma=sigma,
                    sigma_seq=sigma_seq,
                    input_feature_dict=input_feature_dict,
                    use_cached_xt_encodings=cycle_no > 1,
                )

            z, s_new, lm_logits = self.get_pair_representation(
                input_feature_dict,
                z,
                s_inputs,
                self.configs,
                inplace_safe,
                chunk_size,
                deepspeed_evo_attention_condition_satisfy,
            )

            trfmd_s_new = self.linear_no_bias_sinit(s_new)
            if cycle_no == 0:
                s_orig = trfmd_s_new
                z_orig = z

            s = s_init + self.linear_no_bias_s(self.layernorm_s(s)) + trfmd_s_new

            s, z = self.pairformer_stack(
                s,
                z,
                pair_mask=None,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                and deepspeed_evo_attention_condition_satisfy,
                use_lma=self.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        encoding_dict = None
        if self.should_seq_struct_encode_after_pairformer and can_seq_struct_recycle:
            s, z, encoding_dict = self._update_reps_with_seq_struct_encode(
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                s_skip=s_orig,
                z_skip=z_orig,
                xt_seq=input_feature_dict["masked_prot_restype"],
                xt_struct=xt_noised_struct,
                sigma=sigma,
                sigma_seq=sigma_seq,
                input_feature_dict=input_feature_dict,
                use_cached_xt_encodings=N_cycle > 1,
            )

        return s_inputs, s, z, s_orig, z_orig, lm_logits, encoding_dict

    def _get_recycle_latents(
        self,
        encode_obj: torch.Tensor,
        seq_or_struct_str: str,
        xt_or_x0_str: str,
        input_feature_dict: dict,
        structure_noise_level: torch.Tensor = None,
    ) -> torch.Tensor:
        """Extracts recycling latents (skip connections) from previous cycle outputs.

        Encodes the given object (sequence or structure) into latent
        representations, applies a layer norm and linear projection, and
        scatter-adds the result to match the full token dimension when needed.

        Args:
            encode_obj: Object to encode -- sequence token indices for "seq" or
                coordinate tensor for "struct".
            seq_or_struct_str: Whether to encode sequence ("seq") or structure
                ("struct").
            xt_or_x0_str: Whether encoding the noised input ("xt") or the
                denoised prediction ("x0"). Selects the corresponding layer
                norm and linear projection.
            input_feature_dict: Dictionary of input features.
            structure_noise_level: Noise level for structure encoding. Only
                used when ``seq_or_struct_str`` is "struct".

        Returns:
            Recycling latent tensor of shape ``(..., N_token, c_s)``.
        """
        assert seq_or_struct_str in {"seq", "struct"}
        assert xt_or_x0_str in {"xt", "x0"}

        latents = None
        if seq_or_struct_str == "seq":
            with torch.no_grad():
                latents = self.lm_module.encode_sequence(encode_obj, input_feature_dict)

            final_layer_norm = getattr(
                self,
                f"lm_encoding_dense_final_{xt_or_x0_str}_{seq_or_struct_str}_layer_norm",
            )

            final_dense = getattr(
                self,
                f"lm_encoding_dense_final_{xt_or_x0_str}_{seq_or_struct_str}_linear",
            )

            latents = final_dense(final_layer_norm(latents))
        else:
            latents = self._get_structure_latents(
                encode_obj,
                input_feature_dict,
                xt_or_x0_str,
                structure_noise_level=structure_noise_level,
            )

        layernorm = getattr(
            self, f"layernorm_s_cycle_{xt_or_x0_str}_{seq_or_struct_str}"
        )
        linear = getattr(
            self, f"linear_no_bias_s_cycle_{xt_or_x0_str}_{seq_or_struct_str}"
        )

        out = linear(layernorm(latents))

        N_token = input_feature_dict["residue_index"].shape[-1]
        if out.shape[-2] != N_token:
            prot_res_mask = input_feature_dict["prot_residue_mask"]

            token_idx = input_feature_dict["token_index"][prot_res_mask]
            if prot_res_mask.ndim == 2:
                token_idx = token_idx.reshape(len(prot_res_mask), -1)

            add_idx = token_idx.unsqueeze(-1).repeat(
                *[1 for _ in range(token_idx.ndim)], out.shape[-1]
            )

            expanded_tnsr_dims, scatter_dim = [], 0
            # If we have a batch size, include that in the expanded tensor dims
            if out.ndim == 3:
                if out.ndim != add_idx.ndim:
                    add_idx = add_idx.unsqueeze(0).repeat(out.shape[0], 1, 1)

                expanded_tnsr_dims.append(out.shape[0])
                scatter_dim = 1

            expanded_tnsr_dims.extend([N_token, out.shape[-1]])

            out_pre = torch.zeros(tuple(expanded_tnsr_dims), device=out.device)
            out = torch.scatter_add(
                input=out_pre, dim=scatter_dim, index=add_idx, src=out
            )

        # Need to write code to scatter for things like we did in old code
        return out

    def predict_x0_seq_struct(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_skip: torch.Tensor,
        z_skip: torch.Tensor,
        xt_struct: torch.Tensor,
        sigma: torch.Tensor,
        sigma_seq: torch.Tensor,
        input_feature_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts denoised x0 for both sequence and structure.

        Runs the joint diffusion module to produce structure coordinates and
        raw sequence logits, applies SUBS parameterization, and samples a
        discrete sequence prediction.

        Args:
            s_inputs: Single representation inputs.
            s_trunk: Single representation from trunk.
            z_trunk: Pair representation from trunk.
            s_skip: Single representation skip connection.
            z_skip: Pair representation skip connection.
            xt_struct: Noised structure coordinates.
            sigma: Structure noise level.
            sigma_seq: Sequence noise level.
            input_feature_dict: Dictionary of input features.

        Returns:
            Tuple of (x0_struct, x0_seq_pred) where x0_struct is the predicted
            denoised structure coordinates and x0_seq_pred is the sampled
            sequence token indices.
        """
        x0_struct, x0_seq_logits_pre = self.diffusion_module(
            x_noisy=xt_struct,
            t_hat_noise_level_struct=sigma,
            t_hat_noise_level_seq=sigma_seq,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_skip=s_skip,
            z_skip=z_skip,
        )

        decoder_logits = self.apply_subs_parameterization(
            x0_seq_logits_pre, input_feature_dict["masked_prot_restype"]
        )

        x0_seq_pred, _, _ = stochastic_sample_from_categorical(decoder_logits)
        return x0_struct, x0_seq_pred

    def _update_reps_with_seq_struct_encode(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_skip: torch.Tensor,
        z_skip: torch.Tensor,
        xt_seq: torch.Tensor,
        xt_struct: torch.Tensor,
        sigma: torch.Tensor,
        sigma_seq: torch.Tensor,
        input_feature_dict: dict,
        use_cached_xt_encodings: bool,
        detach_x0_preds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates single and pair representations by encoding sequence and structure.

        Predicts denoised x0 for both sequence and structure, computes
        recycling latents for both xt and x0, projects pairwise distances,
        and adds all contributions to the trunk representations.

        Args:
            s_inputs: Single representation inputs.
            s_trunk: Single representation from trunk.
            z_trunk: Pair representation from trunk.
            s_skip: Single representation skip connection.
            z_skip: Pair representation skip connection.
            xt_seq: Noised sequence token indices.
            xt_struct: Noised structure coordinates.
            sigma: Structure noise level.
            sigma_seq: Sequence noise level.
            input_feature_dict: Dictionary of input features.
            use_cached_xt_encodings: If True, reuses cached xt encodings from
                a previous call instead of recomputing them.
            detach_x0_preds: If True, detaches x0 predictions from the
                computation graph before encoding.

        Returns:
            Tuple of (s, z, encoding_dict) where s and z are the updated single
            and pair representations, and encoding_dict contains the cached
            sequence and structure encodings for both xt and x0.
        """
        x0_struct, x0_seq = self.predict_x0_seq_struct(
            xt_struct=xt_struct,
            sigma=sigma,
            sigma_seq=sigma_seq,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_skip=s_skip,
            z_skip=z_skip,
        )

        if detach_x0_preds:
            x0_struct, x0_seq = x0_struct.detach(), x0_seq.detach()

        x0_seq_encoding = self._get_recycle_latents(
            x0_seq, "seq", "x0", input_feature_dict
        )
        x0_struct_encoding = self._get_recycle_latents(
            x0_struct, "struct", "x0", input_feature_dict, sigma
        )

        if not use_cached_xt_encodings:
            self.xt_seq_encoding = self._get_recycle_latents(
                xt_seq, "seq", "xt", input_feature_dict
            )

            self.xt_struct_encoding = self._get_recycle_latents(
                xt_struct, "struct", "xt", input_feature_dict, sigma
            )

        dist_one_hot = self.distance_calculator.get_centre_dist_one_hot(
            x0_struct, input_feature_dict
        )
        projected_dists = self.dist_projector(dist_one_hot)

        if x0_seq_encoding.ndim - 1 == s_trunk.ndim:
            s_trunk = s_trunk.unsqueeze(dim=0)
            z_trunk = z_trunk.unsqueeze(dim=0)

        s = (
            s_trunk
            + x0_seq_encoding
            + x0_struct_encoding
            + self.xt_struct_encoding
            + self.xt_seq_encoding
        )

        z = z_trunk + projected_dists

        encoding_dict = {
            "seq": {
                "x0": x0_seq_encoding,
                "xt": self.xt_seq_encoding,
            },
            "struct": {
                "x0": x0_struct_encoding,
                "xt": self.xt_struct_encoding,
            },
        }

        return s, z, encoding_dict

    def _get_structure_latents(
        self,
        structure,
        input_feature_dict,
        xt_or_x0_str=None,
        init_structure_encoder=None,
        return_pre_dense_embeddings=False,
        structure_noise_level=None,
    ):
        """Extracts structure latents from the structure encoder.

        Dispatches to the structure encoder and returns the resulting latent
        representations.

        Args:
            structure: Structure coordinate tensor to encode.
            input_feature_dict: Dictionary of input features.
            xt_or_x0_str: Whether encoding the noised input ("xt") or the
                denoised prediction ("x0").
            init_structure_encoder: Optional override structure encoder. If
                None, uses ``self.structure_encoder``.
            return_pre_dense_embeddings: If True, returns embeddings before
                the final dense layer.
            structure_noise_level: Noise level passed to the structure encoder.

        Returns:
            Structure latent tensor, or None if no structure encoder is
            available.
        """
        structure_encoder = init_structure_encoder or self.structure_encoder

        structure_latents = None
        if structure_encoder is not None:
            if isinstance(structure_encoder, LigandMPNN):
                structure_latents = structure_encoder(
                    structure,
                    input_feature_dict,
                    return_pre_dense_embeddings,
                    structure_noise_level,
                )

            else:
                structure_latents = structure_encoder(
                    structure,
                    input_feature_dict["prot_residue_mask"].astype(bool).sum(),
                    input_feature_dict["backbone_atom_mask"],
                    input_feature_dict["backbone_no_oxygen_atom_mask"],
                )

        return structure_latents

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """Samples from the diffusion model using the cogen inference loop.

        Assembles diffusion sampling hyperparameters from the model config
        (gamma, noise scale, step scale, etc.) and delegates to
        ``sample_diffusion_cogen``, optionally disabling AMP autocasting.

        Args:
            **kwargs: Additional keyword arguments forwarded to the cogen
                sampling function (e.g., model, input_feature_dict,
                noise_schedule, sequence_sampling_strategy).

        Returns:
            The outputs of the cogen diffusion sampling process, including
            coordinates, decoder predictions, and trajectory diagnostics.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
                "seq_backbone_noise_sigma",
                "seq_backbone_noise_start_t",
            ]
        }
        # New params with backward-compatible defaults for older configs
        _configs["gamma_anneal"] = self.configs.sample_diffusion.get(
            "gamma_anneal", "none"
        )
        _configs["noisy_guidance"] = self.configs.sample_diffusion.get(
            "noisy_guidance", None
        )
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )

        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion_cogen
        )(**_configs, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
        sample2feat: SampleDictToFeatures = None,
        task_dict: dict[str, Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            N_cycle (int): Number of cycles of trunk.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []

        pred_dict, log_dict, time_tracker = self._main_inference_inner_loop(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            sample2feat=sample2feat,
            task_dict=task_dict,
        )
        pred_dicts.append(pred_dict)
        log_dicts.append(log_dict)
        time_trackers.append(time_tracker)

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            if key not in dict_list[0]:
                return None

            out = [x[key] for x in dict_list]
            if key != "atom_array":
                out = torch.cat(out, dim=0)
            else:
                if len(dict_list) == 1:
                    out = out[0]

            return out

        def _list_join(dict_list, key):
            if key not in dict_list[0]:
                return None

            return sum([x[key] for x in dict_list], [])

        pred_dicts[0]["x1_denoised"] = torch.stack(pred_dicts[0]["x1_denoised"], dim=0)
        pred_dicts[0]["xt_trajectory"] = torch.stack(
            pred_dicts[0]["xt_trajectory"], dim=0
        )
        pred_dicts[0]["trajectory_noise_levels"] = torch.tensor(
            pred_dicts[0]["trajectory_noise_levels"]
        )
        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
            "full_data": _list_join(pred_dicts, "full_data"),
            "plddt": _cat(pred_dicts, "plddt"),
            "pae": _cat(pred_dicts, "pae"),
            "pde": _cat(pred_dicts, "pde"),
            "x1_denoised": _cat(pred_dicts, "x1_denoised"),
            "xt_trajectory": _cat(pred_dicts, "xt_trajectory"),
            "trajectory_noise_levels": _cat(pred_dicts, "trajectory_noise_levels"),
            "decoder_prediction": _cat(pred_dicts, "decoder_prediction"),
            "resolved": _cat(pred_dicts, "resolved"),
            "atom_array": _cat(pred_dicts, "atom_array"),
        }

        all_pred_dict = {
            key: val for key, val in all_pred_dict.items() if val is not None
        }

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_inner_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
        N_seq_cycle: int = 0,  # not set in other circumstances, but cogen_eval would not be 0,
        cur_seq_cycle: int = 0,
        sample2feat: SampleDictToFeatures = None,
        task_dict: dict[str, Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Inner inference loop that generates a single sample.

        Runs diffusion sampling to produce predicted coordinates, decoder
        predictions, and diagnostic trajectories.

        Args:
            input_feature_dict: Dictionary of input features.
            N_cycle: Number of trunk recycling cycles.
            inplace_safe: Whether to use inplace operations safely.
            chunk_size: Chunk size for memory-efficient operations.
            N_seq_cycle: Total number of sequence cycles (used by cogen eval).
            cur_seq_cycle: Current sequence cycle index.
            sample2feat: Converter from sample dictionaries to feature
                tensors.
            task_dict: Dictionary of task-related information.

        Returns:
            Tuple of (pred_dict, log_dict, time_tracker) containing
            predictions, logging information, and per-stage timing.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_step = self.configs.sample_diffusion["N_step"]
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        N_atom = atom_to_token_idx.shape[-1]
        batch_size = 1 if atom_to_token_idx.ndim == 1 else atom_to_token_idx.shape[0]

        device = "cuda"
        noise_schedule = self.inference_noise_scheduler(N_step=N_step, device=device)

        (
            pred_dict["coordinate"],
            pred_dict["decoder_prediction"],
            pred_dict["atom_array"],
            pred_dict["x1_denoised"],
            pred_dict["trajectory_noise_levels"],
            pred_dict["xt_trajectory"],
            pred_dict["diagnostic_records"],
        ) = self.sample_diffusion(
            model=self,
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            noise_schedule=noise_schedule,
            noise_scheduler=self.inference_noise_scheduler,
            sequence_sampling_strategy=self.sequence_sampling_strategy,
            pairformer_chunk_size=chunk_size,
            inplace_safe=inplace_safe,
            sample2feat=sample2feat,
            task_dict=task_dict,
            x1_pred=True,
            bb_only=self.configs.bb_only,
            random_transform_ref_pos=self.configs.inference_random_transform_ref_pos,
            random_transform_msk_res=self.configs.inference_random_transform_msk_res,
            ref_pos_augment=self.configs.inference_ref_pos_augment,
            use_same_structure_for_all_seqs=self.configs.n_seq_duplicates_per_structure
            > 1,
            inference_loop_strategy_name=self.configs.inference_loop_strategy_name,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_st})
        if N_token > 2000:
            torch.cuda.empty_cache()

        pred_dict["token_array"] = input_feature_dict["token_array"]
        return pred_dict, log_dict, time_tracker

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        sample2feat: SampleDictToFeatures,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            sample2feat (SampleDictToFeatures): Sample to features converter.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any]]:
                Prediction and log dictionaries.
        """

        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        pred_dict, log_dict, time_tracker = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            sample2feat=sample2feat,
        )
        log_dict.update({"time": time_tracker})

        return pred_dict, log_dict
