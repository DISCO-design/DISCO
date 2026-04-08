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

import copy

import torch
import torch.nn as nn
from omegaconf.omegaconf import open_dict
from openfold.model.primitives import LayerNorm
from openfold.utils.checkpointing import get_checkpoint_fn

from disco.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from disco.model.modules.primitives import LinearNoBias, Transition
from disco.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from disco.model.utils import expand_at_dim


class DiffusionConditioning(nn.Module):
    """Implements Algorithm 21 in AF3

    Args:
        sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
        c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
        do_fourier_embed_seq: bool = False,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        # Line1-Line3:
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs, out_features=self.c_s
        )
        # Line8-Line9

        self.do_fourier_embed_seq = do_fourier_embed_seq
        if not self.do_fourier_embed_seq:
            self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
            self.layernorm_n = LayerNorm(c_noise_embedding)
            self.linear_no_bias_n = LinearNoBias(
                in_features=c_noise_embedding, out_features=self.c_s
            )
        else:
            self.fourier_embedding_struct = FourierEmbedding(c=c_noise_embedding)
            self.layernorm_n_struct = LayerNorm(c_noise_embedding)
            self.linear_no_bias_n_struct = LinearNoBias(
                in_features=c_noise_embedding, out_features=self.c_s
            )

            self.fourier_embedding_seq = FourierEmbedding(c=c_noise_embedding)
            self.layernorm_n_seq = LayerNorm(c_noise_embedding)
            self.linear_no_bias_n_seq = LinearNoBias(
                in_features=c_noise_embedding, out_features=self.c_s
            )

        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    def forward(
        self,
        t_hat_noise_level_struct: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        t_hat_noise_level_seq: torch.Tensor | None = None,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        reps_have_batch_dim = z_trunk.ndim == 4 and s_trunk.ndim == 3

        relpe = self.relpe(input_feature_dict)
        if reps_have_batch_dim and s_inputs.ndim < s_trunk.ndim:
            assert len(s_trunk) == len(z_trunk)
            s_inputs = s_inputs.unsqueeze(dim=0).repeat(len(s_trunk), 1, 1)
            relpe = relpe.unsqueeze(dim=0).repeat(len(z_trunk), 1, 1, 1)

        # Pair conditioning
        pair_z = torch.cat(
            tensors=[z_trunk, relpe], dim=-1
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))

        fourier_struct = (
            self.fourier_embedding
            if not self.do_fourier_embed_seq
            else self.fourier_embedding_struct
        )
        linear_struct = (
            self.linear_no_bias_n
            if not self.do_fourier_embed_seq
            else self.linear_no_bias_n_struct
        )
        layernorm_struct = (
            self.layernorm_n
            if not self.do_fourier_embed_seq
            else self.layernorm_n_struct
        )

        struct_noise_n = fourier_struct(
            t_hat_noise_level=torch.log(
                input=t_hat_noise_level_struct / self.sigma_data
            )
            / 4
        ).to(
            single_s.dtype
        )  # [..., N_sample, c_in]

        struct_s = linear_struct(layernorm_struct(struct_noise_n)).unsqueeze(
            dim=-2
        )  # [..., N_sample, N_tokens, c_s]

        seq_s = None
        if self.do_fourier_embed_seq and t_hat_noise_level_seq is not None:
            seq_noise_n = self.fourier_embedding_seq(
                t_hat_noise_level=torch.log(
                    input=t_hat_noise_level_seq / self.sigma_data
                )
                / 4
            ).to(
                single_s.dtype
            )  # [..., N_sample, c_in]

            seq_s = self.linear_no_bias_n_seq(
                self.layernorm_n_seq(seq_noise_n)
            ).unsqueeze(
                dim=-2
            )  # [..., N_sample, N_tokens, c_s]

        if not reps_have_batch_dim:
            single_s = single_s.unsqueeze(dim=-3)

        single_s = single_s + struct_s
        if seq_s is not None:
            single_s = single_s + seq_s

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        if not self.training and pair_z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return single_s, pair_z


class DiffusionModule(nn.Module):
    """Implements Algorithm 20 in AF3

    Args:
        sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        c_token (int, optional): feature channel of token (single a). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
        atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
        atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing is performed.
        use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
            only effective if blocks_per_ckpt is not None.
        initialization: initialize the diffusion module according to initialization config.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {"n_blocks": 24, "n_heads": 16},
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        blocks_per_ckpt: int | None = None,
        use_fine_grained_checkpoint: bool = False,
        initialization: dict[str, str | float | bool] | None = None,
        do_fourier_embed_seq: bool = False,
        do_lm_skip_connection: bool = False,
        call_super_init: bool = True,
    ) -> None:
        if call_super_init:
            super().__init__()

        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data,
            c_z=c_z,
            c_s=c_s,
            c_s_inputs=c_s_inputs,
            do_fourier_embed_seq=do_fourier_embed_seq,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token)

        if do_lm_skip_connection:
            self.layernorm_s_skip = LayerNorm(c_s)
            self.linear_no_bias_s_skip = LinearNoBias(in_features=c_s, out_features=c_s)

            self.layernorm_z_skip = LayerNorm(c_z)
            self.linear_no_bias_z_skip = LinearNoBias(in_features=c_z, out_features=c_z)

        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        if initialization is not None:
            self.init_parameters(initialization)

    def init_parameters(self, initialization: dict):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.

        Args:
            initialization (dict): A dictionary containing initialization settings.
        """
        if initialization.get("zero_init_condition_transition", False):
            self.diffusion_conditioning.transition_z1.zero_init()
            self.diffusion_conditioning.transition_z2.zero_init()
            self.diffusion_conditioning.transition_s1.zero_init()
            self.diffusion_conditioning.transition_s2.zero_init()

        self.atom_attention_encoder.linear_init(
            zero_init_atom_encoder_residual_linear=initialization.get(
                "zero_init_atom_encoder_residual_linear", False
            ),
            he_normal_init_atom_encoder_small_mlp=initialization.get(
                "he_normal_init_atom_encoder_small_mlp", False
            ),
            he_normal_init_atom_encoder_output=initialization.get(
                "he_normal_init_atom_encoder_output", False
            ),
        )

        if initialization.get("glorot_init_self_attention", False):
            for (
                block
            ) in (
                self.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks
            ):
                block.attention_pair_bias.glorot_init()

        for block in self.diffusion_transformer.blocks:
            if initialization.get("zero_init_adaln", False):
                block.attention_pair_bias.layernorm_a.zero_init()
                block.conditioned_transition_block.adaln.zero_init()
            if initialization.get("zero_init_residual_condition_transition", False):
                nn.init.zeros_(
                    block.conditioned_transition_block.linear_nobias_b.weight
                )

        if initialization.get("zero_init_atom_decoder_linear", False):
            nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_a.weight)

        if initialization.get("zero_init_dit_output", False):
            nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_out.weight)

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level_struct: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        t_hat_noise_level_seq: torch.Tensor | None = None,
        s_skip: torch.Tensor | None = None,
        z_skip: torch.Tensor | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        encoding_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The raw network to be trained.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level_struct.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        a_token, q_skip, c_skip, p_skip = self.get_all_latents(
            r_noisy,
            t_hat_noise_level_struct,
            input_feature_dict,
            s_inputs,
            s_trunk,
            z_trunk,
            t_hat_noise_level_seq,
            s_skip,
            z_skip,
            inplace_safe,
            chunk_size,
        )

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict,
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
                use_reentrant=False,
            )
        else:
            # Broadcast token activations to atoms and run Sequence-local Atom Attention
            r_update = self.atom_attention_decoder(
                input_feature_dict=input_feature_dict,
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        return r_update

    def get_all_latents(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level_struct: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        t_hat_noise_level_seq: torch.Tensor | None = None,
        s_skip: torch.Tensor | None = None,
        z_skip: torch.Tensor | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        N_sample = r_noisy.size(-3)

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        # Conditioning, shared across difference samples
        # Diffusion_conditioning consumes 7-8G when token num is 768,
        # use checkpoint here if blocks_per_ckpt is not None.
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(
                self.diffusion_conditioning,
                t_hat_noise_level_struct,
                input_feature_dict,
                s_inputs,
                s_trunk,
                z_trunk,
                t_hat_noise_level_seq=t_hat_noise_level_seq,
                inplace_safe=inplace_safe,
                use_reentrant=False,
            )
        else:
            s_single, z_pair = self.diffusion_conditioning(
                t_hat_noise_level_struct=t_hat_noise_level_struct,
                t_hat_noise_level_seq=t_hat_noise_level_seq,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                inplace_safe=inplace_safe,
            )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Expand embeddings to match N_sample
        reps_have_batch_dim = z_trunk.ndim == 4 and s_trunk.ndim == 3
        if not reps_have_batch_dim:
            s_trunk = expand_at_dim(
                s_trunk, dim=-3, n=N_sample
            )  # [..., N_sample, N_token, c_s]
            z_pair = expand_at_dim(
                z_pair, dim=-4, n=N_sample
            )  # [..., N_sample, N_token, N_token, c_z]

        if s_skip is not None and z_skip is not None:
            s_trunk = s_trunk + self.linear_no_bias_s_skip(
                self.layernorm_s_skip(s_skip)
            )
            z_pair = z_pair + self.linear_no_bias_z_skip(self.layernorm_z_skip(z_skip))

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                self.atom_attention_encoder,
                input_feature_dict,
                r_noisy,
                s_trunk,
                z_pair,
                inplace_safe,
                chunk_size,
                use_reentrant=False,
            )
        else:
            # Sequence-local Atom Attention and aggregation to coarse-grained tokens
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                input_feature_dict=input_feature_dict,
                r_l=r_noisy,
                s=s_trunk,
                z=z_pair,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        # Full self-attention on token level.
        if inplace_safe:
            a_token += self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        else:
            a_token = a_token + self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single,
            z=z_pair,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        a_token = self.layernorm_a(a_token)

        return a_token, q_skip, c_skip, p_skip

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level_struct: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_skip: torch.Tensor | None = None,
        z_skip: torch.Tensor | None = None,
        t_hat_noise_level_seq: torch.Tensor | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level_struct**2)[
                ..., None, None
            ]
        )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level_struct=t_hat_noise_level_struct,
            t_hat_noise_level_seq=t_hat_noise_level_seq,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_skip=s_skip,
            z_skip=z_skip,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        s_ratio = (t_hat_noise_level_struct / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level_struct[..., None, None]
            / torch.sqrt(1 + s_ratio**2)
            * r_update
        ).to(r_update.dtype)

        return x_denoised


class JointDiffusionModule(DiffusionModule):
    """Joint diffusion module combining structure and sequence denoising.

    Extends DiffusionModule with a separate sequence decoder head
    (atom_attention_decoder_seq) to jointly denoise atomic coordinates
    and predict sequence logits in a single forward pass.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {"n_blocks": 24, "n_heads": 16},
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        blocks_per_ckpt: int | None = None,
        use_fine_grained_checkpoint: bool = False,
        do_fourier_embed_seq: bool = False,
        do_lm_skip_connection: bool = False,
        initialization: dict[str, str | float | bool] | None = None,
    ) -> None:
        nn.Module.__init__(self)

        atom_decoder_seq_conf = copy.deepcopy(atom_decoder)

        with open_dict(atom_decoder_seq_conf):
            atom_decoder_seq_conf.is_seq_decoder = True

        self.atom_attention_decoder_seq = AtomAttentionDecoder(
            **atom_decoder_seq_conf,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        super().__init__(
            sigma_data=sigma_data,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            atom_encoder=atom_encoder,
            transformer=transformer,
            atom_decoder=atom_decoder,
            blocks_per_ckpt=blocks_per_ckpt,
            use_fine_grained_checkpoint=use_fine_grained_checkpoint,
            initialization=initialization,
            do_fourier_embed_seq=do_fourier_embed_seq,
            do_lm_skip_connection=do_lm_skip_connection,
            call_super_init=False,
        )

    def init_parameters(self, initialization: dict):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.

        Args:
            initialization (dict): A dictionary containing initialization settings.
        """
        super().init_parameters(initialization)

        if initialization.get("zero_init_atom_decoder_linear", False):
            nn.init.zeros_(self.atom_attention_decoder_seq.linear_no_bias_a.weight)

        if initialization.get("zero_init_dit_output", False):
            nn.init.zeros_(self.atom_attention_decoder_seq.linear_no_bias_out.weight)

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level_struct: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        t_hat_noise_level_seq: torch.Tensor | None = None,
        s_skip: torch.Tensor | None = None,
        z_skip: torch.Tensor | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        compute_seq_logits: bool = True,
        encoding_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The raw network to be trained.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            compute_seq_logits (bool): Whether to compute sequence logits

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        a_token, q_skip, c_skip, p_skip = self.get_all_latents(
            r_noisy,
            t_hat_noise_level_struct,
            input_feature_dict,
            s_inputs,
            s_trunk,
            z_trunk,
            t_hat_noise_level_seq,
            s_skip,
            z_skip,
            inplace_safe,
            chunk_size,
        )

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        # Also, pass in directly the sequence encoding for structure prediction
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict,
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
                encoding_dict=encoding_dict["seq"]
                if encoding_dict is not None
                else None,
                use_reentrant=False,
            )
        else:
            # Broadcast token activations to atoms and run Sequence-local Atom Attention
            r_update = self.atom_attention_decoder(
                input_feature_dict=input_feature_dict,
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                encoding_dict=encoding_dict["seq"]
                if encoding_dict is not None
                else None,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        seq_logits = None
        if compute_seq_logits:
            if blocks_per_ckpt and self.use_fine_grained_checkpoint:
                checkpoint_fn = get_checkpoint_fn()
                seq_logits = checkpoint_fn(
                    self.atom_attention_decoder_seq,
                    input_feature_dict,
                    a_token,
                    q_skip,
                    c_skip,
                    p_skip,
                    inplace_safe,
                    chunk_size,
                    encoding_dict=encoding_dict["struct"]
                    if encoding_dict is not None
                    else None,
                    use_reentrant=False,
                )
            else:
                # Broadcast token activations to atoms and run Sequence-local Atom Attention
                seq_logits = self.atom_attention_decoder_seq(
                    input_feature_dict=input_feature_dict,
                    a=a_token,
                    q_skip=q_skip,
                    c_skip=c_skip,
                    p_skip=p_skip,
                    encoding_dict=encoding_dict["struct"]
                    if encoding_dict is not None
                    else None,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return r_update, seq_logits

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level_struct: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_skip: torch.Tensor | None = None,
        z_skip: torch.Tensor | None = None,
        t_hat_noise_level_seq: torch.Tensor | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        encoding_dict: dict[str, torch.Tensor] | None = None,
        compute_seq_logits: bool | None = True,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level_struct**2)[
                ..., None, None
            ]
        )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update, seq_logits = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level_struct=t_hat_noise_level_struct,
            t_hat_noise_level_seq=t_hat_noise_level_seq,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_skip=s_skip,
            z_skip=z_skip,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            compute_seq_logits=compute_seq_logits,
            encoding_dict=encoding_dict,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        s_ratio = (t_hat_noise_level_struct / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level_struct[..., None, None]
            / torch.sqrt(1 + s_ratio**2)
            * r_update
        ).to(r_update.dtype)

        return (x_denoised, seq_logits) if seq_logits is not None else x_denoised
