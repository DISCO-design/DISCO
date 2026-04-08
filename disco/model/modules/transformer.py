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

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.model.primitives import LayerNorm
from openfold.utils.checkpointing import checkpoint_blocks

from disco.data.constants import STD_RESIDUES
from disco.model.modules.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    broadcast_token_to_local_atom_pair,
    LinearNoBias,
    rearrange_qk_to_dense_trunk,
)
from disco.model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    permute_final_dims,
)


class AttentionPairBias(nn.Module):
    """Implements Algorithm 24 in AF3

    Args:
        has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
        n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
        c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
    """

    def __init__(
        self,
        has_s: bool = True,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
    ) -> None:
        super().__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            # Line 13
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )
        else:
            self.layernorm_a = LayerNorm(c_a)
        # Line 6-11
        self.local_attention_method = "local_cross_attention"
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
        )
        self.layernorm_z = LayerNorm(c_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

    def glorot_init(self):
        """Initializes attention linear layers (Q, K, V) with Xavier uniform and zeros Q bias."""
        nn.init.xavier_uniform_(self.attention.linear_q.weight)
        nn.init.xavier_uniform_(self.attention.linear_k.weight)
        nn.init.xavier_uniform_(self.attention.linear_v.weight)
        nn.init.zeros_(self.attention.linear_q.bias)

    def local_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            a (torch.Tensor): atom embedding
                [..., N_atom, c_a]
            s (torch.Tensor): atom embedding
                [..., N_atom, c_s]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(a.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        a = self.attention(
            q_x=a,
            kv_x=a,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return a

    def standard_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Performs standard multihead attention with pair bias on full token sequences.

        Used by Algorithm 7/20. Computes attention bias from pair embeddings and
        applies multihead attention with gating.

        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """

        # Multi-head attention bias
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [2, 0, 1])  # [..., n_heads, N_token, N_token]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)

        return a

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: int | None = None,
        n_keys: int | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                s,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(a, s, z, inplace_safe=inplace_safe)

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a


class DiffusionTransformerBlock(nn.Module):
    """Implements Algorithm 23[Line2-Line3] in AF3

    Args:
        c_a (int, optional): single embedding dimension.
        c_s (int, optional): single embedding dimension.
        c_z (int, optional): pair embedding dimension.
        n_heads (int, optional): number of heads for DiffusionTransformerBlock.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_heads: int,  # could be 16 or 4 or ... in AF3
        biasinit: float = -2.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.attention_pair_bias = AttentionPairBias(
            has_s=True, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z, biasinit=biasinit
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: int | None = None,
        n_keys: int | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z] or [..., n_block, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the output of DiffusionTransformerBlock
                [..., N, c_a]
        """
        attn_out = self.attention_pair_bias(
            a=a,
            s=s,
            z=z,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.conditioned_transition_block(a=attn_out, s=s)
        out_a = ff_out + attn_out
        # Avoid s/z to be deleted by torch.utils.checkpoint
        return out_a, s, z


class DiffusionTransformer(nn.Module):
    """Implements Algorithm 23 in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_blocks (int): number of blocks in DiffusionTransformer.
        n_heads (int): number of heads in attention.
        blocks_per_ckpt: number of DiffusionTransformer blocks in each activation checkpoint
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_blocks: int,  # could be 3 or 24 in AF3
        n_heads: int,  # could be 16 or 4 or ... in AF3
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        n_queries: int | None = None,
        n_keys: int | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        clear_cache_between_blocks: bool = False,
    ):
        """Prepares blocks as partials with shared kwargs for checkpointed execution."""
        blocks = [
            partial(
                b,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: int | None = None,
        n_keys: int | None = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.

        Returns:
            torch.Tensor: the output of DiffusionTransformer
                [..., N, c_a]
        """
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt
        )
        del s, z
        return a


class AtomTransformer(nn.Module):
    """Implements Algorithm 7 in AF3.
    Performs local transformer among atom embeddings, with bias predicted from atom pair embeddings.

    Args:
        c_atom int: embedding dim for atom feature. Defaults to 128.
        c_atompair int: embedding dim for atompair feature. Defaults to 16.
        n_blocks (int, optional): number of block in AtomTransformer. Defaults to 3.
        n_heads (int, optional): nubmer of heads in attention. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt: number of AtomTransformer/DiffusionTransformer blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p (torch.Tensor): atompair embedding in dense block shape.
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the output of AtomTransformer
                [..., N_atom, c_atom]
        """
        n_blocks, n_queries, n_keys = p.shape[-4:-1]

        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )


class ConditionedTransitionBlock(nn.Module):
    """Implements Algorithm 25 in AF3

    Args:
        c_a (int, optional): single embedding dim (single feature aggregated atom info).
        c_s (int, optional):  single embedding dim.
        n (int, optional): channel scale factor. Defaults to 2.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(in_features=c_a, out_features=n * c_a)
        self.linear_nobias_a2 = LinearNoBias(in_features=c_a, out_features=n * c_a)
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]

        Returns:
            torch.Tensor: the updated a from ConditionedTransitionBlock
                [..., N, c_a]
        """
        a = self.adaln(a, s)
        b = F.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a


class AtomAttentionEncoder(nn.Module):
    """Implements Algorithm 5 in AF3

    Args:
        has_coords (bool): whether the module input will contains coordinates (r_l).
        c_token (int): token embedding dim.
        c_atom (int, optional): atom embedding dim. Defaults to 128.
        c_atompair (int, optional): atompair embedding dim. Defaults to 16.
        c_s (int, optional):  single embedding dim. Defaults to 384.
        c_z (int, optional): pair embedding dim. Defaults to 128.
        n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
        n_heads (int, optionall): number of heads in AtomTransformer. Defaults to 4.
        blocks_per_ckpt: number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,  # 384 or 768
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.local_attention_method = "local_cross_attention"

        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            # Line9
            self.layernorm_s = LayerNorm(self.c_s)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s, out_features=self.c_atom
            )
            # Line10
            self.layernorm_z = LayerNorm(self.c_z)  # memory bottleneck
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z, out_features=self.c_atompair
            )
            # Line11
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom
            )
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def linear_init(
        self,
        zero_init_atom_encoder_residual_linear: bool = False,
        he_normal_init_atom_encoder_small_mlp: bool = False,
        he_normal_init_atom_encoder_output: bool = False,
    ):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.

        Args:
            zero_init_atom_encoder_residual_linear (bool): Whether to zero-initialize the residual linear layers.
            he_normal_init_atom_encoder_small_mlp (bool): Whether to initialize the small MLP layers with He normal initialization.
            he_normal_init_atom_encoder_output (bool): Whether to initialize the output layer with He normal initialization.
        """

        if zero_init_atom_encoder_residual_linear:
            nn.init.zeros_(self.linear_no_bias_invd.weight)
            nn.init.zeros_(self.linear_no_bias_v.weight)
            nn.init.zeros_(self.linear_no_bias_s.weight)
            nn.init.zeros_(self.linear_no_bias_z.weight)
            nn.init.zeros_(self.linear_no_bias_r.weight)
            nn.init.zeros_(self.linear_no_bias_cl.weight)
            nn.init.zeros_(self.linear_no_bias_cm.weight)
        if he_normal_init_atom_encoder_small_mlp:
            for layer in self.small_mlp:
                if not isinstance(layer, torch.nn.modules.activation.ReLU):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        a=0,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
        if he_normal_init_atom_encoder_output:
            nn.init.kaiming_normal_(
                self.linear_no_bias_q.weight, a=0, mode="fan_in", nonlinearity="relu"
            )

    def forward(
        self,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        r_l: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            r_l (torch.Tensor, optional): noisy position.
                [..., N_sample, N_atom, 3] if has_coords else None.
            s (torch.Tensor, optional): single embedding.
                [..., N_sample, N_token, c_s] if has_coords else None.
            z (torch.Tensor, optional): pair embedding
                [..., N_sample, N_token, N_token, c_z] if has_coords else None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., (N_sample), N_token, c_token]
            q_l:
                [..., (N_sample), N_atom, c_atom]
            c_l:
                [..., (N_sample), N_atom, c_atom]
            p_lm:
                [..., (N_sample), N_atom, N_atom, c_atompair]

        """

        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        # Create the atom single conditioning: Embed per-atom meta data
        # [..., N_atom, C_atom]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        N_atom = input_feature_dict["ref_pos"].shape[-2]
        ddv = next(self.small_mlp.parameters()).device
        c_l = self.linear_no_bias_f(
            torch.cat(
                [
                    input_feature_dict[name]
                    .reshape(*batch_shape, N_atom, self.input_feature[name])
                    .to(ddv)
                    for name in self.input_feature
                ],
                dim=-1,
            )
        )

        # Line2-Line4: Embed offsets between atom reference positions

        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[
                input_feature_dict["ref_pos"].to(ddv),
                input_feature_dict["ref_space_uid"].to(ddv),
            ],
            k=[
                input_feature_dict["ref_pos"].to(ddv),
                input_feature_dict["ref_space_uid"].to(ddv),
            ],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=True,
        )

        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
            "mask_trunked"
        ].unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, C_atompair]

        # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
        if inplace_safe:
            p_lm += (
                self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm += self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm

        # Line7: Initialise the atom single representation as the single conditioning
        q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        n_token = None
        if r_l is not None:
            N_sample = r_l.size(-3)

            # Broadcast the single and pair embedding from the trunk
            n_token = s.size(-2)

            c_l_to_add = self.linear_no_bias_s(
                self.layernorm_s(
                    broadcast_token_to_atom(
                        x_token=s, atom_to_token_idx=atom_to_token_idx
                    )
                )
            )  # [..., N_sample, N_atom, c_atom]

            c_l_prepped, p_lm_prepped, q_l_prepped = c_l, p_lm, q_l
            if c_l.ndim != c_l_to_add.ndim:
                c_l_prepped = c_l.unsqueeze(dim=-3)
                p_lm_prepped = p_lm.unsqueeze(dim=-5)
                q_l_prepped = q_l.unsqueeze(dim=-3)

            c_l = c_l_prepped + self.linear_no_bias_s(
                self.layernorm_s(
                    broadcast_token_to_atom(
                        x_token=s, atom_to_token_idx=atom_to_token_idx
                    )
                )
            )  # [..., N_sample, N_atom, c_atom]

            z_local_pairs, _ = broadcast_token_to_local_atom_pair(
                z_token=z,
                atom_to_token_idx=atom_to_token_idx,
                n_queries=self.n_queries,
                n_keys=self.n_keys,
                compute_mask=False,
                broadcast_atom_to_token_idx=self.training,
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_z]
            p_lm = p_lm_prepped + self.linear_no_bias_z(
                self.layernorm_z(z_local_pairs)
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_atompair]

            # Add the noisy positions
            q_l = q_l_prepped + self.linear_no_bias_r(
                r_l
            )  # [..., N_sample, N_atom, c_atom]

        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        if inplace_safe:
            p_lm += self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            p_lm += self.small_mlp(p_lm)
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
                + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            )  # [..., (N_sample), n_blocks, n_queries, n_keys, c_atompair]

            # Run a small MLP on the pair activations
            p_lm = p_lm + self.small_mlp(p_lm)

        # Cross attention transformer
        q_l = self.atom_transformer(
            q_l, c_l, p_lm, chunk_size=chunk_size
        )  # [..., (N_sample), N_atom, c_atom]

        # Aggregate per-atom representation to per-token representation
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx.to(ddv),
            n_token=n_token,
            reduce="mean",
        )  # [..., (N_sample), N_token, c_token]
        if (not self.training) and (a.shape[-2] > 2000 or q_l.shape[-2] > 20000):
            torch.cuda.empty_cache()
        return a, q_l, c_l, p_lm


class AtomAttentionDecoder(nn.Module):
    """Implements Algorithm 6 in AF3

    Args:
        n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
        c_token (int, optional): feature channel of token (single a). Defaults to 384.
        c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
        c_atompair (int, optional): embedding dim for atom pair embedding.
        blocks_per_ckpt: number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        c_s: int = 384,
        blocks_per_ckpt: int | None = None,
        is_seq_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.return_token_dim_tnsr = is_seq_decoder

        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom)

        n_out_ftrs = len(STD_RESIDUES) if is_seq_decoder else 3
        self.linear_no_bias_out = LinearNoBias(
            in_features=c_atom, out_features=n_out_ftrs
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        input_feature_dict: dict[str, torch.Tensor | int | float | dict],
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        encoding_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_token]
            q_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p_skip (torch.Tensor): atompair single embedding
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the updated noisy coordinates or seq decoder logits
                [..., N_atom, 3]
        """
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            self.linear_no_bias_a(
                broadcast_token_to_atom(
                    x_token=a, atom_to_token_idx=input_feature_dict["atom_to_token_idx"]
                )  # [..., N_atom, c_token]
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        if self.return_token_dim_tnsr:
            n_token = input_feature_dict["residue_index"].shape[-1]
            new_shape = list(q.shape)
            new_shape[-2] = n_token

            q_tokens = torch.zeros(new_shape, device=q.device, dtype=q.dtype)

            idx = input_feature_dict["atom_to_token_idx"].unsqueeze(-1)
            while idx.ndim < q_tokens.ndim:
                idx = idx.unsqueeze(0)

            if q_tokens.shape[0] == idx.shape[0]:
                new_shape[0] = 1

            new_shape[-2] = 1
            idx = idx.repeat(*new_shape)

            q = torch.scatter_add(input=q_tokens, dim=-2, src=q, index=idx)

        # Map to positions update
        r = self.linear_no_bias_out(self.layernorm_q(q))

        if self.return_token_dim_tnsr:
            prot_res_mask = input_feature_dict["prot_residue_mask"]
            if r.ndim == 3 and prot_res_mask.ndim == 1:
                r = r[:, prot_res_mask]
            elif r.ndim == 3 and prot_res_mask.ndim == 2:
                # Only same length proteins are allowed
                num_prot_tokens = prot_res_mask.sum(dim=-1)
                assert (num_prot_tokens[0] == num_prot_tokens).all()

                r = r[prot_res_mask].reshape(
                    *prot_res_mask.shape[:-1], num_prot_tokens[0], r.shape[-1]
                )

            else:
                r = r[prot_res_mask]

        return r
