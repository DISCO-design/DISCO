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

import os
from functools import partial, wraps

import numpy as np
import torch
from torch import nn, tensor
from transformers import AutoModelForMaskedLM, AutoTokenizer

from disco.data.constants import MASK_TOKEN_IDX, PRO_STD_RESIDUES
from disco.utils.logger import get_logger
from disco.utils.seq.res_constant import get_residue_constants

fastln_is_installed = os.getenv("LAYERNORM_TYPE", None) == "fast_layernorm"
if fastln_is_installed:
    # LayerNorm is a time bottomneck, so we use a custom implementation.
    from disco.model.layer_norm.layer_norm import FusedLayerNorm as LayerNorm
else:
    from torch.nn import LayerNorm

logger = get_logger(__name__)

# constants
IS_MOLECULE_TYPES = 5
IS_PROTEIN_INDEX = 0
IS_RNA_INDEX = 1
IS_DNA_INDEX = 2
IS_LIGAND_INDEX = -2
IS_METAL_ION_INDEX = -1
IS_NON_PROTEIN_INDICES = slice(1, 5)

IS_PROTEIN, IS_RNA, IS_DNA, IS_LIGAND, IS_METAL_ION = tuple(
    (IS_MOLECULE_TYPES + i if i < 0 else i)
    for i in [
        IS_PROTEIN_INDEX,
        IS_RNA_INDEX,
        IS_DNA_INDEX,
        IS_LIGAND_INDEX,
        IS_METAL_ION_INDEX,
    ]
)


def esm_tokens_to_sequence(aa_ids: torch.Tensor) -> str | list[str]:
    if aa_ids.ndim == 1:
        sequence_data = "".join(
            [(ESM_MASK_TOKEN if i == MASK_TOKEN_IDX else restypes[i]) for i in aa_ids]
        )

    elif aa_ids.ndim == 2:
        sequence_data = [
            "".join(
                [
                    (ESM_MASK_TOKEN if i == MASK_TOKEN_IDX else restypes[i])
                    for i in seq_aas
                ]
            )
            for seq_aas in aa_ids
        ]

    else:
        raise ValueError(f"aa_ids has {aa_ids.ndim} dimensions which is invalid")

    return sequence_data


def remove_plms(fn):
    """Decorator to remove PLMs from the model before calling the inner function and then restore
    them afterwards."""

    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_plms = hasattr(self, "plms")
        if has_plms:
            plms = self.plms
            delattr(self, "plms")

        out = fn(self, *args, **kwargs)

        if has_plms:
            self.plms = plms

        return out

    return inner


# constants

aa_constants = get_residue_constants(res_chem_index=IS_PROTEIN)
restypes = aa_constants.restypes + ["X"]

ESM_MASK_TOKEN = "<mask>"  # nosec


class HFBertModel(torch.nn.Module):
    """HuggingFace BERT-based language model for protein sequence encoding.

    Wraps a pretrained masked language model (e.g., DPLM 650M) to produce
    single and pair representations from protein sequences. Supports
    single-chain protein inputs. Raises an error if multiple protein chains
    are detected, as the base language model was only trained on single-chain
    data.

    Args:
        model_name (str): Key into MODEL_REGISTRY for the pretrained model.
    """

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        model_name = MODEL_REGISTRY[model_name]
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="eager"
        )
        self.batch_converter = AutoTokenizer.from_pretrained(model_name)
        self.register_buffer("dummy", tensor(0), persistent=False)
        self.embed_dim = self.model.config.hidden_size
        self.embed_pair_dim = (
            self.model.config.num_attention_heads * self.model.config.num_hidden_layers
        )
        self.esm_s_combine = nn.Parameter(
            torch.zeros(self.model.config.num_hidden_layers + 1)
        )
        self.use_esm_attn_map = True

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def attn_head(self):
        return self.model.config.num_attention_heads

    @property
    def single_dim(self):
        return self.model.config.hidden_size

    def _validate_single_chain(
        self,
        chain_ids: torch.Tensor,
    ) -> None:
        """Validates that only a single protein chain is present.

        The base language model was only trained on single-chain data and
        predicts multi-chain proteins poorly, so we reject multi-chain inputs.

        Args:
            chain_ids (torch.Tensor): Chain identifiers per residue.
                [..., N_residues]

        Raises:
            ValueError: If more than one protein chain is detected.
        """
        chain_id_changes = chain_ids.diff().nonzero()

        if chain_id_changes.numel() > 0:
            raise ValueError(
                "Multiple protein chains detected. The base language model was "
                "only trained on single-chain data and predicts multi-chain "
                "proteins poorly. Please provide a single protein chain."
            )

    def forward(
        self,
        aa_ids: torch.Tensor,
        chain_ids: torch.Tensor,
        validate_single_chain: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes a protein sequence using the pretrained language model.

        Validates single-chain input, tokenizes, runs the PLM forward pass,
        and extracts single representations (weighted sum of hidden states),
        pair representations (from attention maps), and LM logits.

        Args:
            aa_ids (torch.Tensor): Amino acid token indices.
                [..., N_residues]
            chain_ids (torch.Tensor): Chain identifiers per residue.
                [..., N_residues]
            validate_single_chain (bool): If True, validates that only a single
                protein chain is present. Set to False when the asym_id may
                include non-protein chains (e.g. ligands).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - single_repns: Single residue representations. [B, L, C]
                - pair_repns: Pair representations from attention. [B, L, L, C]
                - lm_logits: Language model logits. [B, L, vocab_size]
        """
        if validate_single_chain:
            self._validate_single_chain(chain_ids)

        sequence_data = esm_tokens_to_sequence(aa_ids)

        batch_tokens = self.batch_converter(sequence_data, return_tensors="pt")
        batch_tokens = {k: v.to(self.dummy.device) for k, v in batch_tokens.items()}

        # forward through plm
        result = self.model(
            **batch_tokens, output_attentions=True, output_hidden_states=True
        )

        # postprocess — strip BOS/EOS tokens added by the tokenizer
        single_repns = torch.stack(result["hidden_states"], dim=2)
        single_repns = single_repns[:, 1:-1]  # B, L, nLayers, C
        single_repns = (
            self.esm_s_combine.softmax(0)[None, None, :] @ single_repns
        ).squeeze(2)
        pair_repns = (
            # B, L, L, nlayers, nheads
            torch.stack(result["attentions"], dim=2)
            .permute(0, 4, 3, 1, 2)
            .flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.use_esm_attn_map
            else None
            # B, L, L, nlayers * nheads
        )

        lm_logits = result["logits"][:, 1:-1]
        if lm_logits.shape[0] == 1:
            lm_logits = lm_logits.squeeze()

        return single_repns, pair_repns, lm_logits


MODEL_REGISTRY = {
    "esm2_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
    "esm2_15B": "facebook/esm2_t48_15B_UR50D",
    "dplm_150M": "airkingbd/dplm_150m",
    "dplm_650M": "airkingbd/dplm_650m",
    "dplm_3B": "airkingbd/dplm_3b",
    "evoflow_650M": "zhangzhi/EvoFlow-650M-context-3070",
    "evoflow_650M_Base": "zhangzhi/EvoFlow-650M",
    "evoflow_650M_calibrated": "zhangzhi/EvoFlow-650M-context-3070-calibration",
}

PLMRegistry = {
    model_name: partial(HFBertModel, model_name) for model_name in MODEL_REGISTRY
}


class LMWrapper(nn.Module):
    """Wrapper to integrate different PLMs into the model.

    Args:
        lm_name: The name of the PLM to use.
        output_dim: The output dimension of pair representations.
    """

    def __init__(
        self,
        lm_name: str,
        output_dim: int,
        single_rep_output_dim: int,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self.lm_name = lm_name
        assert (
            self.lm_name in PLMRegistry
        ), f"LM name {self.lm_name} not found in PLMRegistry"
        self.lm_model = PLMRegistry[self.lm_name]()
        if freeze_lm:
            self.lm_model.requires_grad_(False)
            self.lm_model.eval()
        use_pair_rep = self.lm_model.embed_pair_dim is not None
        self.final_layer = nn.Sequential(
            nn.Linear(self.lm_model.embed_pair_dim, output_dim),
            LayerNorm(output_dim),
        )
        self.final_layer_single_rep = nn.Sequential(
            nn.Linear(self.lm_model.embed_dim, single_rep_output_dim),  # 1280 x 449
            LayerNorm(single_rep_output_dim),
        )
        self.use_pair_rep = use_pair_rep

    def forward(self, input_feature_dict, z, s_inputs):
        """Computes PLM embeddings and integrates them with diffusion embeddings.

        Extracts protein sequence from the feature dict, runs the PLM to obtain
        single and pair representations, and adds them into the existing pair
        embedding matrix (z) and single input embeddings (s_inputs).

        Args:
            input_feature_dict (dict): Input feature dictionary containing
                sequence info ('masked_prot_restype' or 'restype_id'),
                'prot_residue_mask', and 'asym_id'.
            z (torch.Tensor): Pair embedding matrix to augment.
                [..., N_token, N_token, c_z]
            s_inputs (torch.Tensor): Single input embeddings to augment.
                [..., N_token, c_s_inputs]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Updated pair embedding z with PLM pair info added.
                - Updated single input embeddings s_inputs with PLM single info.
                - LM logits from the PLM forward pass.
        """
        # during inference, there is no masked_prot_restype
        if "masked_prot_restype" in input_feature_dict:
            seq = input_feature_dict["masked_prot_restype"]
        elif "true_restype_id" in input_feature_dict:
            seq = torch.tensor(
                [
                    PRO_STD_RESIDUES[x]
                    for x in input_feature_dict["restype_id"]
                    if x in PRO_STD_RESIDUES
                ]
            )
        else:  # no true_restype_id during inference
            seq = torch.tensor(
                [PRO_STD_RESIDUES[x] for x in input_feature_dict["restype_id"]]
            )

        assert (seq >= 0).all(), f"Invalid residue type: {seq}"
        assert (
            (seq < len(PRO_STD_RESIDUES)) | (seq == MASK_TOKEN_IDX)
        ).all(), f"Invalid residue type: {seq}"
        assert len(seq) > 0, f"Zero length sequence: {seq}"

        if "prot_residue_mask" in input_feature_dict:
            prot_residue_mask = input_feature_dict["prot_residue_mask"]
        elif "true_restype_id" in input_feature_dict:
            prot_residue_mask = [
                True if x in PRO_STD_RESIDUES else False
                for x in input_feature_dict["true_restype_id"]
            ]
        else:  # during inference
            prot_residue_mask = np.ones(
                len(input_feature_dict["restype_id"]), dtype=bool
            )

        asym_id = input_feature_dict["asym_id"][prot_residue_mask].reshape(len(seq), -1)
        lm_output = self.lm_model(seq, asym_id)
        if len(lm_output) == 3:
            single_rep, pair_rep, lm_logits = lm_output
        else:
            single_rep, pair_rep = lm_output
            lm_logits = None
            logger.warning("LM logits not available")
        pair = self.final_layer(pair_rep)
        single = self.final_layer_single_rep(single_rep)

        squeeze_cond = z.ndim <= 3 or (
            z.ndim == 4 and single.ndim == 3 and single.shape[0] == 1
        )

        if squeeze_cond:
            pair = pair.squeeze()
            single = single.squeeze()

        to_add = single
        if single.ndim == 3:
            to_add = single.flatten(0, 1)

        s_inputs[prot_residue_mask] = s_inputs[prot_residue_mask] + to_add

        newmat_paired = None
        if prot_residue_mask.ndim == 1:
            newmat_paired = add_matrix_subset(z, prot_residue_mask, pair)
        else:
            paired_mats = []
            for i in range(len(prot_residue_mask)):
                paired_mats.append(
                    add_matrix_subset(z[i], prot_residue_mask[i], pair[i])
                )

            newmat_paired = torch.stack(paired_mats)

        return newmat_paired, s_inputs, lm_logits

    def encode_sequence(
        self,
        protein_sequence: torch.Tensor,
        input_feature_dict: dict,
    ) -> torch.Tensor:
        """Encodes a protein sequence and returns single representations.

        Args:
            protein_sequence (torch.Tensor): Amino acid token indices.
            input_feature_dict (dict): Feature dict containing 'asym_id' for
                chain information.

        Returns:
            torch.Tensor: Single residue representations from the PLM.
        """
        outputs = self.lm_model(
            protein_sequence,
            input_feature_dict["asym_id"],
            validate_single_chain=False,
        )

        single_rep = outputs[0]
        return single_rep


def add_matrix_subset(z, prot_mask, pair):
    """Adds pair embeddings to a protein-masked subset of the pair representation matrix.

    Uses the protein residue mask to index into the pair matrix z and adds
    the PLM-derived pair embeddings at the corresponding positions.

    Args:
        z (torch.Tensor): Full pair representation matrix.
            [..., N_token, N_token, c_z]
        prot_mask (torch.Tensor): Boolean mask indicating protein residues.
            [N_token]
        pair (torch.Tensor): PLM pair embeddings to add.
            [..., N_prot, N_prot, c_z]

    Returns:
        torch.Tensor: Updated pair representation matrix with PLM pair info added.
    """
    assert z.ndim in {3, 4}
    if z.shape[z.ndim - 3] != z.shape[z.ndim - 2]:
        logger.warning(f"z shape not square: {z.shape}")
        return z

    # Ensure pair has the correct shape
    if len(pair.shape) > 3:
        pair = pair.squeeze()

    mask_index = prot_mask.nonzero(as_tuple=True)[0]

    # Create torch equivalents of np.ix_
    row_idx = mask_index.unsqueeze(1)  # shape (k, 1)
    col_idx = mask_index.unsqueeze(0)  # shape (1, k)

    def _inner_subset(inner_z):
        # Add the corresponding pair values to the subset of z
        try:
            if pair.ndim == 1:
                inner_z[row_idx, col_idx] += pair
            else:
                inner_z[row_idx, col_idx] += pair[: len(mask_index), : len(mask_index)]
        except ValueError as e:
            logger.error(
                f"mismatch index len z_indices:{row_idx}, {col_idx}, z:{z.shape} and pair, with mask index {len(mask_index)}"
            )
            logger.error(e)

        return inner_z

    return _inner_subset(z) if z.ndim == 3 else torch.vmap(_inner_subset)(z)
