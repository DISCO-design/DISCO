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

import torch

from LigandMPNN.data_utils import featurize
from LigandMPNN.model_utils import ProteinMPNN as ProteinMPNN_Ligand
from openfold.model.primitives import LayerNorm

from disco.data.constants import (
    ATOMIC_NUM_TO_ELE_NAME,
    LMPNN_ELEMENT_DICT,
    OUR_ELE_TO_LMPNN_MAP,
    OUR_RES_IDX_TO_LMPNN_RES_IDX,
)

LMPNN_DEFAULT_DIR = os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__))),
    "../../../packages/LigandMPNN",
)

_ENCODE_VMAP_IN_DIMS = (
    0,  # Y
    0,  # Y_m
    0,  # Y_t
    0,  # X
    0,  # S
    0,  # mask
    None,  # R_idx
    None,  # chain_labels
    None,  # chain_mask
    0,  # structure_noise_level
    None,  # number_of_ligand_atoms
    None,  # model
)

_SAMPLE_VMAP_IN_DIMS = (*_ENCODE_VMAP_IN_DIMS, None)


def call_encode(
    Y,
    Y_m,
    Y_t,
    X,
    S,
    mask,
    R_idx,
    chain_labels,
    chain_mask,
    structure_noise_level,
    number_of_ligand_atoms,
    model,
):
    pre_dict = {
        "Y": Y,
        "Y_m": Y_m,
        "Y_t": Y_t,
        "X": X,
        "S": S,
        "mask": mask,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
        "chain_mask": chain_mask,
    }

    model_inputs = featurize(
        pre_dict,
        number_of_ligand_atoms=number_of_ligand_atoms,
        model_type="ligand_mpnn",
        is_cogen_train_loop=True,
    )

    model_inputs["structure_noise_level"] = structure_noise_level

    node_embeddings, edge_embeddings, edge_idx = model.encode(model_inputs)

    return node_embeddings.squeeze(dim=0)


def call_sample(
    Y,
    Y_m,
    Y_t,
    X,
    S,
    mask,
    R_idx,
    chain_labels,
    chain_mask,
    number_of_ligand_atoms,
    model,
    other_args_dict,
):
    pre_dict = {
        "Y": Y,
        "Y_m": Y_m,
        "Y_t": Y_t,
        "X": X,
        "S": S,
        "mask": mask,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
        "chain_mask": chain_mask,
    }

    model_inputs = featurize(
        pre_dict,
        number_of_ligand_atoms=number_of_ligand_atoms,
        model_type="ligand_mpnn",
        is_cogen_train_loop=True,
    )

    model_inputs.update(**other_args_dict)

    outs = model.sample(model_inputs)
    return outs["log_probs"]


class LigandMPNN(torch.nn.Module):
    """LigandMPNN inverse folding model, adapted from LigandMPNN Github repository
    (https://github.com/dauparas/LigandMPNN).

    Args:
        rootdir:
                Root directory for ProteinMPNN package. Default to 'packages/ProteinMPNN'.
        num_samples:
                Number of samples to be drawn from the sequence distribution. Default to 8.
        ca_only:
                Whether to use only Ca atom coordinates. Default to False.
    """

    def __init__(
        self,
        rootdir=LMPNN_DEFAULT_DIR,
        device="cpu",
        k_neighbors=48,
        ca_only=False,
        arg_pack_side_chains=False,
        set_eval=True,
        c_s=384,
        batch_size=4,
        use_v1_arch=False,
        embed_diffusion_time=False,
        num_edges=32,
        atom_context_num=25,
    ):
        super().__init__()

        self.model = ProteinMPNN_Ligand(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0,
            k_neighbors=num_edges,
            model_type="ligand_mpnn",
            ligand_mpnn_use_side_chain_context=False,
            embed_diffusion_time=embed_diffusion_time,
        )
        self.model.device = device
        self.device = device
        self.batch_size = batch_size
        self.embed_diffusion_time = embed_diffusion_time

        if set_eval:
            self.model.eval()

        self.ca_only = ca_only

        self.atom_context_num = atom_context_num

        dict_len = len(OUR_RES_IDX_TO_LMPNN_RES_IDX)
        self.lmpnn_logit_idx_to_ours = torch.tensor(
            [OUR_RES_IDX_TO_LMPNN_RES_IDX[i] for i in range(dict_len)],
            device=self.device,
            dtype=torch.long,
        )

        lmpnn_hidden_dim = self.model.encoder_layers[-1].dense.W_out.out_features

        self.final_layer_norm = LayerNorm(lmpnn_hidden_dim)
        self.final_dense = torch.nn.Linear(lmpnn_hidden_dim, c_s)

        # Turn off grads for the decoder layers and W_out so we don't
        # get unused parameter errors when running backward pass
        #
        # The last encoder layer's layers which only operate on the edge embeddings
        # we also need to turn off as we eventually only return the node embeddings
        # from the overall model
        last_enc_layer = self.model.encoder_layers[-1]
        modules_to_switch_off = [
            self.model.W_s,
            self.model.W_out,
            last_enc_layer.W11,
            last_enc_layer.W12,
            last_enc_layer.W13,
            last_enc_layer.norm3,
            *self.model.decoder_layers,
        ]

        for module in modules_to_switch_off:
            for param in module.parameters():
                param.requires_grad = False

    def _to_lmpnn_format(self, x_noised, input_feature_dict, structure_noise_level):
        """Converts DISCO internal feature format to LigandMPNN input format.

        Extracts backbone atoms, ligand coordinates, and element types from the
        DISCO feature dict and reformats them into the dictionary structure
        expected by LigandMPNN's featurize/encode functions.

        Args:
            x_noised (torch.Tensor): Noised atomic coordinates.
                [N_sample, N_atom, 3]
            input_feature_dict (dict): DISCO input feature dictionary containing
                keys like 'is_protein', 'ref_element', 'backbone_atom_mask', etc.
            structure_noise_level (torch.Tensor): Noise level for structure diffusion.

        Returns:
            dict: LigandMPNN-formatted input dictionary with keys 'Y', 'Y_m',
                'Y_t', 'X', 'S', 'mask', 'R_idx', 'chain_labels', 'chain_mask',
                and 'structure_noise_level'.
        """
        is_protein = input_feature_dict["is_protein"].bool()
        atomic_ids = input_feature_dict["ref_element"][~is_protein].argmax(dim=1)

        # LMPNN expects things the BB order to be N, CA, C, O
        bb_atom_mask = input_feature_dict["backbone_atom_mask"]
        if bb_atom_mask.ndim == 1:
            bb_atoms = x_noised[:, bb_atom_mask]
        else:
            n_bb_atoms_by_sample = bb_atom_mask.sum(dim=1)
            assert (n_bb_atoms_by_sample[0] == n_bb_atoms_by_sample).all()

            bb_atoms = x_noised[bb_atom_mask].reshape(
                x_noised.shape[0], n_bb_atoms_by_sample[0], x_noised.shape[2]
            )

        bb_atoms = bb_atoms.reshape(bb_atoms.shape[0], bb_atoms.shape[1] // 4, 4, 3)

        to_lmpnn_ele = lambda x: LMPNN_ELEMENT_DICT[
            OUR_ELE_TO_LMPNN_MAP[ATOMIC_NUM_TO_ELE_NAME[x.item()]]
        ]

        lmpnn_eles = torch.tensor(
            list(map(to_lmpnn_ele, atomic_ids)),
            device=x_noised.device,
            dtype=torch.int32,
        )

        lmpnn_eles = lmpnn_eles.unsqueeze(0).repeat(len(bb_atoms), 1)

        have_non_prot_atoms = (~is_protein).any()
        Y, Y_m, Y_t = None, None, None
        if have_non_prot_atoms:
            if is_protein.ndim > 1:
                Y = x_noised[~is_protein].reshape(len(is_protein), -1, 3)
            else:
                Y = x_noised[:, ~is_protein]

            Y_m = torch.ones_like(Y)[..., 0]
            Y_t = lmpnn_eles
        else:
            Y = torch.zeros(
                (len(x_noised), 1, 3), device=x_noised.device, dtype=torch.float
            )
            Y_m = torch.zeros(
                (len(x_noised), 1), device=x_noised.device, dtype=torch.int32
            )
            Y_t = torch.zeros(
                (len(x_noised), 1), device=x_noised.device, dtype=torch.int32
            )

        return {
            "Y": Y,
            "Y_m": Y_m,
            "Y_t": Y_t,
            "X": bb_atoms,
            "S": torch.ones(bb_atoms.shape[:2], device=bb_atoms.device),
            "mask": torch.ones(bb_atoms.shape[:2], device=bb_atoms.device),
            "R_idx": input_feature_dict["backbone_residue_index"],
            "chain_labels": input_feature_dict["backbone_chain_label"],
            "chain_mask": torch.ones_like(input_feature_dict["backbone_chain_label"]),
            "structure_noise_level": structure_noise_level,
        }

    def forward(
        self,
        x_noised,
        input_feature_dict,
        return_pre_dense_embeddings=False,
        structure_noise_level=None,
    ):
        """Forward pass encoding protein/ligand structures via LigandMPNN.

        Converts inputs to LigandMPNN format, runs vmapped encoding over
        samples, and projects node embeddings to the target dimension.

        Args:
            x_noised (torch.Tensor): Noised atomic coordinates.
                [N_sample, N_atom, 3]
            input_feature_dict (dict): DISCO input feature dictionary.
            return_pre_dense_embeddings (bool): If True, also returns the raw
                LigandMPNN node embeddings before the final projection.
                Defaults to False.
            structure_noise_level (torch.Tensor): Noise level for conditioning.

        Returns:
            torch.Tensor: Compressed node embeddings of shape [N_sample, N_token, c_s].
                If return_pre_dense_embeddings is True, returns a tuple of
                (raw_embeddings, compressed_embeddings).
        """
        lmpnn_input_dict = self._to_lmpnn_format(
            x_noised, input_feature_dict, structure_noise_level
        )

        vmap_fxn = torch.vmap(
            call_encode, in_dims=_ENCODE_VMAP_IN_DIMS, chunk_size=self.batch_size
        )

        if len(lmpnn_input_dict["structure_noise_level"]) != len(lmpnn_input_dict["Y"]):
            lmpnn_input_dict["structure_noise_level"] = lmpnn_input_dict[
                "structure_noise_level"
            ].repeat(len(lmpnn_input_dict["Y"]))

        node_embeddings = vmap_fxn(
            lmpnn_input_dict["Y"],
            lmpnn_input_dict["Y_m"],
            lmpnn_input_dict["Y_t"],
            lmpnn_input_dict["X"],
            lmpnn_input_dict["S"],
            lmpnn_input_dict["mask"],
            lmpnn_input_dict["R_idx"],
            lmpnn_input_dict["chain_labels"],
            lmpnn_input_dict["chain_mask"],
            lmpnn_input_dict["structure_noise_level"],
            self.atom_context_num,
            self.model,
        )

        compressed_embeddings = self.final_dense(self.final_layer_norm(node_embeddings))

        if return_pre_dense_embeddings:
            return node_embeddings, compressed_embeddings
        else:
            return compressed_embeddings
