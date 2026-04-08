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

from dataclasses import dataclass

import biotite.structure
import numpy as np

from disco.data.ccd import (
    MASK_REF_ATOM_MAP,
    MASK_REF_CHARGE,
    MASK_REF_MASK,
    MASK_REF_POS,
)
from disco.data.constants import MASK_RESNAME
from disco.data.parser import AddAtomArrayAnnot
from disco.data.tokenizer import AtomArrayTokenizer, TokenArray
from disco.utils.geometry import random_transform
from disco.utils.logger import get_logger

logger = get_logger(__name__)

_N_BACKBONE_ATOMS = 4


@dataclass
class MaskingResult:
    """Dataclass containing the results of a masking operation.

    Attributes:
        atom_array: Biotite AtomArray after masking (side chains removed,
            ref features updated).
        token_array: TokenArray re-tokenized from the masked atom array.
        masked_res_indices: Array of atom indices that were masked.
    """

    atom_array: biotite.structure.AtomArray
    token_array: TokenArray
    masked_res_indices: np.ndarray


class TaskManager:
    """
    TaskManager class is responsible for managing tasks. - Data transformer (post loading)

    * folding (input: sequence, mask structure)
    * inverse_folding (input: masked sequence, structure)
    * cogen (input: masked sequence, masked structure)

    * masking type (also percentage of function):
        * diffuse random masking (UNK for sequence, Gaussian noise for structure)
        * motif masking
            * proximity
            * chain-based
        * generation based (mask)

    Input:
    sequence, structure, task, mask_type, percentage_mask, schedule_masking

    output: sequence mask index

    """

    def __init__(
        self,
        transform_masked_ref_pos=True,
        ref_pos_augment=True,
    ):
        self.structure = [
            "label_dict",
            "coordinate",
        ]  # n x 3 (all atoms), n is the same for
        self.type_of_seq = [
            "input_feature_dict",
            ["is_rna", "is_dna", "is_prot", "is_lig"],
        ]
        self.backbone_atoms = ["CA", "C", "N", "O"]
        self.na_backbone_atoms = ["P", "C4'", "C1'"]

        self._transform_masked_ref_pos = transform_masked_ref_pos
        self._ref_pos_augment = ref_pos_augment

    def mask(self, atom_array):
        """mask atom array, then return it"""
        self.preprocess_atom_array(atom_array)

        masked_res_indices = np.flatnonzero(atom_array.cano_seq_resname == "MSK")

        logger.debug(
            f"length of masked_res_indices: {len(masked_res_indices)} total: {len(self.sym_uid_hashes)}"
        )

        atom_array = self.mask_ref_information(atom_array, masked_res_indices)
        atom_array = self.mask_side_chains(atom_array, masked_res_indices)

        return MaskingResult(
            atom_array=atom_array,
            token_array=AtomArrayTokenizer(atom_array=atom_array).get_token_array(),
            masked_res_indices=masked_res_indices,
        )

    def preprocess_atom_array(self, atom_array):
        """Extracts and caches atom array properties needed for masking.

        Stores residue IDs, entity/molecule IDs, protein and RNA masks, and
        unique symmetry-UID hashes on ``self`` for use by downstream masking
        methods.

        Args:
            atom_array: Biotite AtomArray with annotations including
                ``ref_space_uid``, ``entity_mol_id``, ``res_id``,
                ``is_protein``, and ``is_rna``.
        """
        self.ref_space_uid = atom_array.ref_space_uid
        self.entity_mol_id = atom_array.entity_mol_id
        self.res_id = atom_array.res_id
        # Stupid hash for unique values
        self.sym_uid_hashes = self.entity_mol_id * 10000 + self.res_id
        self.is_protein_mask = atom_array.is_protein.astype("bool")
        self.is_rna_mask = atom_array.is_rna.astype("bool")
        self.is_protein_indices = np.where(self.is_protein_mask)[0]
        self.is_rna_indices = np.where(self.is_rna_mask)[0]
        self.is_molecule_indices = self.is_protein_indices
        self.sym_protein_uid_hashes = self.sym_uid_hashes[self.is_protein_mask]
        self.sym_uuid_hashes = np.unique(self.sym_protein_uid_hashes)

    def mask_ref_information(
        self,
        atom_array,
        masked_res_indices,
        backbone_atoms=None,
    ):
        """
        Various ref information is used in the atom attention encoder when
        computing s_init. As such, here we mask the ref information so it
        cannot be used by the model downstream to memorize sequence information.

        Args:
            atom_array (dict): The atom array.

        Returns:
            dict: A new bioassembly dictionary with ref information of masked backbone
                  atoms changed to that of the UNK residue.

        """
        backbone_atoms = self.backbone_atoms
        backbone_atom_indices = np.flatnonzero(
            np.isin(atom_array.atom_name, backbone_atoms)
        )
        masked_backbone_indices = np.intersect1d(
            backbone_atom_indices, masked_res_indices
        )

        mask_idx = np.isin(np.arange(len(atom_array)), masked_backbone_indices)

        # This can happen if we're doing cogen and t is close to 0
        if not mask_idx.any():
            return atom_array

        masked_atom_array = atom_array[mask_idx]
        ref_pos, ref_mask, ref_charge = [], [], []
        for atom in atom_array[mask_idx]:
            atom_sub_idx = MASK_REF_ATOM_MAP[atom.atom_name]
            ref_pos.append(MASK_REF_POS[atom_sub_idx])
            ref_mask.append(MASK_REF_MASK[atom_sub_idx])
            ref_charge.append(MASK_REF_CHARGE[atom_sub_idx])

        ref_pos = np.array(ref_pos)
        if self._transform_masked_ref_pos:
            trfmd_ref_pos = []
            for ref_space_uid in np.unique(masked_atom_array.ref_space_uid):
                trfmd_ref_pos.append(
                    random_transform(
                        ref_pos[masked_atom_array.ref_space_uid == ref_space_uid],
                        apply_augmentation=self._ref_pos_augment,
                        centralize=True,
                    )
                )

            ref_pos = np.concatenate(trfmd_ref_pos, axis=0)

        ref_pos_all = atom_array.ref_pos
        ref_charge_all = atom_array.ref_charge
        ref_mask_all = atom_array.ref_mask

        ref_pos_all[mask_idx] = ref_pos
        ref_charge_all[mask_idx] = np.array(ref_charge).astype(int)
        ref_mask_all[mask_idx] = np.array(ref_mask).astype(int)

        atom_array.set_annotation("ref_pos", ref_pos_all)
        atom_array.set_annotation("ref_charge", ref_charge_all)
        atom_array.set_annotation("ref_mask", ref_mask_all)

        return atom_array

    def mask_side_chains(
        self,
        atom_array,
        masked_res_indices,
        backbone_atoms=None,
    ):
        """
        Apply a placeholder mask to the bioassembly dictionary.

        Args:
            atom_array (biotite.AtomArray): The atom array.
            mask (np.ndarray): Boolean mask for the atom_array (True for atoms to keep, False to mask).

        Returns:
            dict: A new bioassembly dictionary with placeholders applied.
        """
        backbone_atoms = self.backbone_atoms
        non_backbone_atom_indices = np.where(
            ~np.isin(atom_array.atom_name, backbone_atoms)
        )[0]
        masked_side_chain_indices = np.intersect1d(
            non_backbone_atom_indices, masked_res_indices
        )
        # non_backbone_atom_indices may also include non protein non backbone atoms
        masked_side_chain_non_protein_indices = np.where(
            ~np.isin(masked_side_chain_indices, self.is_protein_indices)
        )[0]
        side_chain_mask = np.ones(len(atom_array), dtype=bool)
        side_chain_mask[masked_side_chain_indices] = False

        # Mask the atom array. Set only the cano resname as its used in the data
        # leave the resname as the true resname
        atom_array.set_annotation(
            "clean_cano_seq_resname", np.copy(atom_array.cano_seq_resname)
        )
        atom_array.cano_seq_resname[masked_res_indices] = MASK_RESNAME
        atom_array = atom_array[side_chain_mask]  # Mask the atom array
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask_taskmanager(
            atom_array
        )
        return atom_array
