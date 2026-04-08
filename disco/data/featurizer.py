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
from collections import defaultdict

import numpy as np
import torch
from biotite.structure import AtomArray, get_residue_starts

from disco.data.constants import get_all_elems, PRO_STD_RESIDUES_VALS_SET, STD_RESIDUES
from disco.data.tokenizer import TokenArray
from disco.data.utils import get_histag_mask, get_ligand_polymer_bond_mask
from disco.utils.geometry import random_transform

_CA_ATOM_OFFSET = 1
_LMPNN_OCCUPANCY_CUTOFF = 0.8


class Featurizer:
    """Featurizer for converting token and atom arrays into model input features.

    Args:
        cropped_token_array (TokenArray): TokenArray object after cropping
        cropped_atom_array (AtomArray): AtomArray object after cropping
        ref_pos_augment (bool): Boolean indicating whether apply random rotation and translation on ref_pos
        lig_atom_rename (bool): Boolean indicating whether rename atom name for ligand atoms
    """

    def __init__(
        self,
        cropped_token_array: TokenArray,
        cropped_atom_array: AtomArray,
        ref_pos_augment: bool = True,
        lig_atom_rename: bool = False,
        # task_manager: TaskManager = None,
    ) -> None:
        self.cropped_token_array = cropped_token_array

        self.cropped_atom_array = cropped_atom_array
        self.ref_pos_augment = ref_pos_augment
        self.lig_atom_rename = lig_atom_rename

    @staticmethod
    def encoder(encode_def_list: list[str], input_list: list[str]) -> torch.Tensor:
        """
        Encode a list of input values into a binary format using a specified encoding definition list.

        Args:
            encode_def_list (list): A list of encoding definitions.
            input_list (list): A list of input values to be encoded.

        Returns:
            torch.Tensor: A tensor representing the binary encoding of the input values.
        """
        onehot_dict = {}
        num_keys = len(encode_def_list)
        for index, key in enumerate(encode_def_list):
            onehot = [0] * num_keys
            onehot[index] = 1
            onehot_dict[key] = onehot

        onehot_encoded_data = [onehot_dict[item] for item in input_list]
        onehot_tensor = torch.Tensor(onehot_encoded_data)
        return onehot_tensor

    @staticmethod
    def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "restype"
        One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown,
        4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap.
        Ligands represented as “unknown amino acid”.

        Args:
            restype_list (List[str]): A list of residue types.
                                      The residue type of ligand should be "UNK" in the input list.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded residue types
        """

        return Featurizer.encoder(list(STD_RESIDUES.keys()) + ["-"], restype_list)

    @staticmethod
    def elem_onehot_encoded(elem_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_element"
        One-hot encoding of the element atomic number for each atom
        in the reference conformer, up to atomic number 128.

        Args:
            elem_list (List[str]): A list of element symbols.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded elements
        """
        return Featurizer.encoder(get_all_elems(), elem_list)

    @staticmethod
    def ref_atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) − 32, and names are padded to length 4.

        Args:
            atom_name_list (List[str]): A list of atom names.

        Returns:
            torch.Tensor:  A Tensor of character encoded atom names
        """
        onehot_dict = {}
        for index, key in enumerate(range(64)):
            onehot = [0] * 64
            onehot[index] = 1
            onehot_dict[key] = onehot
        # [N_atom, 4, 64]
        mol_encode = []
        for atom_name in atom_names:
            # [4, 64]
            atom_encode = []
            for name_str in atom_name.ljust(4):
                atom_encode.append(onehot_dict[ord(name_str) - 32])
            mol_encode.append(atom_encode)
        onehot_tensor = torch.Tensor(mol_encode)
        return onehot_tensor

    def get_token_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get token features.
        The size of these features is [N_token].

        Returns:
            Dict[str, torch.Tensor]: A dict of token features.
        """
        token_features = {}

        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]

        restype = centre_atoms.cano_seq_resname
        restype_onehot = self.restype_onehot_encoded(restype)
        if hasattr(centre_atoms, "clean_cano_seq_resname"):
            true_restype = centre_atoms.clean_cano_seq_resname
        else:
            true_restype = centre_atoms.res_name
        token_features["true_restype_id"] = true_restype
        token_features["token_index"] = torch.arange(0, len(self.cropped_token_array))
        token_features["residue_index"] = torch.Tensor(
            centre_atoms.res_id.astype(int)
        ).long()
        token_features["asym_id"] = torch.Tensor(centre_atoms.asym_id_int).long()
        token_features["entity_id"] = torch.Tensor(centre_atoms.entity_id_int).long()
        token_features["sym_id"] = torch.Tensor(centre_atoms.sym_id_int).long()
        token_features["restype"] = restype_onehot
        token_features["restype_id"] = restype

        return token_features

    def get_chain_perm_features(self) -> dict[str, torch.Tensor]:
        """
        The chain permutation use "entity_mol_id", "mol_id" and "mol_atom_index"
        instead of the "entity_id", "asym_id" and "residue_index".

        The shape of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: A dict of chain permutation features.
        """

        chain_perm_features = {}
        chain_perm_features["mol_id"] = torch.Tensor(
            self.cropped_atom_array.mol_id
        ).long()
        chain_perm_features["mol_atom_index"] = torch.Tensor(
            self.cropped_atom_array.mol_atom_index
        ).long()
        chain_perm_features["entity_mol_id"] = torch.Tensor(
            self.cropped_atom_array.entity_mol_id
        ).long()
        return chain_perm_features

    def get_renamed_atom_names(self) -> np.ndarray:
        """
        Rename the atom names of ligands to avioid information leakage.

        Returns:
            np.ndarray: A numpy array of renamed atom names.
        """
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        new_atom_names = copy.deepcopy(self.cropped_atom_array.atom_name)
        for start, stop in zip(res_starts[:-1], res_starts[1:], strict=False):
            res_mol_type = self.cropped_atom_array.mol_type[start]
            if res_mol_type != "ligand":
                continue

            elem_count = defaultdict(int)
            new_res_atom_names = []
            for elem in self.cropped_atom_array.element[start:stop]:
                elem_count[elem] += 1
                new_res_atom_names.append(f"{elem.upper()}{elem_count[elem]}")
            new_atom_names[start:stop] = new_res_atom_names
        return new_atom_names

    def get_reference_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get reference features.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of reference features.
        """
        ref_pos = []
        for ref_space_uid in np.unique(self.cropped_atom_array.ref_space_uid):
            res_ref_pos = random_transform(
                self.cropped_atom_array.ref_pos[
                    self.cropped_atom_array.ref_space_uid == ref_space_uid,
                ],
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )
            ref_pos.append(res_ref_pos)
        ref_pos = np.concatenate(ref_pos)

        ref_features = {}
        ref_features["ref_pos"] = torch.Tensor(ref_pos)
        ref_features["ref_mask"] = torch.Tensor(self.cropped_atom_array.ref_mask).long()
        ref_features["ref_element"] = Featurizer.elem_onehot_encoded(
            self.cropped_atom_array.element
        ).long()
        ref_features["ref_charge"] = torch.Tensor(
            self.cropped_atom_array.ref_charge
        ).long()

        if self.lig_atom_rename:
            atom_names = self.get_renamed_atom_names()
        else:
            atom_names = self.cropped_atom_array.atom_name

        ref_features["ref_atom_name_chars"] = Featurizer.ref_atom_name_chars_encoded(
            atom_names
        ).long()
        ref_features["ref_space_uid"] = torch.Tensor(
            self.cropped_atom_array.ref_space_uid
        ).long()
        return ref_features

    def get_bond_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8
        A 2D matrix indicating if there is a bond between any atom in token i and token j,
        restricted to just polymer-ligand and ligand-ligand bonds and bonds less than 2.4 Å during training.
        The size of bond feature is [N_token, N_token].

        Returns:
            Dict[str, torch.Tensor]: A dict of bond features.
        """
        bond_features = {}
        num_tokens = len(self.cropped_token_array)
        adj_matrix = self.cropped_atom_array.bonds.adjacency_matrix().astype(int)

        token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
        atom_bond_mask = adj_matrix > 0

        for i in range(num_tokens):
            atoms_i = self.cropped_token_array[i].atom_indices
            token_i_mol_type = self.cropped_atom_array.mol_type[atoms_i[0]]
            token_i_res_name = self.cropped_atom_array.res_name[atoms_i[0]]
            token_i_ref_space_uid = self.cropped_atom_array.ref_space_uid[atoms_i[0]]
            unstd_res_token_i = (
                token_i_res_name not in STD_RESIDUES and token_i_mol_type != "ligand"
            )
            is_polymer_i = token_i_mol_type in ["protein", "dna", "rna"]

            for j in range(i + 1, num_tokens):
                atoms_j = self.cropped_token_array[j].atom_indices
                token_j_mol_type = self.cropped_atom_array.mol_type[atoms_j[0]]
                token_j_res_name = self.cropped_atom_array.res_name[atoms_j[0]]
                token_j_ref_space_uid = self.cropped_atom_array.ref_space_uid[
                    atoms_j[0]
                ]
                unstd_res_token_j = (
                    token_j_res_name not in STD_RESIDUES
                    and token_j_mol_type != "ligand"
                )
                is_polymer_j = token_j_mol_type in ["protein", "dna", "rna"]

                # The polymer-polymer (std-std, std-unstd, and inter-unstd) bond will not be included in token_bonds.
                if is_polymer_i and is_polymer_j:
                    is_same_res = token_i_ref_space_uid == token_j_ref_space_uid
                    unstd_res_bonds = unstd_res_token_i and unstd_res_token_j
                    if not (is_same_res and unstd_res_bonds):
                        continue

                sub_matrix = atom_bond_mask[np.ix_(atoms_i, atoms_j)]
                if np.any(sub_matrix):
                    token_adj_matrix[i, j] = 1
                    token_adj_matrix[j, i] = 1
        bond_features["token_bonds"] = torch.Tensor(token_adj_matrix)
        return bond_features

    def get_extra_features(self) -> dict[str, torch.Tensor]:
        """
        Get other features not listed in AlphaFold3 SI Chapter 2.8 Table 5.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of extra features.
        """
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        # Ensure the order of the atom_to_token_idx is the same as the atom_array
        atom_to_token_idx = [
            atom_to_token_idx_dict[atom_idx]
            for atom_idx in range(len(self.cropped_atom_array))
        ]

        extra_features = {}
        extra_features["atom_to_token_idx"] = torch.Tensor(atom_to_token_idx).long()
        extra_features["atom_to_tokatom_idx"] = torch.Tensor(
            self.cropped_atom_array.tokatom_idx
        ).long()

        extra_features["is_protein"] = torch.Tensor(
            self.cropped_atom_array.is_protein
        ).long()
        extra_features["is_ligand"] = torch.Tensor(
            self.cropped_atom_array.is_ligand
        ).long()
        extra_features["is_dna"] = torch.Tensor(self.cropped_atom_array.is_dna).long()
        extra_features["is_rna"] = torch.Tensor(self.cropped_atom_array.is_rna).long()
        if "resolution" in self.cropped_atom_array._annot:
            extra_features["resolution"] = torch.Tensor(
                [self.cropped_atom_array.resolution[0]]
            )
        else:
            extra_features["resolution"] = torch.Tensor([-1])
        return extra_features

    def get_mask_features(self) -> dict[str, torch.Tensor]:
        """
        Generate mask features for the cropped atom array.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing various mask features.
        """
        mask_features = {}

        mask_features["pae_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.centre_atom_mask
        ).long()

        mask_features["plddt_m_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.plddt_m_rep_atom_mask
        ).long()  # [N_atom]

        mask_features["distogram_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.distogram_rep_atom_mask
        ).long()  # [N_atom]

        mask_features["modified_res_mask"] = torch.Tensor(
            self.cropped_atom_array.modified_res_mask
        ).long()

        lig_polymer_bonds = get_ligand_polymer_bond_mask(self.cropped_atom_array)
        num_atoms = len(self.cropped_atom_array)
        bond_mask_mat = np.zeros((num_atoms, num_atoms))
        for i, j, _ in lig_polymer_bonds:
            bond_mask_mat[i, j] = 1
            bond_mask_mat[j, i] = 1
        mask_features["bond_mask"] = torch.Tensor(
            bond_mask_mat
        ).long()  # [N_atom, N_atom]

        # The backbone residue index and chain label isn't really a mask feature but it's easier to compute here

        mask_features["backbone_atom_mask"] = torch.zeros(
            len(self.cropped_atom_array), dtype=torch.bool
        )
        mask_features["backbone_no_oxygen_atom_mask"] = torch.zeros(
            len(self.cropped_atom_array), dtype=torch.bool
        )

        mask_features["backbone_residue_index"] = []
        mask_features["backbone_chain_label"] = []
        mask_features["res_occ_cutoff_mask"] = []
        mask_features["res_is_resolved_mask"] = []
        for token in self.cropped_token_array:
            if token.value in PRO_STD_RESIDUES_VALS_SET:
                start_idx = token.atom_indices[0]
                mask_features["backbone_atom_mask"][start_idx : start_idx + 4] = True
                mask_features["backbone_no_oxygen_atom_mask"][
                    start_idx : start_idx + 3
                ] = True

                ca_atom = self.cropped_atom_array[start_idx + _CA_ATOM_OFFSET]
                assert ca_atom.atom_name == "CA"
                mask_features["backbone_residue_index"].append(ca_atom.res_id)
                mask_features["backbone_chain_label"].append(ca_atom.asym_id_int)

                bb_atoms = self.cropped_atom_array[start_idx : start_idx + 4]
                if hasattr(bb_atoms, "occupancy"):
                    occ_cutoff_mask = (
                        bb_atoms.occupancy > _LMPNN_OCCUPANCY_CUTOFF
                    ).all()
                    mask_features["res_occ_cutoff_mask"].append(occ_cutoff_mask)
                else:
                    mask_features["res_occ_cutoff_mask"].append(True)

                if hasattr(bb_atoms, "is_resolved"):
                    mask_features["res_is_resolved_mask"].append(
                        bb_atoms.is_resolved.all()
                    )
                else:
                    mask_features["res_is_resolved_mask"].append(False)

        mask_features["backbone_residue_index"] = torch.tensor(
            mask_features["backbone_residue_index"]
        )
        mask_features["backbone_chain_label"] = torch.tensor(
            mask_features["backbone_chain_label"]
        )
        mask_features["res_occ_cutoff_mask"] = torch.tensor(
            mask_features["res_occ_cutoff_mask"]
        )
        mask_features["res_is_resolved_mask"] = torch.tensor(
            mask_features["res_is_resolved_mask"]
        )
        mask_features["histag_mask"] = get_histag_mask(
            self.cropped_token_array, self.cropped_atom_array
        )

        return mask_features

    def get_all_input_features(self):
        """
        Get input features from cropped data.

        Returns:
            Dict[str, torch.Tensor]: a dict of features.
        """
        features = {}
        token_features = self.get_token_features()
        features.update(token_features)

        bond_features = self.get_bond_features()
        features.update(bond_features)

        reference_features = self.get_reference_features()
        features.update(reference_features)

        extra_features = self.get_extra_features()
        features.update(extra_features)

        chain_perm_features = self.get_chain_perm_features()
        features.update(chain_perm_features)

        mask_features = self.get_mask_features()
        features.update(mask_features)
        return features
