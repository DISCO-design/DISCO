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
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

from disco.data.constants import STD_RESIDUES

_TOTAL_NUM_AAS = 20
_MIN_HISTAG_LENGTH = 5


def remove_numbers(s: str) -> str:
    """
    Remove numbers from a string.

    Args:
        s (str): input string

    Returns:
        str: a string with numbers removed.
    """
    return re.sub(r"\d+", "", s)


def get_starts_by(
    atom_array: AtomArray, by_annot: str, add_exclusive_stop=False
) -> np.ndarray:
    """get start indices by given annotation in an AtomArray

    Args:
        atom_array (AtomArray): Biotite AtomArray
        by_annot (str): annotation to group by, eg: 'chain_id', 'res_id', 'res_name'
        add_exclusive_stop (bool, optional): add exclusive stop (len(atom_array)). Defaults to False.

    Returns:
        np.ndarray: start indices of each group, shape = (n,), eg: [0, 10, 20, 30, 40]
    """
    annot = getattr(atom_array, by_annot)
    # If annotation change, a new start
    annot_change_mask = annot[1:] != annot[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new residue
    starts = np.where(annot_change_mask)[0] + 1

    # The first start is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], starts, [atom_array.array_length()]))
    else:
        return np.concatenate(([0], starts))


def get_ligand_polymer_bond_mask(
    atom_array: AtomArray, lig_include_ions=False
) -> np.ndarray:
    """
    Ref AlphaFold3 SI Chapter 3.7.1.
    Get bonds between the bonded ligand and its parent chain.

    Args:
        atom_array (AtomArray): biotite atom array object.
        lig_include_ions (bool): whether to include ions in the ligand.

    Returns:
        np.ndarray: bond records between the bonded ligand and its parent chain.
                    e.g. np.array([[atom1, atom2, bond_order]...])
    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(
            atom_array.label_asym_id, return_counts=True
        )
        chain_id_to_count_map = dict(zip(unique_chain_id, counts, strict=False))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    # identify polymer by mol_type (protein, rna, dna, ligand)
    polymer_mask = np.isin(atom_array.mol_type, ["protein", "rna", "dna"])

    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]

    lig_polymer_bond_indices = np.where(
        (lig_mask[idx_i] & polymer_mask[idx_j])
        | (lig_mask[idx_j] & polymer_mask[idx_i])
    )[0]
    if lig_polymer_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds._bonds[
            lig_polymer_bond_indices
        ]  # np.array([[atom1, atom2, bond_order]...])
    return lig_polymer_bonds


def get_clean_data(atom_array: AtomArray) -> AtomArray:
    """
    Removes unresolved atoms from the AtomArray.

    Args:
        atom_array (AtomArray): The input AtomArray containing atoms.

    Returns:
        AtomArray: A new AtomArray with unresolved atoms removed.
    """
    atom_array_wo_unresol = atom_array.copy()
    atom_array_wo_unresol = atom_array[atom_array.is_resolved]
    return atom_array_wo_unresol


def save_atoms_to_cif(
    output_cif_file: str,
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    pdb_id: str,
) -> None:
    """
    Save atom array data to a CIF file.

    Args:
        output_cif_file (str): The output path for saving the atom array in CIF format.
        atom_array (AtomArray): The atom array to be saved.
        entity_poly_type: The entity poly type information.
        pdb_id: The PDB ID for the entry.
    """
    cifwriter = CIFWriter(atom_array, entity_poly_type)
    cifwriter.save_to_cif(
        output_path=output_cif_file,
        entry_id=pdb_id,
        include_bonds=True,
    )


def save_atoms_to_pdb(
    output_pdb_file: str,
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    pdb_id: str,
) -> None:
    """
    Save atom array data to a PDB file.

    Args:
        output_pdb_file (str): The output path for saving the atom array in PDB format.
        atom_array (AtomArray): The atom array to be saved.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
    """
    pdb_file = PDBFile()
    atom_array.chain_id = [a[0] for a in atom_array.label_asym_id]
    pdb_file.set_structure(atom_array)
    pdb_file.write(output_pdb_file)


def save_structure_pdb(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
):
    """Save predicted structure with coordinates to a PDB file.

    Copies the atom array, replaces coordinates with predicted values, and
    writes the result to a PDB file. If ``is_resolved`` is present, also
    saves a version without unresolved atoms.

    Args:
        atom_array (AtomArray): The original AtomArray containing the structure.
        pred_coordinate (torch.Tensor): The predicted coordinates, shape (N_atom, 3).
        output_fpath (str): The output file path for saving the PDB file.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
    """
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    save_atoms_to_pdb(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved"):
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        save_atoms_to_pdb(
            output_fpath.replace(".pdb", "_wounresol.pdb"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


def save_structure_cif(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
):
    """
    Save the predicted structure to a CIF file.

    Args:
        atom_array (AtomArray): The original AtomArray containing the structure.
        pred_coordinate (torch.Tensor): The predicted coordinates for the structure.
        output_fpath (str): The output file path for saving the CIF file.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
    """
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    save_atoms_to_cif(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved"):
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        save_atoms_to_cif(
            output_fpath.replace(".cif", "_wounresol.cif"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


class CIFWriter:
    """Write AtomArray to cif.

    Args:
        atom_array (AtomArray): Biotite AtomArray object.
        entity_poly_type (dict[str, str], optional): A dict of label_entity_id to entity_poly_type. Defaults to None.
                                                     If None, "the entity_poly" and "entity_poly_seq" will not be written to the cif.
    """

    def __init__(self, atom_array: AtomArray, entity_poly_type: dict[str, str] = None):
        self.atom_array = atom_array
        self.entity_poly_type = entity_poly_type

    def _get_entity_poly_and_entity_poly_seq_block(self):
        entity_poly = defaultdict(list)
        for entity_id, entity_type in self.entity_poly_type.items():
            label_asym_ids = np.unique(
                self.atom_array.label_asym_id[
                    self.atom_array.label_entity_id == entity_id
                ]
            )
            label_asym_ids_str = ",".join(label_asym_ids)

            if label_asym_ids_str == "":
                # The entity not in current atom_array
                continue

            entity_poly["entity_id"].append(entity_id)
            entity_poly["pdbx_strand_id"].append(label_asym_ids_str)
            entity_poly["type"].append(entity_type)

        entity_poly_seq = defaultdict(list)
        for entity_id, label_asym_ids_str in zip(
            entity_poly["entity_id"], entity_poly["pdbx_strand_id"], strict=False
        ):
            first_label_asym_id = label_asym_ids_str.split(",")[0]
            first_asym_chain = self.atom_array[
                self.atom_array.label_asym_id == first_label_asym_id
            ]
            chain_starts = struc.get_chain_starts(
                first_asym_chain, add_exclusive_stop=True
            )
            asym_chain = first_asym_chain[
                chain_starts[0] : chain_starts[1]
            ]  # ensure the asym chain is a single chain

            res_starts = struc.get_residue_starts(asym_chain, add_exclusive_stop=False)
            asym_chain_entity_id = asym_chain[res_starts].label_entity_id.tolist()
            asym_chain_hetero = [
                "n" if not i else "y" for i in asym_chain[res_starts].hetero
            ]
            asym_chain_res_name = asym_chain[res_starts].res_name.tolist()
            asym_chain_res_id = asym_chain[res_starts].res_id.tolist()

            entity_poly_seq["entity_id"].extend(asym_chain_entity_id)
            entity_poly_seq["hetero"].extend(asym_chain_hetero)
            entity_poly_seq["mon_id"].extend(asym_chain_res_name)
            entity_poly_seq["num"].extend(asym_chain_res_id)

        block_dict = {
            "entity_poly": pdbx.CIFCategory(entity_poly),
            "entity_poly_seq": pdbx.CIFCategory(entity_poly_seq),
        }
        return block_dict

    def save_to_cif(
        self, output_path: str, entry_id: str = None, include_bonds: bool = False
    ):
        """
        Save AtomArray to cif.

        Args:
            output_path (str): Output path of cif file.
            entry_id (str, optional): The value of "_entry.id" in cif. Defaults to None.
                                      If None, the entry_id will be the basename of output_path (without ".cif" extension).
            include_bonds (bool, optional): Whether to include  bonds in the cif. Defaults to False.
                                            If set to True and `array` has associated ``bonds`` , the
                                            intra-residue bonds will be written into the ``chem_comp_bond``
                                            category.
                                            Inter-residue bonds will be written into the ``struct_conn``
                                            independent of this parameter.

        """
        if entry_id is None:
            entry_id = os.path.basename(output_path).replace(".cif", "")

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        if self.entity_poly_type is not None:
            block_dict.update(self._get_entity_poly_and_entity_poly_seq_block())

        block = pdbx.CIFBlock(block_dict)
        cif = pdbx.CIFFile(
            {
                os.path.basename(output_path).replace(".cif", "")
                + "_predicted_by_disco": block
            }
        )
        pdbx.set_structure(cif, self.atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block.get("atom_site")
        atom_site["label_entity_id"] = self.atom_array.label_entity_id
        cif.write(output_path)


def make_dummy_feature(
    features_dict: Mapping[str, torch.Tensor],
    dummy_feats: Sequence = ["msa"],
) -> dict[str, torch.Tensor]:
    num_token = features_dict["token_index"].shape[0]
    num_atom = features_dict["atom_to_token_idx"].shape[0]
    num_msa = 1
    num_templ = 4
    num_pockets = 30
    feat_shape, _ = get_data_shape_dict(
        num_token=num_token,
        num_atom=num_atom,
        num_msa=num_msa,
        num_templ=num_templ,
        num_pocket=num_pockets,
    )
    for feat_name in dummy_feats:
        if feat_name not in ["msa", "template"]:
            cur_feat_shape = feat_shape[feat_name]
            features_dict[feat_name] = torch.zeros(cur_feat_shape)
    if "msa" in dummy_feats:
        # features_dict["msa"] = features_dict["restype"].unsqueeze(0)
        features_dict["msa"] = torch.nonzero(features_dict["restype"])[:, 1].unsqueeze(
            0
        )
        assert features_dict["msa"].shape == feat_shape["msa"]
        features_dict["has_deletion"] = torch.zeros(feat_shape["has_deletion"])
        features_dict["deletion_value"] = torch.zeros(feat_shape["deletion_value"])
        features_dict["profile"] = features_dict["restype"]
        assert (
            features_dict["profile"].shape == feat_shape["profile"]
        ), f"dict shape: {features_dict['profile'].shape, feat_shape['profile']}"
        features_dict["deletion_mean"] = torch.zeros(feat_shape["deletion_mean"])
        for key in [
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ]:
            features_dict[key] = torch.tensor(0, dtype=torch.int32)

    if "template" in dummy_feats:
        features_dict["template_restype"] = (
            torch.ones(feat_shape["template_restype"]) * 31
        )  # gap
        features_dict["template_all_atom_mask"] = torch.zeros(
            feat_shape["template_all_atom_mask"]
        )
        features_dict["template_all_atom_positions"] = torch.zeros(
            feat_shape["template_all_atom_positions"]
        )
    return features_dict


def data_type_transform(
    feat_or_label_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Transforms data types of features/labels to appropriate torch types.

    Casts entries whose keys appear in ``IntDataList`` to ``torch.long``.

    Args:
        feat_or_label_dict: Mapping of feature or label names to tensors.

    Returns:
        The same mapping with integer-type features cast to ``torch.long``.
    """
    for key, value in feat_or_label_dict.items():
        if key in IntDataList:
            feat_or_label_dict[key] = value.to(torch.long)

    return feat_or_label_dict


# List of "index" or "type" data
# Their data type should be int
IntDataList = [
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "ref_space_uid",
    "template_restype",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
]


# shape of the data
def get_data_shape_dict(num_token, num_atom, num_msa, num_templ, num_pocket):
    """
    Generate a dictionary containing the shapes of all data.

    Args:
        num_token (int): Number of tokens.
        num_atom (int): Number of atoms.
        num_msa (int): Number of MSA sequences.
        num_templ (int): Number of templates.
        num_pocket (int): Number of pockets to the same interested ligand.

    Returns:
        dict: A dictionary containing the shapes of all data.
    """
    # Features in AlphaFold3 SI Table5
    feat = {
        # Token features
        "residue_index": (num_token,),
        "token_index": (num_token,),
        "asym_id": (num_token,),
        "entity_id": (num_token,),
        "sym_id": (num_token,),
        "restype": (num_token, len(STD_RESIDUES) + 1),
        # chain permutation features
        "entity_mol_id": (num_atom,),
        "mol_id": (num_atom,),
        "mol_atom_index": (num_atom,),
        # Reference features
        "ref_pos": (num_atom, 3),
        "ref_mask": (num_atom,),
        "ref_element": (num_atom, 128),  # note: 128 elem in the paper
        "ref_charge": (num_atom,),
        "ref_atom_name_chars": (num_atom, 4, 64),
        "ref_space_uid": (num_atom,),
        # Msa features
        # "msa": (num_msa, num_token, 32),
        "msa": (num_msa, num_token),
        "has_deletion": (num_msa, num_token),
        "deletion_value": (num_msa, num_token),
        "profile": (num_token, len(STD_RESIDUES) + 1),
        "deletion_mean": (num_token,),
        # Template features
        "template_restype": (num_templ, num_token),
        "template_all_atom_mask": (num_templ, num_token, 37),
        "template_all_atom_positions": (num_templ, num_token, 37, 3),
        "template_pseudo_beta_mask": (num_templ, num_token),
        "template_backbone_frame_mask": (num_templ, num_token),
        "template_distogram": (num_templ, num_token, num_token, 39),
        "template_unit_vector": (num_templ, num_token, num_token, 3),
        # Bond features
        "token_bonds": (num_token, num_token),
    }

    # Extra features needed
    extra_feat = {
        # Input features
        "atom_to_token_idx": (num_atom,),  # after crop
        "atom_to_tokatom_idx": (num_atom,),  # after crop
        "pae_rep_atom_mask": (num_atom,),  # same as "pae_rep_atom_mask" in label_dict
        "is_distillation": (1,),
    }

    # Label
    label = {
        "coordinate": (num_atom, 3),
        "coordinate_mask": (num_atom,),
        # "centre_atom_mask": (num_atom,),
        # "centre_centre_distance": (num_token, num_token),
        # "centre_centre_distance_mask": (num_token, num_token),
        "distogram_rep_atom_mask": (num_atom,),
        "pae_rep_atom_mask": (num_atom,),
        "plddt_m_rep_atom_mask": (num_atom,),
        "modified_res_mask": (num_atom,),
        "bond_mask": (num_atom, num_atom),
        "is_protein": (num_atom,),  # Atom level, not token level
        "is_rna": (num_atom,),
        "is_dna": (num_atom,),
        "is_ligand": (num_atom,),
        "resolution": (1,),
        # Metrics
        "interested_ligand_mask": (
            num_pocket,
            num_atom,
        ),
        "pocket_mask": (
            num_pocket,
            num_atom,
        ),
    }

    # Merged
    all_feat = {**feat, **extra_feat}
    return all_feat, label


def mask_runs(
    x: torch.Tensor, sentinel_value: int, min_run_length: int
) -> torch.Tensor:
    """Identifies consecutive runs of a sentinel value and returns a boolean mask.

    Args:
        x: 1-D input tensor.
        sentinel_value: The value whose consecutive runs should be detected.
        min_run_length: Minimum run length to be included in the mask.

    Returns:
        Boolean tensor of the same shape as ``x``, True where a position
        belongs to a qualifying run.
    """
    is_sentinel = (x == sentinel_value).int()

    zero_tnsr = torch.tensor([0], device=x.device)
    padded = torch.cat([zero_tnsr, is_sentinel, zero_tnsr])
    diff = padded[1:] - padded[:-1]

    starts = (diff == 1).nonzero(as_tuple=True)[0]
    ends = (diff == -1).nonzero(as_tuple=True)[0]
    lengths = ends - starts

    # Filter runs with length >= 3
    valid = lengths >= min_run_length
    starts = starts[valid]
    ends = ends[valid]

    # Vectorized construction of index ranges
    # Create a flat list of indices from all runs
    if len(starts) == 0:
        return torch.zeros_like(x, dtype=torch.bool)

    all_indices = torch.cat(
        [torch.arange(s, e) for s, e in zip(starts, ends, strict=False)]
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[all_indices] = True

    return mask


def get_histag_mask(token_array: "TokenArray", atom_array: AtomArray):
    """Identifies polyhistidine (His-tag) regions in protein chains.

    Scans each protein chain for consecutive runs of histidine tokens and
    returns a mask that is False for His-tag positions and True elsewhere.

    Args:
        token_array: TokenArray with per-token values and atom indices.
        atom_array: Biotite AtomArray with chain information.

    Returns:
        Boolean tensor over protein tokens (True = not a His-tag).
        Empty tensor if no protein chains are found.
    """
    atom_tok_idx = np.concatenate(
        [
            np.full(len(tok.atom_indices), fill_value=i)
            for i, tok in enumerate(token_array)
        ]
    )

    masks = []
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    for chain_start, chain_end in zip(
        chain_starts[:-1], chain_starts[1:], strict=False
    ):
        start_tok_idx = atom_tok_idx[chain_start]
        end_tok_idx = atom_tok_idx[-1 if chain_end == len(atom_tok_idx) else chain_end]

        if chain_end == len(atom_tok_idx):
            end_tok_idx += 1

        tokens = torch.tensor(token_array[start_tok_idx:end_tok_idx].get_values())
        if (tokens > _TOTAL_NUM_AAS).all():
            continue

        chain_histag_mask = mask_runs(
            tokens[tokens <= _TOTAL_NUM_AAS],
            sentinel_value=STD_RESIDUES["HIS"],
            min_run_length=_MIN_HISTAG_LENGTH,
        )

        masks.append(chain_histag_mask)

    # Want to return a true if not a histag and false if so, so negate here
    return ~torch.cat(masks) if len(masks) > 0 else torch.tensor([])
