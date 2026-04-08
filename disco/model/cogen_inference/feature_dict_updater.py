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
import multiprocessing

import numpy as np
import torch
from biotite.structure import AtomArray

from disco.data.constants import MASK_RESNAME, PRO_RES_IDX_TO_RESNAME
from disco.data.infer_data_pipeline import build_inference_features, COLLATE_KEYS
from disco.data.json_to_feature import SampleDictToFeatures
from disco.data.parser import AddAtomArrayAnnot
from disco.data.tokenizer import TokenArray
from disco.utils.geometry import random_transform
from disco.utils.torch_utils import to_device

logger = logging.getLogger(__name__)


def update_atom_array(
    token_array: TokenArray,
    atom_array: AtomArray,
    xt_seq: torch.Tensor,
    xt_seq_changed_idx: torch.Tensor,
    random_transform_ref_pos: bool,
    random_transform_msk_res: bool,
    ref_pos_augment: bool,
) -> AtomArray:
    """
    Given an amino acid sequence that has been changed at some indices and
    an atom array, updates the atom array's restype, cano_seq_resname,
    clean_cano_seq_resname, and reference information for the atoms corresponding
    to the residue(s) that were unmasked.

    Args:
        token_array (TokenArray): The current token array at this point in inference
        atom_array (AtomArray): The current atom array at this point in inference
        xt_seq (torch.Tensor): A non-one-hot tensor which has the current masked and
                               unmasked AA residues
        xt_seq_changed_idx (torch.Tensor): A boolean tensor where a particular index
                                           is True if that element of xt_seq changed
                                           at this time t.

    Returns:
        (AtomArray): The original atom array but with the attributes described
                     in the method summary updated to the new residue(s) in xt_seq.
    """
    is_protein = np.flatnonzero(
        np.array(
            [atom_array[tok.atom_indices[0]].is_protein for tok in token_array],
            dtype=bool,
        )
    )

    xt_seq_changed_idx = xt_seq_changed_idx.cpu().numpy()
    changed_tok_arr = token_array[is_protein][np.flatnonzero(xt_seq_changed_idx)]
    atom_idxs = np.concatenate([tok.atom_indices for tok in changed_tok_arr])
    xt_seq_arr_idxs = np.concatenate(
        [np.array([i] * len(tok.atom_indices)) for i, tok in enumerate(changed_tok_arr)]
    )

    xt_resnames = np.array([PRO_RES_IDX_TO_RESNAME[i.item()] for i in xt_seq])

    # Need to set each of res_name, cano_seq_resname, and clean_cano_seq_resname
    resnames = atom_array.res_name
    resnames[atom_idxs] = xt_resnames[xt_seq_changed_idx][xt_seq_arr_idxs]
    atom_array.set_annotation("restype", resnames)

    if hasattr(atom_array, "cano_seq_resname"):
        cano_seq_resnames = atom_array.cano_seq_resname
        cano_seq_resnames[atom_idxs] = xt_resnames[xt_seq_changed_idx][xt_seq_arr_idxs]
        atom_array.set_annotation("cano_seq_resname", cano_seq_resnames)
    else:
        logger.warning("No cano_seq_resname available")

    if hasattr(atom_array, "clean_cano_seq_resname"):
        clean_cano_seq_resnames = atom_array.clean_cano_seq_resname
        clean_cano_seq_resnames[atom_idxs] = xt_resnames[xt_seq_changed_idx][
            xt_seq_arr_idxs
        ]
        atom_array.set_annotation("clean_cano_seq_resname", clean_cano_seq_resnames)
    else:
        logger.warning("No clean_cano_seq_resname available")

    ref_pos, ref_charge, ref_mask = AddAtomArrayAnnot.add_ref_feat_info(
        atom_array, atom_idxs
    )

    if random_transform_ref_pos:
        trfmd_ref_pos = []
        for ref_space_uid in np.unique(atom_array.ref_space_uid):
            atom_arr_idx = atom_array.ref_space_uid == ref_space_uid
            continue_cond = (
                not random_transform_msk_res
                and atom_array[atom_arr_idx][0].res_name == MASK_RESNAME
            )

            if continue_cond:
                continue

            trfmd_ref_pos.append(
                random_transform(
                    ref_pos[atom_arr_idx],
                    apply_augmentation=ref_pos_augment,
                    centralize=True,
                )
            )

        ref_pos = np.concatenate(trfmd_ref_pos, axis=0)

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)

    return atom_array


def update_ftr_dict_new_single_seq(
    token_array: TokenArray,
    atom_array: AtomArray,
    sample2feat: SampleDictToFeatures,
    xt_seq: torch.Tensor,
    xt_seq_changed_idx: torch.Tensor,
    bb_only: bool,
    random_transform_ref_pos: bool,
    random_transform_msk_res: bool,
    ref_pos_augment: bool,
) -> dict:
    """
    Updates the features in the input feature dict according to the changes in
    the AA sequence in xt_seq.

    Args:
        input_feature_dict (dict): The current input feature dict
        sample2feat (SampleDictToFeatures): An already initialized featurizer
        xt_seq (torch.Tensor): The current amino acid sequence at time t in
                               the inference process.
        xt_seq_changed_idx (torch.Tensor): A boolean tensor where a particular index
                                           is True if that element of xt_seq changed
                                           at this time t.
    Returns:
        An updated input_feature_dict corresponding to the updated AA sequence.
    """
    atom_array = update_atom_array(
        token_array,
        atom_array,
        xt_seq,
        xt_seq_changed_idx,
        random_transform_ref_pos,
        random_transform_msk_res,
        ref_pos_augment,
    )

    data, atom_array, _ = build_inference_features(
        sample2feat, atom_array=atom_array, bb_only=bb_only
    )

    return data["input_feature_dict"]


def feature_dict_submit_fxn(i, args):
    """Wrapper for parallel feature dict updates.

    Calls update_ftr_dict_new_single_seq and returns the sample index
    alongside the result, suitable for use with multiprocessing pools.

    Args:
        i: Sample index in the batch.
        args: Tuple of arguments to pass to update_ftr_dict_new_single_seq.

    Returns:
        Tuple of (sample_index, updated_feature_dict).
    """
    return i, update_ftr_dict_new_single_seq(*args)


class FeatureDictUpdater:
    def __init__(self, pool_size: int):
        spawn_context = multiprocessing.get_context("spawn")
        self.processor_pool = spawn_context.Pool(processes=pool_size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.processor_pool.close()
        self.processor_pool.terminate()
        del self.processor_pool

    def update(
        self,
        input_feature_dict: dict,
        sample2feat: SampleDictToFeatures,
        xt_seq: torch.Tensor,
        xt_seq_changed_idx: torch.Tensor,
        bb_only: bool,
        random_transform_ref_pos: bool,
        random_transform_msk_res: bool,
        ref_pos_augment: bool,
    ) -> dict:
        output = self._update(
            input_feature_dict,
            sample2feat,
            xt_seq,
            xt_seq_changed_idx,
            bb_only,
            random_transform_ref_pos,
            random_transform_msk_res,
            ref_pos_augment,
        )

        return to_device(output, xt_seq.device)

    def _update(
        self,
        input_feature_dict: dict,
        sample2feat: SampleDictToFeatures,
        xt_seq: torch.Tensor,
        xt_seq_changed_idx: torch.Tensor,
        bb_only: bool,
        random_transform_ref_pos: bool,
        random_transform_msk_res: bool,
        ref_pos_augment: bool,
    ) -> dict:
        """
        Does what update_ftr_dict_new_single_seq does but potentially to multiple sequences
        """
        if xt_seq_changed_idx.ndim == 1:
            return update_ftr_dict_new_single_seq(
                input_feature_dict["token_array"],
                input_feature_dict["atom_array"],
                sample2feat,
                xt_seq,
                xt_seq_changed_idx,
                bb_only,
                random_transform_ref_pos,
                random_transform_msk_res,
                ref_pos_augment,
            )

        samples_with_changes = xt_seq_changed_idx.any(dim=-1).nonzero().flatten()

        args = [
            (
                i.item(),
                (
                    input_feature_dict["token_array"][i],
                    input_feature_dict["atom_array"][i],
                    sample2feat[i],
                    xt_seq[i].cpu(),
                    xt_seq_changed_idx[i].cpu(),
                    bb_only,
                    random_transform_ref_pos,
                    random_transform_msk_res,
                    ref_pos_augment,
                ),
            )
            for i in samples_with_changes
        ]

        results = self.processor_pool.starmap(feature_dict_submit_fxn, args)
        for i, single_input_ftr_dict in results:
            for key in COLLATE_KEYS:
                input_feature_dict[key][i] = single_input_ftr_dict[key]

        return input_feature_dict
