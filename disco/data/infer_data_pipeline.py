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
import json
import logging
import time
import traceback
import warnings
from collections.abc import Mapping
from typing import Any

import hydra
import numpy as np
import omegaconf
import torch
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from disco.data.constants import (
    MASK_STD_RESIDUES,
    PRO_STD_RESIDUES,
    PRO_STD_RESIDUES_VALS_SET,
)
from disco.data.data_pipeline import DataPipeline
from disco.data.json_to_feature import SampleDictToFeatures
from disco.data.task_manager import TaskManager
from disco.data.utils import data_type_transform, make_dummy_feature
from disco.utils.torch_utils import dict_to_tensor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")

COLLATE_KEYS = {
    "token_array",
    "atom_array",
    "restype",
    "backbone_atom_mask",
    "is_protein",
    "profile",
    "deletion_mean",
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "masked_prot_restype",
    "atom_to_token_idx",
    "asym_id",
    "prot_residue_mask",
    "residue_index",
    "entity_id",
    "sym_id",
    "token_index",
    "token_bonds",
}


def get_inference_dataloader(
    fabric, configs: Any, num_eval_seeds: int | list[int] | None = None
) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        task_manager=hydra.utils.instantiate(configs.task_manager),
    )

    if num_eval_seeds is not None:
        datasets_to_concat = []

        list_types = (list, omegaconf.listconfig.ListConfig)
        seed_iter = (
            num_eval_seeds
            if isinstance(num_eval_seeds, list_types)
            else range(num_eval_seeds)
        )

        for seed in seed_iter:
            this_dataset = copy.deepcopy(inference_dataset)
            this_dataset.set_seed(seed)
            datasets_to_concat.append(this_dataset)

        inference_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)

    sampler = None
    if fabric is not None:
        sampler = DistributedSampler(
            dataset=inference_dataset,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
        )

    def collate_fn(batch):
        """Collates batch samples into stacked tensors with padding.

        Stacks tensors in ``COLLATE_KEYS`` across the batch dimension and
        optionally duplicates samples for sequence-level ensembling.

        Args:
            batch: List of sample tuples from the dataset.

        Returns:
            Tuple containing the collated batch as a single-element tuple.
        """
        input_feature_dicts = [x[0]["input_feature_dict"] for x in batch]

        dupe_sample2feat = None
        if configs.n_seq_duplicates_per_structure > 1:
            assert len(input_feature_dicts) == 1
            input_feature_dicts = [
                copy.deepcopy(input_feature_dicts[0])
                for _ in range(configs.n_seq_duplicates_per_structure)
            ]

            dupe_sample2feat = [
                copy.deepcopy(batch[0][-2])
                for _ in range(configs.n_seq_duplicates_per_structure)
            ]

        collated_feature_dict = {}
        for collate_key in COLLATE_KEYS:
            # Needs to be padded soon
            data = [ftr_dict[collate_key] for ftr_dict in input_feature_dicts]
            if torch.is_tensor(data[0]):
                collated_feature_dict[collate_key] = torch.stack(data)
            elif isinstance(data, np.ndarray):
                collated_feature_dict[collate_key] = np.stack(data)
            elif isinstance(data, list):
                collated_feature_dict[collate_key] = data
            else:
                raise ValueError(f"Unrecognized type during collation {type(data)}")

        all_keys = set(input_feature_dicts[0].keys())
        non_collated_keys = all_keys.difference(COLLATE_KEYS)
        for key in non_collated_keys:
            collated_feature_dict[key] = input_feature_dicts[0][key]

        # Make batch[0] mutable
        batch[0] = list(batch[0])
        batch[0][0]["input_feature_dict"] = collated_feature_dict
        for i in range(1, len(batch[0])):
            new_data = [batch[j][i] for j in range(len(batch))]
            if isinstance(new_data[0], str):
                if any(map(lambda x: x != "", new_data)):
                    new_data = "\n".join(new_data)
                else:
                    new_data = ""

            batch[0][i] = new_data

        batch[0][1] = collated_feature_dict["atom_array"]

        if dupe_sample2feat is not None:
            batch[0][-2] = dupe_sample2feat

        return (tuple(batch[0]),)

    use_collate_fn_flag = (
        configs.infer_batch_size > 1 or configs.n_seq_duplicates_per_structure > 1
    )

    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=configs.infer_batch_size,
        sampler=sampler,
        collate_fn=collate_fn if use_collate_fn_flag else lambda batch: batch,
        num_workers=configs.num_workers,
    )
    return dataloader


def build_inference_features(
    sample2feat: SampleDictToFeatures,
    atom_array: AtomArray | None = None,
    bb_only: bool = False,
) -> tuple[dict, AtomArray, dict]:
    """
    Given an already initialized sample2feat builds inference features. Can
    take an optional atom_array, in which case this method does not remake
    the atom array in sample2feat and instead reuses the atom_array passed in.
    This is used during inference when a residue is changed by the discrete
    diffusion model.

    Args:
        sample2feat (SampleDictToFeatures): Object which makes base of feature dict
        atom_array (Optional[biotite.AtomArray]): If specified, use this atom array when getting
                                                  features from sample2feat instead of generating
                                                  it from scratch.

    Returns:
        Tuple[dict, biotite.AtomArray, dict]: First dict is features, then the atom array,
                                              then a dict which tracks how much time the
                                              featurizing took.
    """
    t0 = time.time()
    features_dict, atom_array, token_array = sample2feat.get_feature_dict(
        atom_array, called_during_generation=atom_array is not None, bb_only=bb_only
    )

    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    entity_poly_type = sample2feat.entity_poly_type
    t1 = time.time()

    # Msa features
    entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
    msa_features = {}

    # Make dummy features for not implemented features
    dummy_feats = ["template"]
    if len(msa_features) == 0:
        dummy_feats.append("msa")
    else:
        msa_features = dict_to_tensor(msa_features)
        features_dict.update(msa_features)
    features_dict = make_dummy_feature(
        features_dict=features_dict,
        dummy_feats=dummy_feats,
    )

    # Transform to right data type
    feat = data_type_transform(feat_or_label_dict=features_dict)

    masked_residues = PRO_STD_RESIDUES | MASK_STD_RESIDUES
    cond = lambda x, i: (
        (x in masked_residues)
        and (feat["true_restype_id"][i] in masked_residues)
        and (token_array[i].value in PRO_STD_RESIDUES_VALS_SET)
    )

    feat["atom_array"] = atom_array
    feat["token_array"] = token_array
    feat["masked_prot_restype"] = torch.tensor(
        [masked_residues[x] for i, x in enumerate(feat["restype_id"]) if cond(x, i)]
    )

    feat["prot_residue_mask"] = torch.tensor(
        [token.value in PRO_STD_RESIDUES_VALS_SET for token in token_array]
    )

    t2 = time.time()

    data = {}
    data["input_feature_dict"] = feat

    # Add dimension related items
    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]

    stats = {}
    for mol_type in ["ligand", "protein", "dna", "rna"]:
        mol_type_mask = feat[f"is_{mol_type}"].bool()
        stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
        stats[f"{mol_type}/token"] = len(
            torch.unique(feat["atom_to_token_idx"][mol_type_mask])
        )

    N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
    data.update(
        {
            "N_asym": torch.tensor([N_asym]),
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
        }
    )

    def formatted_key(key):
        """Formats a ``type/unit`` key into ``N_{type}_{unit}`` format.

        Args:
            key: String in ``"mol_type/unit"`` format (e.g. ``"protein/atom"``).

        Returns:
            Formatted string such as ``"N_prot_atom"`` or ``"N_lig_token"``.
        """
        type_, unit = key.split("/")
        if type_ == "protein":
            type_ = "prot"
        elif type_ == "ligand":
            type_ = "lig"
        else:
            pass
        return f"N_{type_}_{unit}"

    data.update(
        {
            formatted_key(k): torch.tensor([stats[k]])
            for k in [
                "protein/atom",
                "ligand/atom",
                "dna/atom",
                "rna/atom",
                "protein/token",
                "ligand/token",
                "dna/token",
                "rna/token",
            ]
        }
    )
    data.update({"entity_poly_type": entity_poly_type})
    t3 = time.time()
    time_tracker = {
        "crop": t1 - t0,
        "featurizer": t2 - t1,
        "added_feature": t3 - t2,
    }

    return data, atom_array, time_tracker


class InferenceDataset(Dataset):
    """PyTorch Dataset for inference from JSON input files.

    Reads a JSON file containing one or more sample dictionaries, featurizes
    each sample on access, and returns feature dicts, atom arrays, and
    metadata.

    Args:
        input_json_path: Path to the input JSON file.
        dump_dir: Directory for output artifacts.
        task_manager: TaskManager instance for masking.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        task_manager: TaskManager,
        seed: int | None = None,
    ) -> None:
        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.task_manager = task_manager
        self.seed = None
        with open(self.input_json_path) as f:
            self.inputs = json.load(f)

    def set_seed(self, seed: int):
        self.seed = seed

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        sample2feat = SampleDictToFeatures(
            single_sample_dict, self.task_manager, backbone_only=True
        )
        return *build_inference_features(sample2feat), sample2feat

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _, sample2feat = self.process_one(
                single_sample_dict=single_sample_dict
            )

            error_message = ""
        except Exception as e:
            data, atom_array, sample2feat = {}, None, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        data["seed"] = self.seed
        return data, atom_array, sample2feat, error_message
