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

import os
from pathlib import Path

import torch
from biotite.structure import AtomArray

from disco.data.utils import save_structure_cif, save_structure_pdb
from disco.utils.file_io import save_json
from disco.utils.torch_utils import round_values


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """
    # Remove atom_coordinate
    full_confidence_dict.pop("atom_coordinate")
    # Remove atom_is_polymer
    full_confidence_dict.pop("atom_is_polymer")
    # Keep two decimal places
    full_confidence_dict = round_values(full_confidence_dict)
    return full_confidence_dict


class DataDumper:
    """Saves model predictions (structures and confidence scores) to disk.

    Handles writing predicted coordinates as structure files (CIF/PDB) and
    confidence scores as JSON files, organized by dataset, sample, and seed.

    Args:
        base_dir: Root directory where all prediction outputs are saved.
        need_atom_confidence: If True, also saves per-atom confidence
            data to JSON files.
    """

    def __init__(self, base_dir, need_atom_confidence: bool = False):
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        step=None,
        file_format: str = "cif",
        dump_dir: str | None = None,
        append_preds_dir: bool = True,
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
            step (int, optional): The step number. Defaults to None.
        """
        if dump_dir is None:
            dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed, step)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        structure_paths = self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            file_format=file_format,
            append_preds_dir=append_preds_dir,
        )
        return structure_paths

    def _get_dump_dir(
        self, dataset_name: str, sample_name: str, seed: int, step: int
    ) -> str:
        """Generates the directory path for dumping prediction data.

        Constructs a nested directory path under ``base_dir``. When ``step``
        is provided, the step number is included in the path hierarchy.

        Args:
            dataset_name: Name of the dataset (used as a subdirectory).
            sample_name: Identifier for the sample (e.g. PDB ID).
            seed: Random seed used for this prediction run.
            step: Training or inference step number. If not None, a
                ``step_{step}`` directory is inserted into the path.

        Returns:
            Absolute path string for the dump directory.
        """
        if step is not None:
            return os.path.join(
                self.base_dir, dataset_name, f"step_{step}", sample_name, f"seed_{seed}"
            )
        dump_dir = os.path.join(
            self.base_dir, dataset_name, sample_name, f"seed_{seed}"
        )
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        file_format: str = "cif",
        append_preds_dir: bool = True,
    ):
        """
        Dump raw predictions from the model:
            structure: Save the predicted coordinates as CIF files.
            confidence: Save the confidence data as JSON files.
        """
        if append_preds_dir:
            prediction_save_dir = os.path.join(dump_dir, "predictions")
            os.makedirs(prediction_save_dir, exist_ok=True)
        else:
            prediction_save_dir = dump_dir

        # Dump structure
        if "atom_array" in pred_dict:
            atom_array = pred_dict["atom_array"]
            if isinstance(atom_array, list) and len(atom_array) == 1:
                atom_array = atom_array[0]

        structure_paths = self._save_structure(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            file_format=file_format,
        )
        # Dump confidence
        if "summary_confidence" in pred_dict:
            self._save_confidence(
                data=pred_dict,
                prediction_save_dir=prediction_save_dir,
                sample_name=pdb_id,
            )

        return structure_paths

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        file_format: str = "cif",
        pred_dict=None,
        idx_offset: int = 0,
    ):
        """Saves predicted structures to CIF or PDB files.

        Iterates over the sample dimension of ``pred_coordinates`` and writes
        each predicted structure as a separate file.

        Args:
            pred_coordinates: Predicted atom coordinates with shape
                ``(N_sample, N_atom, 3)``.
            prediction_save_dir: Directory to write structure files into.
            sample_name: Base name for output files (e.g. PDB ID).
            atom_array: Biotite AtomArray used as the structural template.
            entity_poly_type: Mapping of entity IDs to polymer types.
            file_format: Output format, either ``"cif"`` or ``"pdb"``.
            pred_dict: Optional prediction dictionary. If it contains an
                ``"atom_array"`` key, per-sample atom arrays are used instead.
            idx_offset: Starting index offset for naming output files,
                added to the sample index in the filename.

        Returns:
            List of file path strings for the saved structure files.
        """
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]
        output_fpaths = []
        for idx in range(N_sample):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_sample_{idx + idx_offset}.{file_format}",
            )
            if file_format == "cif":
                save_fn = save_structure_cif
            elif file_format == "pdb":
                save_fn = save_structure_pdb
            else:
                raise ValueError(
                    f"Unsupported file format: {file_format!r}. Must be 'cif' or 'pdb'."
                )
            if pred_dict is not None and "atom_array" in pred_dict:
                atom_array = pred_dict["atom_array"]
                if isinstance(atom_array, list):
                    atom_array = atom_array[idx]

                save_fn(
                    atom_array=atom_array,
                    pred_coordinate=pred_coordinates[
                        idx, : len(atom_array), :
                    ],  # remove padding
                    output_fpath=output_fpath,
                    entity_poly_type=entity_poly_type,
                    pdb_id=sample_name,
                )
            else:
                save_fn(
                    atom_array=atom_array,
                    pred_coordinate=pred_coordinates[
                        idx, : len(atom_array), :
                    ],  # remove padding
                    output_fpath=output_fpath,
                    entity_poly_type=entity_poly_type,
                    pdb_id=sample_name,
                )
            output_fpaths.append(output_fpath)
        return output_fpaths

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
        sorted_by_ranking_score: bool = True,
    ):
        """Saves confidence scores and full confidence data to JSON files.

        Writes per-sample summary confidence JSON files, optionally sorted
        by ranking score (highest first). When ``need_atom_confidence`` is
        enabled, also writes full per-atom confidence data.

        Args:
            data: Prediction dictionary containing ``"summary_confidence"``
                (list of dicts) and optionally ``"full_data"`` (list of dicts).
            prediction_save_dir: Directory to write JSON files into.
            sample_name: Base name for output files (e.g. PDB ID).
            sorted_by_ranking_score: If True, files are ranked by
                descending ``ranking_score`` from the summary confidence.
        """
        N_sample = len(data["summary_confidence"])
        for idx in range(N_sample):
            if self.need_atom_confidence:
                data["full_data"][idx] = get_clean_full_confidence(
                    data["full_data"][idx]
                )
        sorted_indices = range(N_sample)
        if sorted_by_ranking_score:
            sorted_indices = sorted(
                range(N_sample),
                key=lambda i: data["summary_confidence"][i]["ranking_score"],
                reverse=True,
            )

        for rank, idx in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_summary_confidence_sample_{idx}.json",
            )
            save_json(data["summary_confidence"][idx], output_fpath, indent=4)
            if self.need_atom_confidence:
                output_fpath = os.path.join(
                    prediction_save_dir,
                    f"{sample_name}_full_data_sample_{idx}.json",
                )
                save_json(data["full_data"][idx], output_fpath, indent=None)
