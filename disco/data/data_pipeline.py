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

from collections import defaultdict

import torch
from biotite.structure import AtomArray

torch.multiprocessing.set_sharing_strategy("file_system")


class DataPipeline:
    """Data pipeline utilities for processing AtomArray objects.

    Provides static helper methods for extracting entity-level metadata
    from Biotite AtomArray structures.
    """

    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id
