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

import numpy as np


def run_lengths(arr: np.ndarray):
    """Calculates lengths of consecutive runs of identical values in an array.

    Args:
        arr: 1-D NumPy array of values.

    Returns:
        1-D NumPy integer array where each element is the length of a
        consecutive run of equal values in ``arr``.
    """
    if arr.size == 0:
        return np.array([], dtype=int)

    # Find where the value changes
    change_indices = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    # Add start and end indices
    indices = np.concatenate(([0], change_indices, [len(arr)]))
    # Compute lengths of runs
    return np.diff(indices)
