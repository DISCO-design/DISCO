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
import pickle

import pandas as pd
import torch


class DistWrapper:
    """Wrapper providing convenient access to distributed training environment variables.

    Reads ``RANK``, ``LOCAL_RANK``, ``LOCAL_WORLD_SIZE``, and ``WORLD_SIZE`` from the environment.
    """

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)


DIST_WRAPPER = DistWrapper()


def traverse_and_aggregate(gathered_dict, aggregation_func=None):
    """Recursively merges a list of nested dicts into a single dict.

    Leaf values from each dict are collected into lists. If an
    ``aggregation_func`` is provided, it is applied to each leaf list.

    Args:
        gathered_dict: List of dicts with identical structure to merge.
        aggregation_func: Optional callable applied to each list of leaf
            values. If None, leaves are kept as lists.

    Returns:
        A single merged dict mirroring the input structure.
    """
    merged_dict = {}
    keys = gathered_dict[0].keys()
    for key in keys:
        value = [d[key] for d in gathered_dict if key in d]
        if isinstance(value[0], dict):
            merged_dict[key] = traverse_and_aggregate(
                value, aggregation_func=aggregation_func
            )
        else:
            if aggregation_func is not None:
                value = aggregation_func(value)
            merged_dict[key] = value

    return merged_dict


def gather_and_merge(fabric, metrics, aggregation_func=None):
    """Gathers metrics from all DDP workers and merges them into a single dict.

    Uses ``fabric.all_gather`` to collect metrics from every rank, then
    delegates to :func:`traverse_and_aggregate` for recursive merging.

    Args:
        fabric: Lightning Fabric instance for distributed communication.
        metrics: Nested dict of metric data to gather.
        aggregation_func: Optional callable applied to each list of leaf
            values after gathering.

    Returns:
        Merged dict containing aggregated metrics from all ranks.
    """
    # print(metrics)
    # fix different size
    fixed_metrics = fabric.all_gather(metrics)
    merged_metrics = traverse_and_aggregate([fixed_metrics], aggregation_func)
    return merged_metrics


def gather_dfs(fabric, df):
    """Gathers DataFrames from all DDP ranks and concatenates them.

    Serializes each local DataFrame to bytes, pads to a uniform length,
    gathers across all ranks, deserializes, and concatenates into a single
    DataFrame.

    Args:
        fabric: Lightning Fabric instance for distributed communication.
        df: Local ``pandas.DataFrame`` to gather.

    Returns:
        pandas.DataFrame: Concatenation of DataFrames from all ranks.
    """
    serialized = pickle.dumps(df)

    # Convert to tensor
    byte_tensor = torch.ByteTensor(list(serialized))

    # All gather lengths first
    length = torch.tensor([len(byte_tensor)], device=fabric.device)
    all_lengths = fabric.all_gather(length)
    max_len = all_lengths.max().item()

    # Pad tensors to max_len
    if len(byte_tensor) < max_len:
        pad = torch.zeros(max_len - len(byte_tensor), dtype=torch.uint8)
        byte_tensor = torch.cat([byte_tensor, pad])

    gathered_bytes = fabric.all_gather(byte_tensor)

    # Deserialize
    dfs = []
    for i, length in enumerate(all_lengths):
        serialized = bytes(gathered_bytes[i][:length])
        dfs.append(pickle.loads(serialized))

    return pd.concat(dfs)
