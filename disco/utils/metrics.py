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


# from lightning.fabric.fabric import all_reduce

import numpy as np
import torch

from disco.utils.distributed import gather_and_merge
from disco.utils.logger import get_logger

logger = get_logger(__name__)

COMMON_AGGREGATOR = {
    "avg": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
    "pct90": lambda x: np.percentile(x, 90),
    "pct99": lambda x: np.percentile(x, 99),
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
}


class SimpleMetricAggregator:
    """A quite simple metrics calculator that only do simple metrics aggregation.

    Args:
        aggregator_names: List of aggregation function names to apply
            (e.g., ``"avg"``, ``"median"``, ``"max"``). Must be keys in
            ``COMMON_AGGREGATOR``.
        need_gather: If True, gathers metrics across DDP ranks before
            aggregation. Defaults to True.
        fabric: Lightning Fabric instance used for distributed gathering.
    """

    def __init__(self, aggregator_names=None, need_gather=True, fabric=None):
        super().__init__()
        self.need_gather = need_gather
        self._metric_data = {}
        self.fabric = fabric

        self.aggregators = {name: COMMON_AGGREGATOR[name] for name in aggregator_names}

    def add(self, key, value, namespace="default"):
        """Adds a metric value under a namespace and key.

        Args:
            key: Metric name within the namespace.
            value: Metric value. Can be a float, int, ``torch.Tensor``, or
                ``np.ndarray``.
            namespace: Logical grouping for the metric. Defaults to
                ``"default"``.
        """
        value_dict = self._metric_data.setdefault(namespace, {})
        value_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        value_dict[key].append(value)

    def calc(self):
        """Calculates aggregated metrics across all added values.

        Gathers data across DDP ranks if ``need_gather`` is True, then applies
        each configured aggregation function to every metric. Sentinel values
        of -1.0 are excluded before aggregation.

        Returns:
            dict: Mapping of ``"namespace/key.aggregator_name"`` to the
            computed aggregate value.
        """
        metric_data, self._metric_data = self._metric_data, {}
        if self.need_gather:
            metric_data = self._check_dict_len(metric_data=metric_data)
            # metric_data = self._check_dict_len_ddp_safe(metric_data=metric_data)
            metric_data = gather_and_merge(
                self.fabric, metric_data, aggregation_func=lambda l: sum(l, [])
            )
        results = {}
        for agg_name, agg_func in self.aggregators.items():
            for namespace, value_dict in metric_data.items():
                for key, data in value_dict.items():
                    if (
                        isinstance(data, list)
                        and len(data) > 0
                        and isinstance(data[0], torch.Tensor)
                    ):
                        data = [d.detach().cpu().numpy() for d in data]
                    plain_key = f"{namespace}/{key}" if namespace != "default" else key
                    plain_key = f"{plain_key}.{agg_name}"
                    data = np.concatenate(data, axis=0)

                    sentinel_idx = data == -1.0
                    if sentinel_idx.all():
                        continue

                    data = data[~sentinel_idx]
                    results[plain_key] = agg_func(data)

        del metric_data
        return results

    def _check_dict_len(self, metric_data):
        """Pads metric lists to equal length across all keys within each namespace.

        This ensures that all inner lists have the same length before DDP
        gathering, which requires uniform tensor sizes. Shorter lists are
        padded with their current mean value.

        Args:
            metric_data: Nested dict of ``{namespace: {key: [values]}}``.

        Returns:
            The ``metric_data`` dict with all inner lists padded to the same
            length.
        """
        max_len = 0
        for key, item in metric_data.items():
            for key2, item2 in item.items():
                logger.debug(f"metric {key2} is of len {len(item2)}")
                if len(item2) > max_len:
                    max_len = len(item2)

        for key, item in metric_data.items():
            for key2, item2 in item.items():
                if len(item2) < max_len:
                    logger.debug(f"Fixing length of {key2} from {len(item2)}")
                    pad_element = [
                        np.array([np.concatenate(metric_data[key][key2]).mean()])
                    ]
                    metric_data[key][key2] += pad_element * (max_len - len(item2))
                else:
                    continue
        # check metric data items for debug multinode
        for key, item in metric_data.items():
            logger.debug(f"Outer key:{key}")
            for key2, item2 in item.items():
                logger.debug(f"Current length of {key2} is {len(item2)}")
        return metric_data

    def _check_dict_len_ddp_safe(self, metric_data):
        """Synchronizes metric tensor lengths across all DDP ranks.

        Converts all metric values to CUDA tensors, finds the global maximum
        length across ranks via all-reduce, and zero-pads shorter tensors to
        that length so that ``all_gather`` can operate on uniform shapes.

        Args:
            metric_data: Nested dict of ``{namespace: {key: values}}``.

        Returns:
            The ``metric_data`` dict with all tensors padded to the global
            maximum length.
        """
        local_max_len = 0

        # Convert everything to tensors and find local max length
        for outer_key, inner_dict in metric_data.items():
            for inner_key, values in inner_dict.items():
                if not isinstance(values, torch.Tensor):
                    inner_dict[inner_key] = torch.tensor(
                        values, dtype=torch.float32, device="cuda"
                    )
                local_max_len = max(local_max_len, inner_dict[inner_key].shape[0])

        # Get global max length across all ranks
        global_max_len = self.fabric.all_reduce(
            torch.tensor(local_max_len, device="cuda"), reduce_op="MAX"
        ).item()

        # Pad all tensors to global_max_len
        for outer_key, inner_dict in metric_data.items():
            for inner_key, tensor in inner_dict.items():
                if tensor.shape[0] < global_max_len:
                    pad_len = global_max_len - tensor.shape[0]
                    padding = torch.zeros(
                        pad_len, dtype=tensor.dtype, device=tensor.device
                    )
                    inner_dict[inner_key] = torch.cat([tensor, padding], dim=0)

        return metric_data
