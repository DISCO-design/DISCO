# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
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

"""Device helpers so inference isn't hard-coded to CUDA.

Also provides a Lightning Fabric accelerator for a single Intel GPU, since
Fabric only ships cpu/cuda/mps/xla accelerators as of 2.6.x.
"""

import torch

from lightning.fabric.accelerators import Accelerator


def accelerator_type() -> str:
    """Active accelerator type ("cuda", "xpu", ...), or "cpu" if none is active.

    current_accelerator() alone only reflects what torch was built with, not
    whether a device is actually there, so check is_available() first.
    """
    if not torch.accelerator.is_available():
        return "cpu"
    return torch.accelerator.current_accelerator().type


def manual_seed_all(seed: int) -> None:
    """Seed the active accelerator's RNG, mirroring torch.cuda.manual_seed_all."""
    acc_type = accelerator_type()
    if acc_type != "cpu":
        getattr(torch, acc_type).manual_seed_all(seed)


def empty_cache() -> None:
    """Release cached accelerator memory, if an accelerator is active."""
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()


def autocast_disabled():
    """Autocast-disabled context for whichever accelerator is active."""
    return torch.autocast(device_type=accelerator_type(), enabled=False)


# Adapted from anomalib's XPUAccelerator (Copyright (C) 2025 Intel Corporation,
# Apache-2.0): https://github.com/open-edge-platform/anomalib
class XPUAccelerator(Accelerator):
    """Fabric accelerator for a single Intel GPU (`torch.device("xpu")`)."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise RuntimeError(f"Device should be xpu, got {device} instead")
        torch.xpu.set_device(device)

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices):
        return devices if isinstance(devices, list) else [devices]

    @staticmethod
    def get_parallel_devices(devices):
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        return "xpu"
