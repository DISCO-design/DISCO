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

"""Tests for disco.utils.device. Mocked so no accelerator has to be present."""

from unittest import mock

import torch

from disco.utils import device as device_utils


def test_accelerator_type_defaults_to_cpu():
    with mock.patch("torch.accelerator.is_available", return_value=False):
        assert device_utils.accelerator_type() == "cpu"


def test_accelerator_type_reports_active_accelerator():
    with mock.patch("torch.accelerator.is_available", return_value=True):
        with mock.patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("xpu"),
        ):
            assert device_utils.accelerator_type() == "xpu"


def test_manual_seed_all_skips_cpu():
    with mock.patch.object(device_utils, "accelerator_type", return_value="cpu"):
        with mock.patch("torch.cuda.manual_seed_all") as cuda_seed:
            device_utils.manual_seed_all(0)
        cuda_seed.assert_not_called()


def test_manual_seed_all_dispatches_to_active_accelerator():
    with mock.patch.object(device_utils, "accelerator_type", return_value="xpu"):
        with mock.patch("torch.xpu.manual_seed_all") as xpu_seed:
            device_utils.manual_seed_all(0)
        xpu_seed.assert_called_once_with(0)


def test_empty_cache_noop_without_accelerator():
    with mock.patch("torch.accelerator.is_available", return_value=False):
        with mock.patch("torch.accelerator.empty_cache") as empty_cache:
            device_utils.empty_cache()
        empty_cache.assert_not_called()


def test_empty_cache_clears_active_accelerator():
    with mock.patch("torch.accelerator.is_available", return_value=True):
        with mock.patch("torch.accelerator.empty_cache") as empty_cache:
            device_utils.empty_cache()
        empty_cache.assert_called_once()


def test_autocast_disabled_uses_active_device_type():
    with mock.patch.object(device_utils, "accelerator_type", return_value="xpu"):
        ctx = device_utils.autocast_disabled()
    assert ctx.device == "xpu"
    assert not ctx._enabled


def test_xpu_accelerator_rejects_non_xpu_device():
    accelerator = device_utils.XPUAccelerator()
    try:
        accelerator.setup_device(torch.device("cpu"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for a non-xpu device")


def test_xpu_accelerator_parses_and_expands_devices():
    accelerator = device_utils.XPUAccelerator
    assert accelerator.parse_devices(0) == [0]
    assert accelerator.parse_devices([0, 1]) == [0, 1]
    assert accelerator.get_parallel_devices([0]) == [torch.device("xpu", 0)]
    assert accelerator.name() == "xpu"


def test_xpu_accelerator_is_available_matches_torch_xpu():
    with mock.patch("torch.xpu.is_available", return_value=True):
        assert device_utils.XPUAccelerator.is_available() is True
    with mock.patch("torch.xpu.is_available", return_value=False):
        assert device_utils.XPUAccelerator.is_available() is False
