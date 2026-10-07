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

"""Tests for the XPU branch of InferenceRunner.init_env, run without any
accelerator present by mocking XPUAccelerator.is_available.
"""

from unittest import mock

import pytest
import torch

from omegaconf import OmegaConf

from runner.inference import InferenceRunner


def _make_runner(**overrides):
    configs = OmegaConf.create(
        {
            "logger": {},
            "fabric": {"accelerator": "xpu", "num_nodes": 1},
            "use_deepspeed_evo_attention": False,
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(configs, key, value, merge=False)
    runner = object.__new__(InferenceRunner)
    runner.configs = configs
    return runner


def test_rejects_multi_node_xpu():
    runner = _make_runner(**{"fabric.num_nodes": 2})
    with pytest.raises(ValueError, match="single node"):
        runner.init_env()


def test_rejects_multi_device_xpu():
    runner = _make_runner(**{"fabric.devices": 2})
    with pytest.raises(ValueError, match="single device"):
        runner.init_env()


def test_rejects_unavailable_xpu():
    runner = _make_runner()
    with mock.patch("runner.inference.XPUAccelerator.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="no XPU device is available"):
            runner.init_env()


def test_rejects_deepspeed_evo_attention_on_xpu():
    runner = _make_runner(**{"use_deepspeed_evo_attention": True})
    with mock.patch("runner.inference.XPUAccelerator.is_available", return_value=True):
        with pytest.raises(ValueError, match="EvoformerAttention"):
            runner.init_env()


def test_init_model_passes_device_to_structure_encoder():
    runner = object.__new__(InferenceRunner)
    runner.device = torch.device("cpu")
    runner.configs = OmegaConf.create(
        {
            "structure_encoder": {
                "use_structure_encoder": True,
                "args": {"_target_": "unused"},
            },
            "sequence_sampling_strategy": {"_target_": "unused"},
        }
    )
    with (
        mock.patch("runner.inference.hydra.utils.instantiate") as instantiate,
        mock.patch("runner.inference.DISCO") as disco_cls,
    ):
        instantiate.side_effect = [mock.sentinel.encoder, mock.sentinel.strategy]
        runner.init_model()

    encoder_call = instantiate.call_args_list[0]
    assert encoder_call.kwargs["device"] == torch.device("cpu")
    disco_cls.assert_called_once_with(
        runner.configs, mock.sentinel.encoder, mock.sentinel.strategy
    )
