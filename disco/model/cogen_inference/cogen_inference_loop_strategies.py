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

import inspect
from abc import ABC, abstractmethod

import torch
from biotite.structure import AtomArray
from tqdm import tqdm

from disco.data.constants import MASK_TOKEN_IDX
from disco.model.cogen_inference.cogen_inference_loop_body_impl import InferenceLoopImpl
from disco.model.cogen_inference.feature_dict_updater import FeatureDictUpdater
from disco.model.utils import InferenceNoiseScheduler


class BaseInferenceLoopStrategy(ABC):
    def __init__(
        self,
        noise_schedule: torch.Tensor,
        ftr_dict_updater: FeatureDictUpdater,
        loop_executor: InferenceLoopImpl,
        noise_scheduler: InferenceNoiseScheduler,
    ):
        self.curr_noise_schedule = noise_schedule
        self.noise_scheduler = noise_scheduler
        self.ftr_dict_updater = ftr_dict_updater
        self.inner_loop_executor = loop_executor

    @abstractmethod
    def run_loop(
        self,
        x_l_init: torch.Tensor,
        xt_seq_init: torch.Tensor,
        input_feature_dict: dict,
    ) -> tuple[
        torch.Tensor,  # The x0 structure
        torch.Tensor,  # The x0 sequence
        AtomArray,  # The x0 atom array
        torch.Tensor,  # The last x0 predicted structure
    ]:
        raise NotImplementedError

    def forward_seq(
        self,
        xt_seq: torch.Tensor,
        new_t: torch.Tensor,
    ):
        """Forward diffusion step for the sequence (masking).

        Randomly masks tokens in the sequence with probability equal to the
        noise level new_t.

        Args:
            xt_seq: Current sequence tensor.
            new_t: Noise level controlling the masking probability.

        Returns:
            Sequence tensor with additional tokens masked.
        """
        should_mask = 1 * (torch.rand_like(xt_seq, dtype=torch.float) <= new_t)
        return (should_mask * MASK_TOKEN_IDX) + ((1 - should_mask) * xt_seq)

    def forward_struct(
        self,
        xt_struct: torch.Tensor,
        new_time: torch.Tensor,
        old_time: torch.Tensor,
    ):
        """Forward diffusion step for the structure (noise addition).

        Adds Gaussian noise to the structure coordinates proportional to the
        difference between the new and old noise levels.

        Args:
            xt_struct: Current structure coordinates tensor.
            new_time: Target time for the noise schedule.
            old_time: Current time for the noise schedule.

        Returns:
            Structure tensor with additional noise applied.
        """
        new_noise = self.noise_scheduler.time_to_noise_lvl(new_time)
        old_noise = self.noise_scheduler.time_to_noise_lvl(old_time)

        noise = torch.randn_like(xt_struct) * (new_noise - old_noise)
        return xt_struct + noise


class BasicInferenceLoopStrategy(BaseInferenceLoopStrategy):
    def run_loop(
        self,
        x_l_init: torch.Tensor,
        xt_seq_init: torch.Tensor,
        input_feature_dict: dict,
    ) -> tuple[
        torch.Tensor,  # The x0 structure
        torch.Tensor,  # The x0 sequence
        AtomArray,  # The x0 atom array
        list[torch.Tensor],  # Trajectory of x0 denoised structures
    ]:
        x_l, xt_seq = x_l_init, xt_seq_init
        can_change_idx = xt_seq == MASK_TOKEN_IDX

        iterator = tqdm(
            enumerate(
                zip(
                    self.curr_noise_schedule[:-1],
                    self.curr_noise_schedule[1:],
                    strict=False,
                )
            ),
            total=len(self.curr_noise_schedule) - 1,
        )

        xt_seq_changed = True
        for i, (c_tau_last, c_tau) in iterator:
            (
                x_l,
                xt_seq,
                input_feature_dict,
                xt_seq_changed,
            ) = self.inner_loop_executor.execute(
                x_l,
                xt_seq,
                c_tau,
                c_tau_last,
                i,
                i - 1,
                input_feature_dict,
                xt_seq_changed,
                can_change_idx,
            )

        return (
            x_l,
            xt_seq,
            input_feature_dict["atom_array"],
            self.inner_loop_executor.x1_denoised,
            self.inner_loop_executor.trajectory_noise_levels,
            self.inner_loop_executor.xt_trajectory,
            self.inner_loop_executor.diagnostic_records,
        )


class InferenceLoopStrategyFactory:
    """Factory for creating inference loop strategy instances.

    Provides a static method to instantiate loop strategies by name,
    automatically pruning kwargs to match the target class constructor.
    """

    @staticmethod
    def _init_obj(new_obj_type: type, **kwargs):
        pruned_kwargs = InferenceLoopStrategyFactory._prune_kwargs(
            new_obj_type, **kwargs
        )

        return new_obj_type(**pruned_kwargs)

    @staticmethod
    def _prune_kwargs(new_obj_type: type, **kwargs):
        mthd_sig = inspect.signature(new_obj_type.__init__)
        param_names = set(mthd_sig.parameters)

        bad_kwargs = [key for key in kwargs.keys() if key not in param_names]
        for bad_kwarg in bad_kwargs:
            del kwargs[bad_kwarg]

        return kwargs

    @staticmethod
    def create_loop_strategy(strategy_name: str, **kwargs):
        lower_strat_name = strategy_name.lower()

        if lower_strat_name == "BasicInferenceLoopStrategy".lower():
            return InferenceLoopStrategyFactory._init_obj(
                BasicInferenceLoopStrategy, **kwargs
            )
        else:
            raise ValueError(f"Invalid sequence loop strategy: {strategy_name}")
