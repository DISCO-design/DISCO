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

from abc import ABC, abstractmethod

import numpy as np


class SequenceInferenceNoiseScheduler(ABC):
    """Abstract base class for sequence noise scheduling during inference.

    Maps structure inference iteration numbers to sequence noise levels (t values
    between 0 and 1), controlling the masking schedule for sequence generation.
    Subclasses implement different functional forms for the noise-to-time mapping.
    """

    def __init__(self):
        self.num_structure_inference_steps = None

    def _get_perc_time_done(self, struct_inf_iter_num: int) -> float:
        """Calculates the completion percentage for the current iteration.

        Args:
            struct_inf_iter_num: Current structure inference iteration number.

        Returns:
            Fraction of total steps completed, in [0, 1]. Returns 0.0 for
            negative iteration numbers.
        """
        if struct_inf_iter_num < 0:
            return 0.0

        return float(struct_inf_iter_num + 1) / self.num_structure_inference_steps

    def start(self, num_structure_inference_steps: int) -> None:
        """Initializes the scheduler with the total number of inference steps.

        Args:
            num_structure_inference_steps: Total number of structure inference
                steps. Must be positive.
        """
        self.num_structure_inference_steps = num_structure_inference_steps
        assert self.num_structure_inference_steps > 0

    def reset(self) -> None:
        """Resets the scheduler state, clearing the stored step count."""
        self.num_structure_inference_steps = None

    @abstractmethod
    def get_t(self, struct_inf_iter_num: int) -> float:
        pass


class PolynomialSequenceInferenceNoiseScheduler(SequenceInferenceNoiseScheduler):
    """Polynomial noise schedule: t_noise = 1 - (perc_time ^ power).

    Produces a decreasing noise level from 1 to 0 as inference progresses,
    with the rate of decrease controlled by the power parameter.
    """

    def __init__(self, power: float = 1.0):
        super().__init__()
        self.power = power
        assert power > 0.0

    def get_t(self, struct_inf_iter_num: int) -> float:
        pre_t = self._get_perc_time_done(struct_inf_iter_num)
        return 1.0 - (pre_t**self.power)


class CosineSequenceInferenceNoiseScheduler(SequenceInferenceNoiseScheduler):
    """Cosine noise schedule: t_noise = cos(pi * perc_time / 2) ^ power.

    Produces a smooth cosine-shaped decrease in noise level from 1 to 0,
    with the sharpness of the transition controlled by the power parameter.
    """

    def __init__(self, power: float = 1.0):
        super().__init__()
        self.power = power
        assert power > 0.0

    def get_t(self, struct_inf_iter_num: int) -> float:
        pre_t = self._get_perc_time_done(struct_inf_iter_num)
        return np.cos(np.pi * pre_t / 2.0) ** self.power


class PowerRatioSequenceInferenceNoiseScheduler(SequenceInferenceNoiseScheduler):
    """Power ratio noise schedule: t_noise = (base^perc_time - 1) / (base - 1).

    Produces an exponentially increasing noise level from 0 to 1 as inference
    progresses, controlled by the base parameter.
    """

    def __init__(self, base: float = 10.0):
        super().__init__()
        self.base = base
        assert base > 0.0

    def get_t(self, struct_inf_iter_num: int) -> float:
        pre_t = self._get_perc_time_done(struct_inf_iter_num)
        return ((self.base**pre_t) - 1) / (self.base - 1)
