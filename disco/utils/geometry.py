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

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def angle_3p(a, b, c):
    """
    Calculate the angle between three points in a 2D space.

    Args:
        a (list or array-like): The coordinates of the first point.
        b (list or array-like): The coordinates of the second point.
        c (list or array-like): The coordinates of the third point.

    Returns:
        float: The angle in degrees (0, 180) between the vectors
               from point a to point b and point b to point c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    cos_theta = np.clip(dot_product / (norm_ab * norm_bc + 1e-4), -1, 1)
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees


def random_transform(
    points, max_translation=1.0, apply_augmentation=False, centralize=True
) -> np.ndarray:
    """
    Randomly transform a set of 3D points.

    Args:
        points (numpy.ndarray): The points to be transformed, shape=(N, 3)
        max_translation (float): The maximum translation value. Default is 1.0.
        apply_augmentation (bool): Whether to apply random rotation/translation on ref_pos

    Returns:
        numpy.ndarray: The transformed points.
    """
    if centralize:
        points = points - points.mean(axis=0)
    if not apply_augmentation:
        return points
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    R = Rotation.random().as_matrix()
    transformed_points = np.dot(points + translation, R.T)
    return transformed_points


class DistOneHotCalculator:
    """Converts pairwise distances between representative atoms into one-hot encoded distance bins.

    Args:
        dist_bin_min: Minimum distance for the bin range.
        dist_bin_max: Maximum distance for the bin range.
        num_bins: Total number of distance bins (including overflow).
    """

    def __init__(
        self,
        dist_bin_min: float,
        dist_bin_max: float,
        num_bins: int,
    ):
        self.bins = torch.linspace(
            dist_bin_min,
            dist_bin_max,
            steps=num_bins - 1,
        ).pow(2)

        self.squared_bin_max = dist_bin_max**2
        self.num_bins = num_bins

    def get_centre_dist_one_hot(
        self, atom_positions: torch.Tensor, input_feature_dict: dict
    ):
        """Computes one-hot encoded distance bins for representative atoms.

        Selects representative atom positions using the distogram mask from
        ``input_feature_dict``, computes squared pairwise distances, bins them,
        and returns one-hot vectors masked to exclude self-interactions and
        distances beyond the maximum bin.

        Args:
            atom_positions: Atom coordinates of shape ``(N_atoms, 3)`` or
                ``(batch, N_atoms, 3)``.
            input_feature_dict: Feature dictionary containing
                ``"distogram_rep_atom_mask"`` and ``"residue_index"`` keys.

        Returns:
            One-hot encoded distance bins of shape
            ``(..., N_tokens, N_tokens, num_bins)``, masked by distance and
            non-self-interaction constraints.
        """
        if self.bins.device != atom_positions.device:
            self.bins = self.bins.to(device=atom_positions.device)

        rep_atom_positions = None
        rep_atom_mask = input_feature_dict["distogram_rep_atom_mask"].to(
            dtype=torch.bool
        )
        match atom_positions.ndim:
            case 2:
                rep_atom_positions = atom_positions[rep_atom_mask]
            case 3:
                rep_atom_positions = atom_positions[:, rep_atom_mask]
            case _:
                raise ValueError(
                    f"Had invalid atom_positions ndim of {atom_positions.ndim}"
                )

        diffs = rep_atom_positions.unsqueeze(dim=-3) - rep_atom_positions.unsqueeze(
            dim=-2
        )
        squared_l2 = diffs.pow(2).sum(dim=-1)

        n_tokens = input_feature_dict["residue_index"].shape[-1]
        not_same_res_mask = ~torch.diag(
            torch.ones((n_tokens,), device=atom_positions.device, dtype=torch.bool)
        )

        if atom_positions.ndim == 3:
            not_same_res_mask = not_same_res_mask.unsqueeze(0).repeat(
                len(atom_positions), 1, 1
            )

        mask = (squared_l2 < self.squared_bin_max) & not_same_res_mask

        bins = self.bins
        while bins.ndim < squared_l2.ndim + 1:
            bins = bins.unsqueeze(dim=0)

        dist_bins = (bins < squared_l2.unsqueeze(dim=-1)).sum(dim=-1)

        one_hots = F.one_hot(dist_bins, num_classes=self.num_bins).to(
            dtype=atom_positions.dtype
        )

        return one_hots * mask[..., None]
