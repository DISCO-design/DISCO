"""Optional coordinate trajectory dumping for DISCO guidance runs.

DISCO already returns denoised and noised coordinate trajectories from the
co-generation loop.  They are large, so this module writes them only when
explicitly enabled.  The default lightweight diagnostics remain JSONL/CSV;
full tensor trajectories are stored as `.pt` files with a compact JSON manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TrajectoryTensorSpec:
    """Metadata for one saved trajectory tensor."""

    key: str
    path: str
    shape: list[int]
    dtype: str
    numel: int
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tensor_to_dtype(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Move a tensor to CPU and cast to the requested storage dtype."""

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            "trajectory dtype must be one of float32, float16, or bfloat16"
        )
    return tensor.detach().cpu().to(dtype=dtype_map[dtype_name])


def maybe_truncate_steps(
    tensor: torch.Tensor,
    *,
    max_steps: int | None,
) -> tuple[torch.Tensor, bool]:
    """Optionally keep only the first `max_steps` trajectory states."""

    if max_steps is None or max_steps <= 0 or tensor.ndim == 0:
        return tensor, False
    if tensor.shape[0] <= max_steps:
        return tensor, False
    return tensor[:max_steps], True


def save_tensor(
    *,
    path: Path,
    key: str,
    tensor: torch.Tensor,
    dtype_name: str,
    max_steps: int | None,
) -> tuple[TrajectoryTensorSpec, bool]:
    """Save one tensor and return manifest metadata plus truncation status."""

    tensor, truncated = maybe_truncate_steps(tensor, max_steps=max_steps)
    tensor = tensor_to_dtype(tensor, dtype_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    size_bytes = tensor.numel() * tensor.element_size()
    return (
        TrajectoryTensorSpec(
            key=key,
            path=str(path),
            shape=list(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            numel=int(tensor.numel()),
            size_bytes=int(size_bytes),
        ),
        truncated,
    )


def tensor_or_none(prediction: dict[str, Any], key: str) -> torch.Tensor | None:
    value = prediction.get(key)
    return value if torch.is_tensor(value) else None


def write_coordinate_trajectory_outputs(
    *,
    trajectory_dir: Path,
    sample_id: str,
    prediction: dict[str, Any],
    save_x1_denoised: bool = True,
    save_xt_trajectory: bool = True,
    save_final_coordinate: bool = False,
    dtype_name: str = "float32",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Write optional coordinate trajectory tensors and a JSON manifest."""

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    saved: list[TrajectoryTensorSpec] = []
    truncated_keys: list[str] = []

    candidates = [
        ("x1_denoised", save_x1_denoised),
        ("xt_trajectory", save_xt_trajectory),
        ("coordinate", save_final_coordinate),
    ]
    for key, enabled in candidates:
        if not enabled:
            continue
        tensor = tensor_or_none(prediction, key)
        if tensor is None:
            continue
        filename = f"{sample_id}_{key}.pt"
        spec, truncated = save_tensor(
            path=trajectory_dir / filename,
            key=key,
            tensor=tensor,
            dtype_name=dtype_name,
            max_steps=max_steps if key != "coordinate" else None,
        )
        saved.append(spec)
        if truncated:
            truncated_keys.append(key)

    noise_levels = tensor_or_none(prediction, "trajectory_noise_levels")
    noise_level_values = (
        []
        if noise_levels is None
        else noise_levels.detach().cpu().reshape(-1).tolist()
    )
    if max_steps is not None and max_steps > 0:
        noise_level_values = noise_level_values[:max_steps]

    manifest_path = trajectory_dir / f"{sample_id}_trajectory_manifest.json"
    manifest = {
        "sample_id": sample_id,
        "trajectory_dir": str(trajectory_dir),
        "manifest_path": str(manifest_path),
        "dtype": dtype_name,
        "max_steps": "" if max_steps is None else max_steps,
        "num_saved_tensors": len(saved),
        "saved_tensors": [spec.as_dict() for spec in saved],
        "trajectory_noise_levels": noise_level_values,
        "truncated_keys": truncated_keys,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "trajectory_manifest": str(manifest_path),
        "trajectory_num_saved_tensors": len(saved),
        "trajectory_tensor_paths": [spec.path for spec in saved],
        "trajectory_truncated_keys": ",".join(truncated_keys),
    }
