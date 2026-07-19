"""Exact data adapter for the public action-conditioned RoboNet iVideoGPT model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dor.adapters.ivideogpt_vp2 import CONTEXT_LENGTH, VP2Window, load_ivideogpt


ROBONET_ACTION_DIM = 5
ROBONET_CONTEXT_LENGTH = 2
ROBONET_DEFAULT_HORIZON = 10


def _center_crop_resize(frames: torch.Tensor, resolution: int) -> torch.Tensor:
    """Match the public iVideoGPT RoboNet evaluation transform."""
    if frames.ndim != 4:
        raise ValueError(f"expected TCHW frames, got {tuple(frames.shape)}")
    height, width = frames.shape[-2:]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    frames = frames[..., top:top + side, left:left + side]
    if side != resolution:
        frames = F.interpolate(
            frames,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return frames


def load_robonet_window_npz(
    path: str | Path,
    *,
    start: int = 0,
    horizon: int = ROBONET_DEFAULT_HORIZON,
    resolution: int = 64,
    device: torch.device | str = "cpu",
) -> VP2Window:
    """Load one contiguous RoboNet window with upstream action alignment.

    RoboNet stores one transition action fewer than frames. The public iVideoGPT
    loader appends a zero action row; the action model discards that final row.
    We append after slicing so a window beginning at ``start`` preserves the
    transition from every retained frame to its successor.
    """
    if start < 0 or horizon < 1:
        raise ValueError("start must be non-negative and horizon must be positive")
    length = ROBONET_CONTEXT_LENGTH + int(horizon)
    with np.load(path, allow_pickle=False) as payload:
        images = np.asarray(payload["image"])
        actions = np.asarray(payload["action"])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB images, got {images.shape}")
    if actions.ndim != 2 or actions.shape[1] != ROBONET_ACTION_DIM:
        raise ValueError(f"expected five-dimensional actions, got {actions.shape}")
    if actions.shape[0] != images.shape[0] - 1:
        raise ValueError(
            f"RoboNet requires T frames and T-1 actions, got {images.shape[0]} and {actions.shape[0]}"
        )
    if start + length > images.shape[0]:
        raise ValueError(
            f"window [{start}, {start + length}) exceeds trajectory length {images.shape[0]}"
        )

    image_window = images[start:start + length]
    action_window = actions[start:start + length - 1]
    action_window = np.concatenate(
        [action_window, np.zeros((1, ROBONET_ACTION_DIM), dtype=actions.dtype)], axis=0
    )
    frames = torch.from_numpy(image_window).permute(0, 3, 1, 2).float().div_(255.0)
    frames = _center_crop_resize(frames, int(resolution))
    return VP2Window(
        episode=Path(path).stem,
        start=int(start),
        frames=frames.to(device),
        actions=torch.from_numpy(action_window).float().to(device),
    )


def load_robonet_ivideogpt(
    upstream_root: str | Path,
    checkpoint_dir: str | Path,
    *,
    horizon: int = ROBONET_DEFAULT_HORIZON,
    device: torch.device | str = "cuda",
):
    """Load the released 64x64 RoboNet checkpoint with its five-action head."""
    if CONTEXT_LENGTH != ROBONET_CONTEXT_LENGTH:
        raise RuntimeError("shared iVideoGPT token layout no longer uses two context frames")
    return load_ivideogpt(
        upstream_root,
        checkpoint_dir,
        horizon=horizon,
        action_dim=ROBONET_ACTION_DIM,
        device=device,
    )
