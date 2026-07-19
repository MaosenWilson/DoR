"""Data adapter for the public action-conditioned BAIR iVideoGPT model.

BAIR robot pushing frames are already 64x64 with 4-dim actions, so this adapter is
the thinnest of the iVideoGPT family: it reuses the shared VP2 token layout and only
handles the BAIR npz schema (``image`` [T,64,64,3], ``action`` [T,4]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dor.adapters.ivideogpt_vp2 import CONTEXT_LENGTH, VP2Window, load_ivideogpt


BAIR_ACTION_DIM = 4
BAIR_CONTEXT_LENGTH = 2


def load_bair_window_npz(
    path: str | Path,
    *,
    start: int = 0,
    horizon: int = 15,
    resolution: int = 64,
    device: torch.device | str = "cpu",
) -> VP2Window:
    """Load one contiguous BAIR window. BAIR stores T frames and T actions aligned;
    the action model reads action ``t`` to predict frame ``t+1``, and discards the
    last action row (no successor). Frames are already 64x64."""
    if start < 0 or horizon < 1:
        raise ValueError("start must be non-negative and horizon must be positive")
    length = BAIR_CONTEXT_LENGTH + int(horizon)
    with np.load(path, allow_pickle=False) as payload:
        images = np.asarray(payload["image"])
        actions = np.asarray(payload["action"])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB images, got {images.shape}")
    if actions.ndim != 2 or actions.shape[1] != BAIR_ACTION_DIM:
        raise ValueError(f"expected {BAIR_ACTION_DIM}-dim actions, got {actions.shape}")
    if start + length > images.shape[0]:
        raise ValueError(
            f"window [{start}, {start + length}) exceeds trajectory length {images.shape[0]}"
        )
    image_window = images[start:start + length]
    action_window = actions[start:start + length].astype(np.float32)
    frames = torch.from_numpy(image_window).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(frames.shape[-2:]) != (resolution, resolution):
        frames = F.interpolate(frames, size=(resolution, resolution),
                               mode="bilinear", align_corners=False, antialias=True)
    return VP2Window(
        episode=Path(path).stem,
        start=int(start),
        frames=frames.to(device),
        actions=torch.from_numpy(action_window).to(device),
    )


def bair_windows_from_trajectory(path, horizon, *, stride=2, device="cpu"):
    """Slice one BAIR trajectory npz into overlapping fixed-horizon windows.

    Lets a single trajectory yield many contexts for a preliminary audit before any
    bulk data download.
    """
    with np.load(path, allow_pickle=False) as payload:
        total = int(np.asarray(payload["image"]).shape[0])
    length = BAIR_CONTEXT_LENGTH + int(horizon)
    starts = list(range(0, total - length + 1, max(1, int(stride))))
    return [load_bair_window_npz(path, start=s, horizon=horizon, device=device) for s in starts]


def load_bair_ivideogpt(upstream_root, checkpoint_dir, *, horizon=15, device="cuda"):
    """Load the released 64x64 BAIR checkpoint with its four-action head."""
    if CONTEXT_LENGTH != BAIR_CONTEXT_LENGTH:
        raise RuntimeError("shared iVideoGPT token layout no longer uses two context frames")
    return load_ivideogpt(upstream_root, checkpoint_dir, horizon=horizon,
                          action_dim=BAIR_ACTION_DIM, device=device)
