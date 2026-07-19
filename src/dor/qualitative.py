"""Auditable utilities for qualitative video-world-model figures.

Scene selection is deliberately ground-truth-only.  Model predictions, rewards,
and evaluation metrics must not enter this module; this prevents qualitative
examples from being selected because a proposed method happens to look better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneCandidate:
    episode: str
    start: int
    motion: float
    feature: np.ndarray


def _as_float_thwc(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim != 4:
        raise ValueError(f"frames must be rank four, got {array.shape}")
    if array.shape[-1] == 3:
        thwc = array
    elif array.shape[1] == 3:
        thwc = np.moveaxis(array, 1, -1)
    else:
        raise ValueError(f"frames must be THWC or TCHW RGB, got {array.shape}")
    thwc = thwc.astype(np.float32, copy=False)
    if thwc.size and float(np.nanmax(thwc)) > 1.5:
        thwc = thwc / 255.0
    return np.clip(thwc, 0.0, 1.0)


def temporal_motion(frames: np.ndarray) -> float:
    """Mean absolute RGB change between adjacent ground-truth frames."""
    thwc = _as_float_thwc(frames)
    if thwc.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(thwc, axis=0)).mean())


def scene_feature(frames: np.ndarray, grid: tuple[int, int] = (8, 10)) -> np.ndarray:
    """Compact GT-only appearance descriptor for deterministic diversity selection.

    The descriptor concatenates uniformly sampled RGB thumbnails from the first,
    middle, and final frame.  It is not a learned semantic representation and is
    never used as an evaluation metric.
    """
    thwc = _as_float_thwc(frames)
    if thwc.shape[0] == 0:
        raise ValueError("frames cannot be empty")
    temporal = np.unique(np.asarray([0, thwc.shape[0] // 2, thwc.shape[0] - 1]))
    rows = np.linspace(0, thwc.shape[1] - 1, grid[0]).round().astype(int)
    cols = np.linspace(0, thwc.shape[2] - 1, grid[1]).round().astype(int)
    sampled = thwc[temporal][:, rows][:, :, cols]
    color_stats = np.concatenate(
        [thwc.mean(axis=(0, 1, 2)), thwc.std(axis=(0, 1, 2))]
    )
    return np.concatenate([sampled.reshape(-1), color_stats]).astype(np.float32)


def select_diverse_scenes(
    candidates: list[SceneCandidate],
    count: int,
    *,
    motion_weight: float = 0.10,
) -> list[SceneCandidate]:
    """Select visually distinct, dynamic scenes without consulting predictions.

    Selection starts from the highest-motion episode and then applies deterministic
    farthest-first traversal in standardized GT appearance space.  A small normalized
    motion term breaks near-ties in favor of trajectories with visible state change.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if len(candidates) < count:
        raise ValueError(f"need at least {count} candidates, got {len(candidates)}")
    if not 0.0 <= motion_weight <= 1.0:
        raise ValueError("motion_weight must lie in [0,1]")

    ordered = sorted(candidates, key=lambda item: (item.episode, item.start))
    features = np.stack([np.asarray(item.feature, dtype=np.float64) for item in ordered])
    if features.ndim != 2:
        raise ValueError("all scene features must be one-dimensional and equal length")
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    features = (features - features.mean(axis=0)) / scale
    motions = np.asarray([item.motion for item in ordered], dtype=np.float64)
    motion_span = float(np.ptp(motions))
    normalized_motion = (
        (motions - motions.min()) / motion_span if motion_span > 1e-12 else np.zeros_like(motions)
    )

    first = int(np.argmax(motions))
    selected = [first]
    while len(selected) < count:
        distance = np.full(len(ordered), np.inf, dtype=np.float64)
        for index in selected:
            squared = np.mean((features - features[index]) ** 2, axis=1)
            distance = np.minimum(distance, squared)
        score = (1.0 - motion_weight) * distance + motion_weight * normalized_motion
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return [ordered[index] for index in selected]


def absolute_residual(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-pixel RGB absolute residual in [0,1], with no per-image renormalization."""
    gt_array = _as_float_thwc(gt)
    pred_array = _as_float_thwc(prediction)
    if gt_array.shape != pred_array.shape:
        raise ValueError(f"shape mismatch: {gt_array.shape} vs {pred_array.shape}")
    return np.abs(gt_array - pred_array)
