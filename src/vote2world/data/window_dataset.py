"""Single-step Vote2World dataset windows.

The production path reads RLVR-World converted `.npz` episodes. Tests and
small probes can use in-memory episodes with the same image/action arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from vote2world.data.leakage import assert_no_future_gt


CONTEXT_LENGTH = 4
SEGMENT_LENGTH = 5
ACTION_DIM = 13
DEFAULT_IMAGE_KEY = "image"


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    images: np.ndarray
    actions: np.ndarray


def _to_chw_float(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"frames must have shape (T,H,W,C) or (T,C,H,W), got {arr.shape}")
    if arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.shape[1] not in (1, 3):
        raise ValueError(f"cannot infer frame channel axis from {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.size and arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def _validate_episode(images: np.ndarray, actions: np.ndarray, episode_id: str) -> None:
    if len(images) != len(actions):
        raise ValueError(f"{episode_id}: image/action lengths differ: {len(images)} vs {len(actions)}")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{episode_id}: actions must have shape (T,13), got {actions.shape}")


def load_npz_episodes(
    data_dir: str | Path,
    *,
    image_key: str = DEFAULT_IMAGE_KEY,
    limit: int | None = None,
) -> list[EpisodeRef]:
    files = sorted(Path(data_dir).glob("*.npz"))
    if limit is not None:
        files = files[:limit]
    episodes: list[EpisodeRef] = []
    for path in files:
        with np.load(path) as data:
            if image_key not in data:
                raise KeyError(f"{path} missing image key {image_key!r}; keys={list(data.keys())}")
            if "action" not in data:
                raise KeyError(f"{path} missing action key")
            images = np.asarray(data[image_key])
            actions = np.asarray(data["action"], dtype=np.float32)
        _validate_episode(images, actions, path.stem)
        episodes.append(EpisodeRef(path.stem, images, actions))
    return episodes


def make_sample_id(episode_id: str, start_index: int) -> str:
    return f"{episode_id}:{start_index}"


class BaseWindowDataset:
    def __init__(
        self,
        episodes: Iterable[EpisodeRef],
        *,
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        self.episodes = list(episodes)
        self.context_length = context_length
        self.index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            _validate_episode(episode.images, episode.actions, episode.episode_id)
            # Need 4 context frames plus o_{t+1} to keep alignment evaluable.
            max_start = len(episode.images) - (context_length + 1)
            for start in range(max_start + 1):
                self.index.append((episode_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def _window(self, item: int) -> tuple[EpisodeRef, int, np.ndarray, np.ndarray]:
        episode_idx, start = self.index[item]
        episode = self.episodes[episode_idx]
        end = start + self.context_length
        frames = _to_chw_float(episode.images[start:end])
        actions = np.asarray(episode.actions[start:end], dtype=np.float32)
        if frames.shape[0] != self.context_length:
            raise ValueError("context frame count mismatch")
        if actions.shape != (self.context_length, ACTION_DIM):
            raise ValueError(f"context actions must be ({self.context_length},{ACTION_DIM}), got {actions.shape}")
        return episode, start, frames, actions


class AdaptationWindowDataset(BaseWindowDataset):
    """Generation-only dataset. It never returns future GT."""

    allowed_keys = {"context_frames", "context_actions", "sample_id", "episode_id", "start_index"}

    def __getitem__(self, item: int) -> dict[str, object]:
        episode, start, frames, actions = self._window(item)
        sample = {
            "context_frames": frames,
            "context_actions": actions,
            "sample_id": make_sample_id(episode.episode_id, start),
            "episode_id": episode.episode_id,
            "start_index": start,
        }
        assert_no_future_gt(sample, path_name="adaptation_dataset")
        return sample


class EvaluationWindowDataset(BaseWindowDataset):
    """Evaluation-only dataset. It includes o_{t+1} target_frame."""

    def __getitem__(self, item: int) -> dict[str, object]:
        episode, start, frames, actions = self._window(item)
        target_index = start + self.context_length
        target = _to_chw_float(episode.images[target_index : target_index + 1])[0]
        return {
            "context_frames": frames,
            "context_actions": actions,
            "target_frame": target,
            "sample_id": make_sample_id(episode.episode_id, start),
            "episode_id": episode.episode_id,
            "start_index": start,
        }


def collate_numpy(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate empty sample list")
    out: dict[str, object] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], np.ndarray):
            out[key] = np.stack(values)
        else:
            out[key] = values
    return out

