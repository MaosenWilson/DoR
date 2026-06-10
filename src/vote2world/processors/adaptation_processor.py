"""Generation-only adaptation processor compatible with RLVR-World layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vote2world.data.leakage import assert_no_future_gt


@dataclass(frozen=True)
class ProcessorConfig:
    context_length: int = 4
    action_dim: int = 13
    action_bins: int = 256
    visual_token_num: int = 4375
    tokens_per_frame: int = 320
    bos_token_id: int = 4631
    eos_token_id: int = 4632

    @property
    def history_length(self) -> int:
        return (self.tokens_per_frame + self.action_dim) * self.context_length

    @property
    def gen_input_length(self) -> int:
        return self.history_length + 1

    @property
    def gen_output_length(self) -> int:
        return self.tokens_per_frame + 1


class VisualTokenizer(Protocol):
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Return visual tokens with shape (B,T,tokens_per_frame)."""


class DummyVisualTokenizer:
    """Deterministic tokenizer for local input-shape tests.

    It is not a model tokenizer. It lets CI verify token layout without loading
    RT-1 weights.
    """

    def __init__(self, tokens_per_frame: int = 320, visual_token_num: int = 4375):
        self.tokens_per_frame = tokens_per_frame
        self.visual_token_num = visual_token_num

    def encode(self, frames: np.ndarray) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim != 5:
            raise ValueError(f"frames must be (B,T,C,H,W), got {arr.shape}")
        b, t = arr.shape[:2]
        base = np.arange(self.tokens_per_frame, dtype=np.int64) % self.visual_token_num
        return np.broadcast_to(base, (b, t, self.tokens_per_frame)).copy()

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        arr = np.asarray(tokens)
        leading = arr.shape[:-1]
        return np.zeros((*leading, 3, 256, 320), dtype=np.float32)


class GenerationOnlyAdaptationProcessor:
    """Build `gen_input_ids` from 4 context frames and 4 actions.

    This intentionally does not create labels or target tokens.
    """

    def __init__(
        self,
        visual_tokenizer: VisualTokenizer,
        action_ranges: np.ndarray,
        config: ProcessorConfig | None = None,
    ) -> None:
        self.config = config or ProcessorConfig()
        self.visual_tokenizer = visual_tokenizer
        ranges = np.asarray(action_ranges, dtype=np.float32)
        if ranges.shape != (self.config.action_dim, 2):
            raise ValueError(f"action_ranges must be ({self.config.action_dim},2), got {ranges.shape}")
        self.action_ranges = ranges

    def discretize_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        original_shape = actions.shape
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"actions last dim must be {self.config.action_dim}, got {actions.shape}")
        flat = actions.reshape(-1, self.config.action_dim)
        min_values = self.action_ranges[:, 0]
        max_values = self.action_ranges[:, 1]
        normalized = np.clip((flat - min_values) / (max_values - min_values + 1e-8), 0.0, 1.0)
        tokens = np.floor(normalized * self.config.action_bins).astype(np.int64)
        tokens = np.clip(tokens, 0, self.config.action_bins - 1)
        return tokens.reshape(original_shape)

    def __call__(self, batch: dict[str, object]) -> dict[str, object]:
        assert_no_future_gt(batch, path_name="adaptation_processor")
        frames = np.asarray(batch["context_frames"], dtype=np.float32)
        actions = np.asarray(batch["context_actions"], dtype=np.float32)
        if frames.ndim == 4:
            frames = frames[None, ...]
        if actions.ndim == 2:
            actions = actions[None, ...]
        if frames.shape[1] != self.config.context_length:
            raise ValueError(f"context frame count must be {self.config.context_length}, got {frames.shape}")
        if actions.shape[1:] != (self.config.context_length, self.config.action_dim):
            raise ValueError(
                f"context actions must be (B,{self.config.context_length},{self.config.action_dim}), got {actions.shape}"
            )

        visual_tokens = np.asarray(self.visual_tokenizer.encode(frames), dtype=np.int64)
        if visual_tokens.shape != (frames.shape[0], self.config.context_length, self.config.tokens_per_frame):
            raise ValueError(f"visual tokenizer returned wrong shape: {visual_tokens.shape}")
        action_tokens = self.discretize_actions(actions) + self.config.visual_token_num
        hist_tokens = np.concatenate([visual_tokens, action_tokens], axis=-1).reshape(frames.shape[0], -1)
        bos = np.full((frames.shape[0], 1), self.config.bos_token_id, dtype=np.int64)
        gen_input_ids = np.concatenate([hist_tokens, bos], axis=-1)
        attention_mask = np.ones_like(gen_input_ids, dtype=np.float32)
        position_ids = np.arange(gen_input_ids.shape[-1], dtype=np.int64)[None, :].repeat(frames.shape[0], axis=0)
        if gen_input_ids.shape[-1] != self.config.gen_input_length:
            raise ValueError(f"gen input length mismatch: {gen_input_ids.shape[-1]}")
        return {
            "gen_input_ids": gen_input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "metadata": {
                "sample_id": batch.get("sample_id"),
                "episode_id": batch.get("episode_id"),
                "start_index": batch.get("start_index"),
                "history_length": self.config.history_length,
                "gen_output_length": self.config.gen_output_length,
            },
        }

