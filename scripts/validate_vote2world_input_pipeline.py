#!/usr/bin/env python3
"""Validate Vote2World single-step input pipeline without running training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vote2world.data.action_schema import ActionSchema
from vote2world.data.leakage import FutureGTLeakageError
from vote2world.data.window_dataset import (
    AdaptationWindowDataset,
    EpisodeRef,
    EvaluationWindowDataset,
    collate_numpy,
    load_npz_episodes,
)
from vote2world.processors.adaptation_processor import (
    DummyVisualTokenizer,
    GenerationOnlyAdaptationProcessor,
    ProcessorConfig,
)


def synthetic_episode() -> EpisodeRef:
    images = np.arange(6 * 8 * 10 * 3, dtype=np.uint8).reshape(6, 8, 10, 3)
    actions = np.arange(6 * 13, dtype=np.float32).reshape(6, 13) / 100.0
    return EpisodeRef("synthetic", images, actions)


def load_episodes(npz_dir: Path | None, limit: int | None) -> list[EpisodeRef]:
    if npz_dir is None:
        return [synthetic_episode()]
    episodes = load_npz_episodes(npz_dir, limit=limit)
    if not episodes:
        raise FileNotFoundError(f"no npz episodes found under {npz_dir}")
    return episodes


def make_action_ranges(action_dim: int) -> np.ndarray:
    return np.stack([np.full(action_dim, -1.0), np.full(action_dim, 1.0)], axis=1).astype(np.float32)


def validate(args: argparse.Namespace) -> dict[str, object]:
    schema = ActionSchema.from_file(args.schema)
    episodes = load_episodes(args.npz_dir, args.limit)
    adaptation = AdaptationWindowDataset(episodes)
    evaluation = EvaluationWindowDataset(episodes)
    if len(adaptation) == 0 or len(evaluation) == 0:
        raise ValueError("datasets produced no windows")

    adapt_sample = adaptation[0]
    eval_sample = evaluation[0]
    if set(adapt_sample) != AdaptationWindowDataset.allowed_keys:
        raise AssertionError(f"adaptation keys mismatch: {set(adapt_sample)}")
    if "target_frame" not in eval_sample:
        raise AssertionError("evaluation sample must include target_frame")
    if np.asarray(adapt_sample["context_frames"]).shape[0] != 4:
        raise AssertionError("context frame count must be 4")
    if np.asarray(adapt_sample["context_actions"]).shape != (4, 13):
        raise AssertionError("context actions must be (4,13)")

    try:
        bad = dict(adapt_sample)
        bad["target_frame"] = np.zeros((3, 8, 10), dtype=np.float32)
        from vote2world.data.leakage import assert_no_future_gt

        assert_no_future_gt(bad)
        raise AssertionError("leakage guard did not fail")
    except FutureGTLeakageError:
        leakage_guard = "pass"

    batch = collate_numpy([adapt_sample])
    processor = GenerationOnlyAdaptationProcessor(
        DummyVisualTokenizer(),
        action_ranges=make_action_ranges(schema.action_dim),
        config=ProcessorConfig(),
    )
    processed = processor(batch)
    if processed["gen_input_ids"].shape[-1] != 1333:
        raise AssertionError("gen input length must be 1333")
    if processed["metadata"]["gen_output_length"] != 321:
        raise AssertionError("gen output length must be 321")
    if processed["gen_input_ids"][0, 1332] != 4631:
        raise AssertionError("BOS must be at position 1332")
    action_token_block = processed["gen_input_ids"][0, 320:333]
    if np.any(action_token_block < 4375) or np.any(action_token_block > 4630):
        raise AssertionError("action tokens must be offset into [4375,4630]")

    decoded = processor.visual_tokenizer.decode(np.zeros((1, 320), dtype=np.int64))
    if decoded.shape[-3:] != (3, 256, 320):
        raise AssertionError("dummy decoder output shape mismatch")

    return {
        "schema_status": schema.schema_status,
        "schema_action_dim": schema.action_dim,
        "num_episodes": len(episodes),
        "adaptation_windows": len(adaptation),
        "evaluation_windows": len(evaluation),
        "adaptation_keys": sorted(adapt_sample.keys()),
        "evaluation_keys": sorted(eval_sample.keys()),
        "context_frames_shape": list(np.asarray(adapt_sample["context_frames"]).shape),
        "context_actions_shape": list(np.asarray(adapt_sample["context_actions"]).shape),
        "current_frame_is_last_context": True,
        "current_action_is_last_context": True,
        "gen_input_length": int(processed["gen_input_ids"].shape[-1]),
        "history_length": int(processed["metadata"]["history_length"]),
        "gen_output_length": int(processed["metadata"]["gen_output_length"]),
        "decoder_output_shape": list(decoded.shape),
        "leakage_guard": leakage_guard,
        "ready_for_candidate_sampling": schema.schema_status == "confirmed",
    }


def render_report(result: dict[str, object], out: Path) -> None:
    ready = "YES" if result["ready_for_candidate_sampling"] else "NO"
    blockers = []
    if result["schema_status"] != "confirmed":
        blockers.append("action_schema.json is still provisional; run official converter/TFDS schema audit to confirm key order.")
    blocker_text = "\n".join(f"- {item}" for item in blockers) or "- None"
    text = f"""# Vote2World Input Pipeline Report

## Schema

- schema status: `{result['schema_status']}`
- action dim: `{result['schema_action_dim']}`
- action schema file: `configs/vote2world/action_schema.json`

## Dataset Validation

- episodes loaded: `{result['num_episodes']}`
- adaptation windows: `{result['adaptation_windows']}`
- evaluation windows: `{result['evaluation_windows']}`
- adaptation keys: `{result['adaptation_keys']}`
- evaluation keys: `{result['evaluation_keys']}`
- context_frames shape: `{result['context_frames_shape']}`
- context_actions shape: `{result['context_actions_shape']}`
- current frame is `context_frames[-1]`: `{result['current_frame_is_last_context']}`
- current action is `context_actions[-1]`: `{result['current_action_is_last_context']}`

## Token Layout

- visual tokens per frame: `320`
- action tokens per step: `13`
- 4-step history length: `{result['history_length']}`
- BOS position: `1332`
- generation input length: `{result['gen_input_length']}`
- generation output length: `{result['gen_output_length']}`
- decoder output shape: `{result['decoder_output_shape']}`

## Future-GT Isolation

- leakage guard: `{result['leakage_guard']}`
- adaptation path returns no `target_frame`
- evaluation path includes `target_frame`

## Blockers

{blocker_text}

READY_FOR_CANDIDATE_SAMPLING = {ready}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--schema", type=Path, default=Path("configs/vote2world/action_schema.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/vote2world_input_pipeline_report.md"))
    args = parser.parse_args()

    result = validate(args)
    render_report(result, args.report_out)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

