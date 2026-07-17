"""Collect frozen real Breakout windows for the IRIS external gate."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _resize(frame: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR))


def _step(env, action: int, skip: int = 4):
    frames, reward, terminated, truncated, info = [], 0.0, False, False, {}
    for _ in range(skip):
        observation, value, terminated, truncated, info = env.step(int(action))
        frames.append(_resize(observation))
        reward += float(value)
        if terminated or truncated:
            break
    selected = frames[-2:] if len(frames) >= 2 else frames
    return np.maximum.reduce(selected), reward, terminated, truncated, info


def _progress(done: int, total: int, started: float):
    elapsed = time.monotonic() - started
    eta = elapsed / max(done, 1) * (total - done)
    width = 24
    filled = int(width * done / total)
    print(
        f"\r[IRIS cache {'#' * filled}{'-' * (width-filled)}] {done}/{total} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        end="",
        flush=True,
    )
    if done == total:
        print(flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--windows_per_episode", type=int, default=8)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument(
        "--collection_policy",
        choices=("uniform", "checkpoint_actor"),
        default="uniform",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream")
    parser.add_argument("--actor_temperature", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import ale_py
    import gymnasium as gym

    if args.context < 1 or args.episodes < 2 or args.windows_per_episode < 1 or args.window_stride < 1:
        raise ValueError("invalid collection sizes")
    gym.register_envs(ale_py)
    environment_id = f"ALE/{args.game}-v5"
    env = gym.make(
        environment_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    )
    destination = Path(args.cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    entries, total = [], args.episodes * args.windows_per_episode
    action_vocab_size = int(env.action_space.n)
    tokenizer = actor = None
    if args.collection_policy == "checkpoint_actor":
        if not args.checkpoint or not args.upstream:
            raise ValueError("checkpoint_actor requires --checkpoint and --upstream")
        from dor.adapters.iris_atari import (
            load_iris,
            load_iris_actor,
            sample_iris_actor_action,
        )

        tokenizer, unused_world_model = load_iris(
            args.upstream,
            args.checkpoint,
            action_vocab_size=action_vocab_size,
            device=args.device,
        )
        actor = load_iris_actor(
            args.upstream,
            args.checkpoint,
            action_vocab_size=action_vocab_size,
            device=args.device,
        )
        del unused_world_model
        torch.cuda.empty_cache()

        def choose_action(frame):
            tensor = torch.from_numpy(
                np.ascontiguousarray(frame).copy()
            ).permute(2, 0, 1).float().div(255.0).to(args.device)
            return sample_iris_actor_action(
                tokenizer, actor, tensor, temperature=args.actor_temperature
            )
    else:
        def choose_action(_frame):
            return int(rng.integers(0, action_vocab_size)), float("nan")

    def reset_segment(reset_seed: int):
        observation, _ = env.reset(seed=reset_seed)
        if args.collection_policy == "checkpoint_actor":
            # NoopResetEnv(noop_max=1) performs one raw no-op before the outer
            # MaxAndSkip wrapper starts repeated-action stepping.
            observation, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                observation, _ = env.reset(seed=reset_seed + 100_000)
            actor.reset(1)
            return _resize(observation)
        current_frame = _resize(observation)
        current_frame, *_ = _step(env, 1)
        return current_frame

    action_counts = np.zeros(action_vocab_size, dtype=np.int64)
    policy_entropies, frame_delta_mse, episode_returns = [], [], []
    actor_sampling_seed = args.seed + 50_000
    torch.manual_seed(actor_sampling_seed)
    started = time.monotonic()
    for episode in range(args.episodes):
        current = reset_segment(args.seed + episode)
        for _ in range(args.warmup):
            action, entropy = choose_action(current)
            next_frame, value, terminated, truncated, _ = _step(env, action)
            action_counts[action] += 1
            if np.isfinite(entropy):
                policy_entropies.append(entropy)
            frame_delta_mse.append(float(np.mean((next_frame.astype(np.float32) - current.astype(np.float32)) ** 2) / 255.0**2))
            current = next_frame
            if terminated or truncated:
                current = reset_segment(args.seed + 1000 + episode)
        frames = [current]
        actions = []
        collected = 0
        episode_return = 0.0
        while collected < args.windows_per_episode:
            action, entropy = choose_action(current)
            next_frame, value, terminated, truncated, _ = _step(env, action)
            action_counts[action] += 1
            if np.isfinite(entropy):
                policy_entropies.append(entropy)
            frame_delta_mse.append(float(np.mean((next_frame.astype(np.float32) - current.astype(np.float32)) ** 2) / 255.0**2))
            episode_return += value
            actions.append(action)
            frames.append(next_frame)
            current = next_frame
            ready = len(frames) >= args.context + 1
            on_stride = ready and (len(actions) - args.context) % args.window_stride == 0
            if on_stride:
                path = destination / f"episode_{episode:03d}_step_{len(actions):04d}.npz"
                np.savez_compressed(
                    path,
                    frames=np.asarray(frames[-(args.context + 1):], dtype=np.uint8),
                    actions=np.asarray(actions[-args.context:], dtype=np.int64),
                    episode=np.asarray(episode),
                    step=np.asarray(len(actions)),
                )
                entries.append({"episode": episode, "step": len(actions), "window_npz": str(path.resolve())})
                collected += 1
                _progress(len(entries), total, started)
            if terminated or truncated:
                current = reset_segment(args.seed + 2000 + episode + collected)
                frames, actions = [current], []
        episode_returns.append(episode_return)
    env.close()
    manifest = {
        "protocol": "IRIS Atari real-transition windows v1",
        "environment": environment_id,
        "action_vocab_size": action_vocab_size,
        "context": args.context,
        "window_stride": args.window_stride,
        "collection_policy": args.collection_policy,
        "actor_temperature": (
            args.actor_temperature if args.collection_policy == "checkpoint_actor" else None
        ),
        "actor_sampling_seed": (
            actor_sampling_seed if args.collection_policy == "checkpoint_actor" else None
        ),
        "checkpoint": args.checkpoint,
        "upstream": args.upstream,
        "action_histogram": action_counts.tolist(),
        "mean_policy_entropy": (
            float(np.mean(policy_entropies)) if policy_entropies else None
        ),
        "frame_delta_mse": {
            "mean": float(np.mean(frame_delta_mse)),
            "median": float(np.median(frame_delta_mse)),
            "q10": float(np.quantile(frame_delta_mse, 0.1)),
            "q90": float(np.quantile(frame_delta_mse, 0.9)),
        },
        "episode_return": episode_returns,
        "seed": args.seed,
        "entries": entries,
    }
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {output}\nIRIS_ATARI_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
