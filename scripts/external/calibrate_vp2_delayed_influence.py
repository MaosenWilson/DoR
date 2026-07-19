"""Calibrate conservative temporal-credit coefficients from VP2 branch rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dor.adapters.ivideogpt_vp2 import (
    decoded_ground_truth,
    frame_rewards,
    load_ivideogpt,
    load_vp2_window_npz,
    prefix_tokens_through_frame,
    sample_continuations_from_prefixes,
    sample_rollout,
    tokenize_ground_truth,
)
from dor.delayed_influence import (
    delayed_influence_ratio,
    shuffle_continuations_within_group,
)
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics


def _episode_balanced(entries: list[dict], contexts: int) -> list[dict]:
    buckets = {}
    for entry in entries:
        buckets.setdefault(str(entry["episode"]), []).append(entry)
    selected = []
    while len(selected) < contexts:
        progressed = False
        for episode in sorted(buckets):
            if buckets[episode]:
                selected.append(buckets[episode].pop(0))
                progressed = True
                if len(selected) == contexts:
                    break
        if not progressed:
            break
    return selected


def _manifest(path: str | Path, horizon: int, contexts: int) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if int(payload["horizon"]) != int(horizon):
        raise ValueError("calibration manifest horizon does not match --horizon")
    entries = list(payload["entries"])
    if contexts < 1 or contexts > len(entries):
        raise ValueError(f"--contexts must lie in [1,{len(entries)}]")
    return _episode_balanced(entries, contexts), hashlib.sha256(raw).hexdigest()


def _episode_bootstrap(immediate, future, episodes, contexts, args, seed):
    episodes = np.asarray(episodes).astype(str)
    contexts = np.asarray(contexts).astype(str)
    unique = np.unique(episodes)
    if len(unique) < 2:
        raise ValueError("episode bootstrap requires at least two episodes")
    rng = np.random.default_rng(seed)
    real_values, shuffled_values = [], []
    for round_index in range(args.bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices, boot_episode, boot_context = [], [], []
        for occurrence, episode in enumerate(sampled):
            selected = np.flatnonzero(episodes == episode)
            indices.extend(selected.tolist())
            boot_episode.extend([f"{episode}#b{occurrence}"] * len(selected))
            boot_context.extend([f"{value}#b{occurrence}" for value in contexts[selected]])
        indices = np.asarray(indices, dtype=np.int64)
        boot_future = future[indices]
        real = delayed_influence_ratio(
            immediate[indices], boot_future, np.asarray(boot_episode),
            folds=args.folds, ridge=args.ridge, seed=seed + round_index,
        )["coefficient"]
        shuffled = shuffle_continuations_within_group(
            boot_future, np.asarray(boot_context), seed=seed + 100_000 + round_index
        )
        null = delayed_influence_ratio(
            immediate[indices], shuffled, np.asarray(boot_episode),
            folds=args.folds, ridge=args.ridge, seed=seed + round_index,
        )["coefficient"]
        real_values.append(real)
        shuffled_values.append(null)
    return np.asarray(real_values), np.asarray(shuffled_values)


def _discounted_future(frame_reward: np.ndarray, prefix_frames: int, gamma: float) -> np.ndarray:
    future = frame_reward[:, prefix_frames:]
    weights = np.power(float(gamma), np.arange(future.shape[1], dtype=np.float64))
    return future @ weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--prefix_candidates", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--prefix_frames", default="1,2,3,4,5,6,7")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--min_episodes", type=int, default=4)
    parser.add_argument("--min_active_blocks", type=int, default=2)
    parser.add_argument(
        "--coefficient_rule",
        choices=("raw_delta_lcb", "normalized_excess_lcb"),
        default="raw_delta_lcb",
        help=(
            "raw_delta_lcb reproduces v1; normalized_excess_lcb is the "
            "conservative fraction of reliability remaining above shuffled null"
        ),
    )
    parser.add_argument("--seed", type=int, default=18427)
    parser.add_argument("--action_dim", type=int, default=4,
                        help="4 for RoboSuite PushCenter, 5 for RoboDesk")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", required=True)
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="skip generation and recompute statistics from an existing --cache",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prefixes = sorted({int(value) for value in args.prefix_frames.split(",") if value.strip()})
    if not prefixes or prefixes[0] < 1 or prefixes[-1] >= args.horizon:
        raise ValueError("prefix_frames must be non-empty values in [1,horizon-1]")
    if args.prefix_candidates < 2 or args.continuations < 2:
        raise ValueError("prefix_candidates and continuations must be at least two")
    if args.bootstrap < 20:
        raise ValueError("bootstrap must be at least 20")
    if args.deterministic:
        set_determinism(args.seed)

    entries, manifest_sha256 = _manifest(args.manifest, args.horizon, args.contexts)
    calibration_episodes = sorted({str(entry["episode"]) for entry in entries})
    if len(calibration_episodes) < args.min_episodes:
        raise ValueError(
            f"calibration needs at least {args.min_episodes} episodes, "
            f"got {calibration_episodes}"
        )
    rows = {
        prefix: {"immediate": [], "future": [], "episode": [], "context": []}
        for prefix in prefixes
    }
    prefix_token_mismatch = 0
    cache_path = Path(args.cache)
    if args.reuse_cache:
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path, allow_pickle=False) as cached:
            for prefix in prefixes:
                for name in ("immediate", "future", "episode", "context"):
                    key = f"{name}_p{prefix}"
                    if key not in cached:
                        raise KeyError(f"cache lacks {key}")
                    rows[prefix][name] = np.asarray(cached[key]).tolist()
        print(f"[reuse] loaded frozen branch cache {cache_path}", flush=True)
    else:
        device = torch.device(args.device)
        tokenizer, model = load_ivideogpt(
            args.upstream, args.checkpoint, horizon=args.horizon,
            action_dim=args.action_dim, device=device,
        )
        metrics = Metrics(device)
        total_jobs = len(entries) * len(prefixes)
        completed, started = 0, time.time()
        for context_index, entry in enumerate(entries):
            window = load_vp2_window_npz(
                entry["window_npz"], action_dim=args.action_dim, device=device
            )
            ground_truth = tokenize_ground_truth(tokenizer, window)
            reachable = decoded_ground_truth(tokenizer, ground_truth)
            base = sample_rollout(
                tokenizer, model, ground_truth, window.actions,
                horizon=args.horizon, group_size=args.prefix_candidates,
                seed=args.seed + context_index * 100_003,
            )
            base_rewards = frame_rewards(metrics, base, window, reachable)["rc"]
            for prefix in prefixes:
                fixed = prefix_tokens_through_frame(base.full_tokens, prefix)
                branched = sample_continuations_from_prefixes(
                    tokenizer, model, fixed, window.actions,
                    prefix_frames=prefix, horizon=args.horizon,
                    continuations=args.continuations,
                    seed=args.seed + context_index * 100_003 + prefix * 1009,
                )
                repeated = fixed.repeat_interleave(args.continuations, dim=0)
                prefix_token_mismatch += int(torch.count_nonzero(
                    branched.full_tokens[:, :fixed.shape[1]] != repeated
                ).item())
                reward = frame_rewards(metrics, branched, window, reachable)["rc"]
                future = _discounted_future(reward, prefix, args.gamma).reshape(
                    args.prefix_candidates, args.continuations
                )
                rows[prefix]["immediate"].extend(base_rewards[:, prefix - 1].tolist())
                rows[prefix]["future"].extend(future.tolist())
                rows[prefix]["episode"].extend([str(entry["episode"])] * args.prefix_candidates)
                context_name = f"{entry['episode']}:{entry['start']}"
                rows[prefix]["context"].extend([context_name] * args.prefix_candidates)
                completed += 1
                elapsed = time.time() - started
                eta = elapsed / completed * (total_jobs - completed)
                print(
                    f"[VP2 delayed influence {_bar(completed / total_jobs)}] "
                    f"{completed}/{total_jobs} context={context_index + 1}/{len(entries)} "
                    f"prefix={prefix} elapsed={_hms(elapsed)} eta={_hms(eta)}",
                    flush=True,
                )

    if prefix_token_mismatch:
        raise RuntimeError(f"branched continuations altered {prefix_token_mismatch} fixed prefix tokens")
    cache_payload = {}
    report_rows = {}
    coefficients = np.zeros(args.horizon, dtype=np.float64)
    for prefix in prefixes:
        immediate = np.asarray(rows[prefix]["immediate"], dtype=np.float64)
        future = np.asarray(rows[prefix]["future"], dtype=np.float64)
        episodes = np.asarray(rows[prefix]["episode"]).astype(str)
        contexts = np.asarray(rows[prefix]["context"]).astype(str)
        estimate = delayed_influence_ratio(
            immediate, future, episodes, folds=args.folds, ridge=args.ridge,
            seed=args.seed + prefix,
        )
        shuffled = shuffle_continuations_within_group(
            future, contexts, seed=args.seed + 10_000 + prefix
        )
        null = delayed_influence_ratio(
            immediate, shuffled, episodes, folds=args.folds, ridge=args.ridge,
            seed=args.seed + prefix,
        )
        boot, null_boot = _episode_bootstrap(
            immediate, future, episodes, contexts, args, args.seed + prefix * 100
        )
        null_q95 = float(np.quantile(null_boot, 0.95))
        paired_delta = boot - null_boot
        paired_delta_ci90 = [
            float(np.quantile(paired_delta, 0.05)),
            float(np.quantile(paired_delta, 0.95)),
        ]
        normalized_excess = np.clip(
            paired_delta / np.maximum(1.0 - null_boot, 1e-6), 0.0, 1.0
        )
        normalized_excess_ci90 = [
            float(np.quantile(normalized_excess, 0.05)),
            float(np.quantile(normalized_excess, 0.95)),
        ]
        if args.coefficient_rule == "normalized_excess_lcb":
            frozen = normalized_excess_ci90[0]
        else:
            frozen = float(max(0.0, paired_delta_ci90[0]))
        coefficients[prefix - 1] = frozen
        report_rows[str(prefix)] = {
            "block_index_zero_based": prefix - 1,
            "prefix_frames": prefix,
            "prefixes": int(len(immediate)),
            "continuations": int(future.shape[1]),
            "coefficient": estimate["coefficient"],
            "coefficient_ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
            "shuffled_coefficient": null["coefficient"],
            "shuffled_coefficient_q95": null_q95,
            "paired_real_minus_shuffle_ci90": paired_delta_ci90,
            "normalized_excess_ci90": normalized_excess_ci90,
            "frozen_coefficient": frozen,
            "oof_immediate_r2": estimate["oof_immediate_r2"],
            "between_variance": estimate["between_variance"],
            "within_variance": estimate["within_variance"],
        }
        cache_payload[f"immediate_p{prefix}"] = immediate.astype(np.float32)
        cache_payload[f"future_p{prefix}"] = future.astype(np.float32)
        cache_payload[f"episode_p{prefix}"] = episodes
        cache_payload[f"context_p{prefix}"] = contexts

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_cache:
        np.savez_compressed(cache_path, **cache_payload)
    active = int(np.count_nonzero(coefficients > 0.0))
    report = {
        "protocol": "VP2 immediate-sufficiency delayed-influence calibration v2",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha256,
        "cache": str(cache_path.resolve()),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "contexts": len(entries),
        "calibration_episodes": calibration_episodes,
        "prefix_candidates": args.prefix_candidates,
        "continuations": args.continuations,
        "bootstrap": args.bootstrap,
        "prefix_token_mismatch": prefix_token_mismatch,
        "cache_reused": bool(args.reuse_cache),
        "coefficient_rule": args.coefficient_rule,
        "minimum_active_blocks": int(args.min_active_blocks),
        "coefficients": coefficients.tolist(),
        "rows": report_rows,
        "active_blocks": active,
        "verdict": (
            "PROVISIONAL-GREEN"
            if active >= args.min_active_blocks
            else "FALLBACK"
        ),
        "scope": "frozen training configuration; not a downstream performance claim",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== VP2 Conservative Adaptive Temporal Calibration ===")
    for prefix, row in report_rows.items():
        print(
            f"prefix={prefix} lambda={row['coefficient']:.3f} "
            f"CI90=[{row['coefficient_ci90'][0]:.3f},{row['coefficient_ci90'][1]:.3f}] "
            f"shuffleQ95={row['shuffled_coefficient_q95']:.3f} "
            f"deltaCI90=[{row['paired_real_minus_shuffle_ci90'][0]:+.3f},"
            f"{row['paired_real_minus_shuffle_ci90'][1]:+.3f}] "
            f"frozen={row['frozen_coefficient']:.3f} "
            f"immediateR2={row['oof_immediate_r2']:+.3f}"
        )
    print(f"coefficients={coefficients.tolist()}")
    print(f"[verdict] {report['verdict']}")
    print(f"saved {output}\nVP2_DELAYED_INFLUENCE_OK", flush=True)


if __name__ == "__main__":
    main()
