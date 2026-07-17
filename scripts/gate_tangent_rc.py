"""Zero-training gate for empirical reachable-tangent RC targets."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from dor.constants import CTX
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.reward_spaces import gt_reward
from dor.tangent_calibration import empirical_tangent_target
from dor.tokenization import build_prompt, decode_tokens, encode_indices


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64) - np.mean(left)
    right = np.asarray(right, dtype=np.float64) - np.mean(right)
    den = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / den) if den > 1e-12 else 0.0


def _cluster_boot(rows, key, draws, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(float(row[key]))
    values = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for index in range(draws):
        boot[index] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    return {
        "mean": float(values.mean()),
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "episodes": int(len(values)),
    }


def _cluster_boot_ratio(rows, numerator, denominator, draws, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(
            (float(row[numerator]), float(row[denominator]))
        )
    pairs = np.asarray([
        np.mean(grouped[name], axis=0) for name in sorted(grouped)
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for index in range(draws):
        sample = pairs[rng.integers(0, len(pairs), size=len(pairs))]
        boot[index] = sample[:, 0].mean() / (sample[:, 1].mean() + 1e-12)
    estimate = pairs[:, 0].mean() / (pairs[:, 1].mean() + 1e-12)
    return {
        "mean": float(estimate),
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "episodes": int(len(pairs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=32)
    parser.add_argument("--exclude_windows", type=int, default=36)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--generation_seed", type=int, default=6301)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default="outputs/analysis/tangent_rc_gate.json")
    args = parser.parse_args()
    if args.deterministic:
        set_determinism(args.generation_seed)

    device = "cuda"
    tokenizer = load_tokenizer(device)
    model = load_world_model(device, "base")
    model.eval()
    metrics = Metrics(device)
    action_ranges = load_action_ranges(device)
    windows = sample_windows(
        list_episodes(), args.exclude_windows + args.n_windows, seed=args.window_seed
    )[args.exclude_windows:]
    rows = []
    started = time.time()

    for index, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, device)
        raw_target = frames[CTX]
        prompt = build_prompt(tokenizer, frames, actions, action_ranges)
        with torch.no_grad():
            candidates = generate_candidates(
                model, prompt, args.K, seed=args.generation_seed + index
            )
            gt_indices = encode_indices(tokenizer, raw_target.unsqueeze(0))
            images = decode_tokens(tokenizer, candidates)
            reachable = decode_tokens(tokenizer, gt_indices.reshape(1, -1))[0]
            tangent, projection, diagnostics = empirical_tangent_target(
                images, reachable, raw_target, ridge=args.ridge
            )
            reversed_target = (reachable - projection).clamp(0.0, 1.0)
            raw_quality = metrics.eval_batch(images, raw_target)
            rc_quality = metrics.eval_batch(images, reachable)
            tangent_quality = metrics.eval_batch(images, tangent)
            reversed_quality = metrics.eval_batch(images, reversed_target)
            code_reward = gt_reward(
                "code", metrics, tokenizer, candidates, images,
                raw_target, gt_indices,
            )

        rewards = {}
        for name, quality in (
            ("raw", raw_quality),
            ("rc", rc_quality),
            ("tangent", tangent_quality),
            ("reversed", reversed_quality),
        ):
            rewards[name] = -(
                np.asarray(quality["lpips"], dtype=np.float64)
                + np.asarray(quality["mse"], dtype=np.float64)
            )
        raw_joint = -rewards["raw"]
        top = {name: int(np.argmax(score)) for name, score in rewards.items()}
        rho = {name: _pearson(score, code_reward) for name, score in rewards.items()}
        rc_gain = rho["rc"] - rho["raw"]
        rows.append({
            "episode": os.path.basename(path),
            "start": int(start),
            "projection_ratio": diagnostics["projection_ratio"],
            "residual_cosine": diagnostics["residual_cosine"],
            "rank_repair_retention": (rho["tangent"] - rho["raw"]) / (rc_gain + 1e-12),
            "rc_delta_rho_vs_raw": rc_gain,
            "tangent_delta_rho_vs_raw": rho["tangent"] - rho["raw"],
            "tangent_delta_rho_vs_rc": rho["tangent"] - rho["rc"],
            "tangent_raw_top_delta_vs_rc": raw_joint[top["tangent"]] - raw_joint[top["rc"]],
            "tangent_raw_top_delta_vs_reversed": raw_joint[top["tangent"]] - raw_joint[top["reversed"]],
            "same_top_as_rc": float(top["tangent"] == top["rc"]),
        })
        done = index + 1
        elapsed = time.time() - started
        print(
            f"[tangent] {_bar(done / len(windows))} {done}/{len(windows)} "
            f"proj={diagnostics['projection_ratio']:.3f} "
            f"ret={rows[-1]['rank_repair_retention']:+.2f} "
            f"topDelta={rows[-1]['tangent_raw_top_delta_vs_rc']:+.5f} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed/done*(len(windows)-done))}",
            flush=True,
        )

    keys = (
        "projection_ratio",
        "residual_cosine",
        "rank_repair_retention",
        "tangent_delta_rho_vs_raw",
        "tangent_delta_rho_vs_rc",
        "tangent_raw_top_delta_vs_rc",
        "tangent_raw_top_delta_vs_reversed",
    )
    summary = {
        key: _cluster_boot(rows, key, args.bootstrap, args.generation_seed + offset)
        for offset, key in enumerate(keys)
    }
    summary["rank_repair_retention"] = _cluster_boot_ratio(
        rows,
        "tangent_delta_rho_vs_raw",
        "rc_delta_rho_vs_raw",
        args.bootstrap,
        args.generation_seed + 100,
    )
    green = (
        summary["projection_ratio"]["ci90"][0] > 0.05
        and summary["tangent_delta_rho_vs_raw"]["ci90"][0] > 0.0
        and summary["rank_repair_retention"]["ci90"][0] > 0.5
        and summary["tangent_raw_top_delta_vs_rc"]["ci90"][1] < 0.0
        and summary["tangent_raw_top_delta_vs_reversed"]["ci90"][1] < 0.0
    )
    payload = {
        "args": vars(args),
        "summary": summary,
        "same_top_as_rc": float(np.mean([row["same_top_as_rc"] for row in rows])),
        "verdict": "GREEN" if green else "RED",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\n=== Empirical Tangent RC Gate ===")
    for key, value in summary.items():
        print(f"{key:38s} mean={value['mean']:+.6f} CI90={value['ci90']}")
    print(f"[verdict] {payload['verdict']}\nsaved {args.out}\nTANGENT_RC_GATE_OK")


if __name__ == "__main__":
    main()
