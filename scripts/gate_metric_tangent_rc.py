"""Gate empirical tangent calibration in the exact MSE+LPIPS geometry."""

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
from dor.metric_tangent import MSELPIPSGeometry, metric_tangent_scores
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.reward_spaces import gt_reward
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
    boot = np.asarray([
        np.mean(values[rng.integers(0, len(values), size=len(values))])
        for _ in range(draws)
    ])
    return {
        "mean": float(values.mean()),
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "episodes": int(len(values)),
    }


def _cluster_ratio(rows, numerator, denominator, draws, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append((row[numerator], row[denominator]))
    pairs = np.asarray([np.mean(grouped[name], axis=0) for name in sorted(grouped)])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        sample = pairs[rng.integers(0, len(pairs), size=len(pairs))]
        boot.append(sample[:, 0].mean() / (sample[:, 1].mean() + 1e-12))
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
    parser.add_argument("--generation_seed", type=int, default=6701)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--anchor_tolerance", type=float, default=2e-5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default="outputs/analysis/metric_tangent_rc_gate.json")
    args = parser.parse_args()
    if args.deterministic:
        set_determinism(args.generation_seed)

    device = "cuda"
    tokenizer = load_tokenizer(device)
    model = load_world_model(device, "base")
    model.eval()
    metrics = Metrics(device)
    geometry = MSELPIPSGeometry(metrics.lpips)
    action_ranges = load_action_ranges(device)
    windows = sample_windows(
        list_episodes(), args.exclude_windows + args.n_windows, seed=args.window_seed
    )[args.exclude_windows:]
    rows, anchor_errors = [], []
    started = time.time()
    for index, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, device)
        raw_target = frames[CTX]
        prompt = build_prompt(tokenizer, frames, actions, action_ranges)
        with torch.no_grad():
            candidate_tokens = generate_candidates(
                model, prompt, args.K, seed=args.generation_seed + index
            )
            gt_indices = encode_indices(tokenizer, raw_target.unsqueeze(0))
            images = decode_tokens(tokenizer, candidate_tokens)
            reachable = decode_tokens(tokenizer, gt_indices.reshape(1, -1))[0]
            candidate_blocks = geometry.extract(images)
            raw_blocks = geometry.extract(raw_target.unsqueeze(0))
            reachable_blocks = geometry.extract(reachable.unsqueeze(0))
            distances, diagnostics = metric_tangent_scores(
                candidate_blocks, reachable_blocks, raw_blocks, ridge=args.ridge
            )
            raw_quality = metrics.eval_batch(images, raw_target)
            rc_quality = metrics.eval_batch(images, reachable)
            code_reward = gt_reward(
                "code", metrics, tokenizer, candidate_tokens, images,
                raw_target, gt_indices,
            )
        expected_raw = np.asarray(raw_quality["lpips"]) + np.asarray(raw_quality["mse"])
        expected_rc = np.asarray(rc_quality["lpips"]) + np.asarray(rc_quality["mse"])
        raw_distance = distances["raw"].float().cpu().numpy()
        rc_distance = distances["rc"].float().cpu().numpy()
        anchor_error = max(
            float(np.max(np.abs(raw_distance - expected_raw))),
            float(np.max(np.abs(rc_distance - expected_rc))),
        )
        anchor_errors.append(anchor_error)
        rewards = {name: -value.float().cpu().numpy() for name, value in distances.items()}
        top = {name: int(np.argmax(value)) for name, value in rewards.items()}
        rho = {name: _pearson(value, code_reward) for name, value in rewards.items()}
        raw_joint = expected_raw
        rows.append({
            "episode": os.path.basename(path),
            "start": int(start),
            "projection_ratio": float(diagnostics["projection_ratio"]),
            "residual_cosine": float(diagnostics["residual_cosine"]),
            "rc_delta_rho_vs_raw": float(rho["rc"] - rho["raw"]),
            "tangent_delta_rho_vs_raw": float(rho["tangent"] - rho["raw"]),
            "tangent_delta_rho_vs_rc": float(rho["tangent"] - rho["rc"]),
            "tangent_raw_top_delta_vs_rc": float(
                raw_joint[top["tangent"]] - raw_joint[top["rc"]]
            ),
            "tangent_raw_top_delta_vs_reversed": float(
                raw_joint[top["tangent"]] - raw_joint[top["reversed"]]
            ),
            "same_top_as_rc": float(top["tangent"] == top["rc"]),
        })
        done = index + 1
        elapsed = time.time() - started
        print(
            f"[metric-tangent] {_bar(done/len(windows))} {done}/{len(windows)} "
            f"anchor={anchor_error:.2e} proj={diagnostics['projection_ratio']:.3f} "
            f"topDelta={rows[-1]['tangent_raw_top_delta_vs_rc']:+.5f} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed/done*(len(windows)-done))}",
            flush=True,
        )

    summary = {
        key: _cluster_boot(rows, key, args.bootstrap, args.generation_seed + offset)
        for offset, key in enumerate((
            "projection_ratio", "residual_cosine", "tangent_delta_rho_vs_raw",
            "tangent_delta_rho_vs_rc", "tangent_raw_top_delta_vs_rc",
            "tangent_raw_top_delta_vs_reversed",
        ))
    }
    summary["rank_repair_retention"] = _cluster_ratio(
        rows, "tangent_delta_rho_vs_raw", "rc_delta_rho_vs_raw",
        args.bootstrap, args.generation_seed + 100,
    )
    green = (
        max(anchor_errors) <= args.anchor_tolerance
        and summary["projection_ratio"]["ci90"][0] > 0.05
        and summary["tangent_delta_rho_vs_raw"]["ci90"][0] > 0.0
        and summary["rank_repair_retention"]["ci90"][0] > 0.5
        and summary["tangent_raw_top_delta_vs_rc"]["ci90"][1] < 0.0
        and summary["tangent_raw_top_delta_vs_reversed"]["ci90"][1] < 0.0
    )
    payload = {
        "args": vars(args), "max_anchor_error": max(anchor_errors),
        "summary": summary,
        "same_top_as_rc": float(np.mean([row["same_top_as_rc"] for row in rows])),
        "verdict": "GREEN" if green else "RED", "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\n=== Metric-Tangent RC Gate ===")
    print(f"max_anchor_error={payload['max_anchor_error']:.3e}")
    for key, value in summary.items():
        print(f"{key:38s} mean={value['mean']:+.6f} CI90={value['ci90']}")
    print(f"[verdict] {payload['verdict']}\nsaved {args.out}\nMETRIC_TANGENT_RC_GATE_OK")


if __name__ == "__main__":
    main()
