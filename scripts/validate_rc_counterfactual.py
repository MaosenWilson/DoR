"""Counterfactual decision gate for the reconstruction-calibrated verifier.

The gate deliberately does *not* train a policy.  It asks whether a first
verifiable-frame candidate selected by RC leads to a better later rollout than
the candidate selected by the raw-GT verifier.  Future raw-GT fidelity is never
used to select either prefix.  A blur target with calibration-frozen distortion
matches the reconstruction floor and controls generic target smoothing.

The unit of inference is a context, clustered by episode; candidates and repeat
samples from the same context are never treated as independent observations.
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from dor.episodes import list_episodes
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges
from dor.multistep import (
    V_MSP,
    detok_chunked,
    discretize_actions,
    load_msp,
    msp_continue,
    msp_rollout,
    msp_sample_windows,
    msp_window,
)


def _blur(x, alpha):
    """A fixed, non-codec sham target; alpha=0 exactly recovers raw GT."""
    low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return ((1.0 - alpha) * x + alpha * low).clamp(0, 1)


def _joint_dist(metrics, x, y):
    q = metrics.eval_batch(x, y)
    return float(np.mean(np.asarray(q["lpips"], float) + np.asarray(q["mse"], float)))


def _metric_mean(metrics, pred, real):
    """Mean raw-GT outcomes over continuation candidates and future frames."""
    out = defaultdict(list)
    for h in range(pred.shape[1]):
        q = metrics.eval_batch(pred[:, h], real[h])
        for name in ("lpips", "mse", "psnr", "ssim"):
            out[name].append(float(np.mean(np.asarray(q[name], float))))
    return {name: float(np.mean(values)) for name, values in out.items()} | {
        "lpips_last": float(np.mean(np.asarray(metrics.eval_batch(pred[:, -1], real[-1])["lpips"], float)))
    }


def _cluster_bootstrap(rows, key, draws, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(float(row[key]))
    groups = list(grouped.values())
    if not groups:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "episodes": 0}
    means = np.asarray([np.mean(x) for x in groups], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for i in range(draws):
        choice = rng.integers(0, len(means), size=len(means))
        boot[i] = float(np.mean(means[choice]))
    return {
        "mean": float(np.mean(means)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "episodes": len(groups),
    }


def _cluster_sign_p(rows, key, lower_is_better):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(float(row[key]))
    x = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    wins = int(np.sum(x < 0)) if lower_is_better else int(np.sum(x > 0))
    losses = int(np.sum(x > 0)) if lower_is_better else int(np.sum(x < 0))
    n = wins + losses
    if n == 0:
        return {"wins": 0, "losses": 0, "p_one_sided": 1.0, "episodes": 0}
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return {"wins": wins, "losses": losses, "p_one_sided": float(p), "episodes": n}


@torch.no_grad()
def _reachable_target(tok, idx_c, idx_d_gt):
    return detok_chunked(tok, idx_c, idx_d_gt)[0]


def _calibrate_blur(tok, windows, args, dev, metrics):
    """Freeze one global blur alpha on a split disjoint from the test contexts."""
    grid = [float(x) for x in args.blur_alphas.split(",")]
    floors, sham = [], [[] for _ in grid]
    for path, start in windows:
        frames, _ = msp_window(path, start, args.T, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            idx_c, idx_d_gt = tok.tokenize(frames.unsqueeze(0))
        real = frames[1:args.T]
        reachable = _reachable_target(tok, idx_c, idx_d_gt)
        floors.append(_joint_dist(metrics, reachable, real))
        for i, alpha in enumerate(grid):
            sham[i].append(_joint_dist(metrics, _blur(real, alpha), real))
    floor = float(np.mean(floors))
    means = [float(np.mean(x)) for x in sham]
    best = int(np.argmin(np.abs(np.asarray(means) - floor)))
    return grid[best], floor, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=64)
    ap.add_argument("--calibration_windows", type=int, default=16)
    ap.add_argument("--exclude_windows", type=int, default=32,
                    help="fixed 24 training plus 8 held-out evaluation windows")
    ap.add_argument("--window_seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=16, help="root candidate group size")
    ap.add_argument("--continuations", type=int, default=4)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--pivots", default="2,3,4,5",
                    help="comma-separated scoreable 1-indexed horizons; all are fixed before outcome analysis")
    ap.add_argument("--generation_seed", type=int, default=9101)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--blur_alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default="outputs/analysis/rc_counterfactual_gate.json")
    args = ap.parse_args()
    if args.T < 6:
        raise ValueError("T must leave at least two future outcome frames after the h=2 pivot")
    pivots = [int(x) for x in args.pivots.split(",") if x.strip()]
    if not pivots or min(pivots) < 2 or max(pivots) >= args.T - 1:
        raise ValueError("pivots must be scoreable horizons in [2, T-2]")
    if args.deterministic:
        set_determinism(args.generation_seed)

    dev = "cuda"
    tok, model = load_msp(dev, "base")
    tok.eval(); model.eval(); model.config.use_cache = True
    metrics, action_ranges = Metrics(dev), load_action_ranges(dev)
    all_windows = msp_sample_windows(
        list_episodes(), args.exclude_windows + args.calibration_windows + args.n_windows,
        args.T, seed=args.window_seed,
    )
    calibration = all_windows[args.exclude_windows:args.exclude_windows + args.calibration_windows]
    windows = all_windows[args.exclude_windows + args.calibration_windows:]
    if len(calibration) != args.calibration_windows or len(windows) != args.n_windows:
        raise RuntimeError("insufficient disjoint windows for calibration/test protocol")

    alpha, rc_floor, blur_grid = _calibrate_blur(tok, calibration, args, dev, metrics)
    print(f"[blur] alpha={alpha:.3f} rc_floor={rc_floor:.6f} "
          f"grid={','.join(f'{x:.6f}' for x in blur_grid)}", flush=True)

    rows = []
    started = time.time()
    total = len(windows) * args.repetitions
    done = 0
    # MSP follows RLVR-World and assigns no direct reward to its first generated
    # frame, hence scoreable horizon h has zero-based index h-1 and an h-frame prefix.
    for wi, (path, start) in enumerate(windows):
        frames, actions = msp_window(path, start, args.T, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            idx_c, idx_d_gt = tok.tokenize(frames.unsqueeze(0))
        ctx_off = (idx_c.reshape(1, -1) + V_MSP).long()
        act_off = discretize_actions(actions, action_ranges)[1:args.T] + 2 * V_MSP
        real = frames[1:args.T]
        reachable = _reachable_target(tok, idx_c, idx_d_gt)
        sham = _blur(real, alpha)
        for rep in range(args.repetitions):
            root_seed = args.generation_seed + wi * 1009 + rep * 1_000_003
            dyn = msp_rollout(model, ctx_off, act_off, max(pivots), args.K, root_seed,
                              temperature=args.temperature, top_k=args.top_k)
            root_pred = detok_chunked(tok, idx_c.expand(args.K, -1, -1), dyn)
            for horizon in pivots:
                prefix_len, pivot = horizon, horizon - 1
                raw_d = np.asarray(metrics.eval_batch(root_pred[:, pivot], real[pivot])["lpips"], float)
                raw_d += np.asarray(metrics.eval_batch(root_pred[:, pivot], real[pivot])["mse"], float)
                rc_d = np.asarray(metrics.eval_batch(root_pred[:, pivot], reachable[pivot])["lpips"], float)
                rc_d += np.asarray(metrics.eval_batch(root_pred[:, pivot], reachable[pivot])["mse"], float)
                blur_d = np.asarray(metrics.eval_batch(root_pred[:, pivot], sham[pivot])["lpips"], float)
                blur_d += np.asarray(metrics.eval_batch(root_pred[:, pivot], sham[pivot])["mse"], float)
                selected = {"raw": int(np.argmin(raw_d)), "rc": int(np.argmin(rc_d)),
                            "blur": int(np.argmin(blur_d))}
                if selected["raw"] != selected["rc"]:
                    future = {}
                    for name, idx in selected.items():
                        # Same random stream for all branches in this context/repetition.
                        cont = msp_continue(
                            model, ctx_off, act_off, dyn[idx, :prefix_len], args.T - 1 - prefix_len,
                            args.continuations, seed=root_seed + 47_111 + horizon,
                            temperature=args.temperature, top_k=args.top_k,
                        )
                        pred = detok_chunked(tok, idx_c.expand(args.continuations, -1, -1), cont)
                        future[name] = _metric_mean(metrics, pred, real[prefix_len:])
                    row = {
                        "episode": os.path.basename(path), "start": int(start), "rep": rep,
                        "pivot_horizon": horizon,
                        "raw_idx": selected["raw"], "rc_idx": selected["rc"], "blur_idx": selected["blur"],
                        "raw_rc_margin": float(raw_d[selected["rc"]] - raw_d[selected["raw"]]),
                        "rc_raw_margin": float(rc_d[selected["raw"]] - rc_d[selected["rc"]]),
                    }
                    for name in ("lpips", "lpips_last", "mse", "psnr", "ssim"):
                        row[f"rc_minus_raw_{name}"] = future["rc"][name] - future["raw"][name]
                        row[f"rc_minus_blur_{name}"] = future["rc"][name] - future["blur"][name]
                    rows.append(row)
            done += 1
            if done % 2 == 0 or done == total:
                elapsed = time.time() - started
                print(f"[gate] {_bar(done / total)} {done}/{total} conflicts={len(rows)} "
                      f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (total-done))}", flush=True)

    result = {
        "protocol": vars(args) | {"pivot_horizons": pivots},
        "blur_calibration": {"alpha": alpha, "rc_floor": rc_floor, "grid_joint_dist": blur_grid},
        "n_conflict_context_repetitions": len(rows),
        "comparisons": {},
    }
    for comparator in ("raw", "blur"):
        for name, lower in (("lpips", True), ("lpips_last", True), ("mse", True),
                            ("psnr", False), ("ssim", False)):
            key = f"rc_minus_{comparator}_{name}"
            result["comparisons"][key] = _cluster_bootstrap(rows, key, args.bootstrap, args.generation_seed + len(key)) | _cluster_sign_p(rows, key, lower)
    primary = result["comparisons"]["rc_minus_raw_lpips"]
    control = result["comparisons"]["rc_minus_blur_lpips"]
    green = (primary["episodes"] >= 10 and primary["ci95"][1] < 0.0 and primary["p_one_sided"] < 0.05
             and control["ci95"][1] < 0.0 and control["p_one_sided"] < 0.05)
    result["verdict"] = "GREEN" if green else "RED"
    result["rows"] = rows
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== RC Counterfactual Decision Gate ===", flush=True)
    for key, value in result["comparisons"].items():
        print(f"{key:27s} mean={value['mean']:+.6f} CI95={value['ci95']} "
              f"wins={value['wins']}/{value['wins'] + value['losses']} p={value['p_one_sided']:.5f}", flush=True)
    print(f"[verdict] {result['verdict']}\nsaved {args.out}\nRC_COUNTERFACTUAL_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
