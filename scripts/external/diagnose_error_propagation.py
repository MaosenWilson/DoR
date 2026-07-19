"""Does prediction error PROPAGATE (early-block choices affect late frames) or is
it LOCAL (each frame independently imperfect)?

Temporal-return credit can only help when errors propagate: block t's advantage is
the reward-to-go, which is only informative about block t if late-frame quality
depends on block t. We measure, across the K candidates of each group, the Pearson
correlation between block-t reward and block-(t+k) reward:

    rho(k) = mean_context corr_i( r[i,t], r[i,t+k] )   averaged over valid t.

rho(k) clearly > 0 and slowly decaying = propagation (temporal credit has purchase).
rho(k) ~ 0 = local errors (reward-to-go injects non-causal noise -> return can hurt).

RT-1 rewards are read from a cached [.,H,K] array; VP2 (RoboSuite/RoboDesk) candidates
are generated on the fly with the same RC reward used in training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def propagation_curve(rewards):
    """rewards [G, H, K] -> rho(k) for k=0..H-1 (cross-candidate corr, mean over t,G)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    G, H, K = rewards.shape
    rho_k = np.full(H, np.nan)
    for k in range(H):
        vals = []
        for g in range(G):
            for t in range(H - k):
                a = rewards[g, t]; b = rewards[g, t + k]
                if a.std() > 1e-9 and b.std() > 1e-9:
                    vals.append(np.corrcoef(a, b)[0, 1])
        if vals:
            rho_k[k] = float(np.mean(vals))
    # early->late: corr between first and last block, per context
    el = []
    for g in range(G):
        a, b = rewards[g, 0], rewards[g, -1]
        if a.std() > 1e-9 and b.std() > 1e-9:
            el.append(np.corrcoef(a, b)[0, 1])
    return rho_k, (float(np.mean(el)) if el else float("nan"))


def _report(name, rewards, out_rows):
    rho_k, early_late = propagation_curve(rewards)
    G, H, K = rewards.shape
    spread = float(np.mean(rewards.std(axis=2)))  # mean cross-candidate reward std
    lag1 = float(rho_k[1]) if H > 1 else float("nan")
    verdict = "PROPAGATES" if (early_late > 0.15 and lag1 > 0.2) else "LOCAL"
    print(f"\n=== {name} : {G} groups x {H} horizons x K={K} ===")
    print("candidate reward spread (mean std over K) = %.5f" % spread)
    print("rho(k): " + "  ".join("k=%d:%+.3f" % (k, rho_k[k]) for k in range(H)))
    print("lag-1 rho = %+.3f ; early->late (block0 vs blockH) = %+.3f => %s"
          % (lag1, early_late, verdict))
    out_rows[name] = {"groups": G, "horizons": H, "K": K, "spread": spread,
                      "rho_k": rho_k.tolist(), "lag1": lag1,
                      "early_late": early_late, "verdict": verdict}


def from_cache(path, key):
    d = np.load(path, allow_pickle=False)
    arr = np.asarray(d[key], dtype=np.float64)      # e.g. [draws, ctx, H, K]
    if arr.ndim == 4:
        arr = arr.reshape(-1, arr.shape[2], arr.shape[3])
    if arr.ndim != 3:
        raise ValueError(f"expected reward array reducible to [G,H,K], got {arr.shape}")
    return arr


def from_vp2(args):
    import torch
    from dor.adapters.ivideogpt_vp2 import (
        decoded_ground_truth, frame_rewards, load_ivideogpt,
        load_vp2_window_npz, sample_rollout, tokenize_ground_truth,
    )
    from dor.grpo import set_determinism
    from dor.metrics import Metrics
    set_determinism(args.seed)
    device = torch.device(args.device)
    tokenizer, model = load_ivideogpt(args.upstream, args.checkpoint,
                                      horizon=args.horizon, action_dim=args.action_dim, device=device)
    metrics = Metrics(device)
    entries = json.loads(Path(args.manifest).read_text())["entries"][: args.contexts]
    rows = []
    for ci, entry in enumerate(entries):
        window = load_vp2_window_npz(entry["window_npz"], action_dim=args.action_dim, device=device)
        gt = tokenize_ground_truth(tokenizer, window)
        reachable = decoded_ground_truth(tokenizer, gt)
        rollout = sample_rollout(tokenizer, model, gt, window.actions,
                                 horizon=args.horizon, group_size=args.K, seed=args.seed + ci * 100_003)
        rc = frame_rewards(metrics, rollout, window, reachable)["rc"]  # [K, H]
        rows.append(rc.T)                                              # [H, K]
        print(f"[propagation gen {ci+1}/{len(entries)}]", flush=True)
    return np.stack(rows)  # [G, H, K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--precomputed_npz", default="")
    ap.add_argument("--reward_key", default="rc_reward")
    ap.add_argument("--upstream", default=""); ap.add_argument("--checkpoint", default="")
    ap.add_argument("--manifest", default=""); ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--action_dim", type=int, default=4); ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--contexts", type=int, default=24); ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="")
    args = ap.parse_args()

    rewards = from_cache(args.precomputed_npz, args.reward_key) if args.precomputed_npz else from_vp2(args)
    out_rows = {}
    _report(args.name, rewards, out_rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_rows, indent=2))
        print("saved", args.out)
    print("PROPAGATION_DIAGNOSTIC_OK")


if __name__ == "__main__":
    main()
