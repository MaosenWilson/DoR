"""Zero-bulk-download preview: is BAIR in the right regime for C2/GAE?

Slices the single released BAIR sample trajectory into overlapping windows and,
with no bulk data download, measures the three prerequisites we found matter:
  * headroom  -- per-horizon base LPIPS/SSIM (does error grow with horizon?)
  * spread    -- cross-candidate reward std (is the group separable?)
  * propagation -- cross-candidate corr r[t] vs r[t+k] (does early affect late?)

A go signal (grow + spread + propagation) justifies downloading more BAIR data.
Windows from one trajectory are correlated, so this is indicative, not final.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dor.adapters.ivideogpt_bair import bair_windows_from_trajectory, load_bair_ivideogpt
from dor.adapters.ivideogpt_vp2 import (
    CONTEXT_LENGTH, decoded_ground_truth, frame_rewards,
    sample_rollout, tokenize_ground_truth,
)
from dor.grpo import set_determinism
from dor.metrics import Metrics


def propagation_curve(rewards):
    """rewards [G,H,K] -> (rho(k) for k=0..H-1, early->late corr); cross-candidate."""
    rewards = np.asarray(rewards, dtype=np.float64)
    G, H, K = rewards.shape
    rho_k = np.full(H, np.nan)
    for k in range(H):
        vals = [np.corrcoef(rewards[g, t], rewards[g, t + k])[0, 1]
                for g in range(G) for t in range(H - k)
                if rewards[g, t].std() > 1e-9 and rewards[g, t + k].std() > 1e-9]
        if vals:
            rho_k[k] = float(np.mean(vals))
    el = [np.corrcoef(rewards[g, 0], rewards[g, -1])[0, 1] for g in range(G)
          if rewards[g, 0].std() > 1e-9 and rewards[g, -1].std() > 1e-9]
    return rho_k, (float(np.mean(el)) if el else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_determinism(args.seed)
    device = torch.device(args.device)
    windows = bair_windows_from_trajectory(args.sample, args.horizon, stride=args.stride, device=device)
    print(f"[preview] {len(windows)} windows from 1 BAIR trajectory, horizon={args.horizon}", flush=True)
    tokenizer, model = load_bair_ivideogpt(args.upstream, args.checkpoint,
                                           horizon=args.horizon, device=device)
    metrics = Metrics(device)

    H = args.horizon
    lp = np.zeros(H); ss = np.zeros(H)
    rc_per_ctx = []
    for ci, window in enumerate(windows):
        gt = tokenize_ground_truth(tokenizer, window)
        reachable = decoded_ground_truth(tokenizer, gt)
        rollout = sample_rollout(tokenizer, model, gt, window.actions,
                                 horizon=H, group_size=args.K, seed=args.seed + ci * 100_003)
        pred = rollout.decoded[:, CONTEXT_LENGTH:]
        raw_gt = window.frames[CONTEXT_LENGTH:]
        for h in range(H):
            q = metrics.eval_batch(pred[:, h], raw_gt[h])
            lp[h] += np.asarray(q["lpips"], float).mean()
            ss[h] += np.asarray(q["ssim"], float).mean() if "ssim" in q else 0.0
        rc_per_ctx.append(frame_rewards(metrics, rollout, window, reachable)["rc"].T)  # [H,K]
        print(f"[preview gen {ci+1}/{len(windows)}]", flush=True)
    lp /= len(windows); ss /= len(windows)
    rewards = np.stack(rc_per_ctx)  # [ctx,H,K]
    rho_k, early_late = propagation_curve(rewards)
    spread = float(np.mean(rewards.std(axis=2)))

    print("\n=== BAIR preview (1 trajectory, %d windows x K=%d) ===" % (len(windows), args.K))
    print("horizon |  LPIPS   SSIM")
    for h in range(H):
        print("  h=%2d  |  %.4f  %.4f" % (h + 1, lp[h], ss[h]))
    print("\nheadroom: LPIPS %.4f->%.4f (growth %+.4f)  SSIM %.3f->%.3f" %
          (lp[0], lp[-1], lp[-1] - lp[0], ss[0], ss[-1]))
    print("spread (cross-candidate reward std) = %.5f" % spread)
    print("propagation lag-1 rho = %+.3f ; rho(k)=%s" %
          (rho_k[1] if H > 1 else float("nan"), [round(float(x), 2) for x in rho_k[:6]]))
    room = (lp[-1] - lp[0] > 0.02) and spread > 0.005
    print("\nPRELIMINARY: %s  (grow>0.02 & spread>0.005 => worth downloading more BAIR data)" %
          ("GO" if room else "MARGINAL/NO"))
    print("BAIR_PREVIEW_OK", flush=True)


if __name__ == "__main__":
    main()
