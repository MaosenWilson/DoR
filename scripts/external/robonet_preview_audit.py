"""Zero-download preview: is RoboNet in the right regime for C2/GAE?

Slices the single released RoboNet sample trajectory into overlapping windows and
measures headroom (per-horizon LPIPS/SSIM growth), candidate spread, and error
propagation -- the three prerequisites we found matter -- with no bulk download.
A go signal justifies fetching more RoboNet data; near-ceiling means skip it.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dor.adapters.ivideogpt_robonet import load_robonet_ivideogpt, load_robonet_window_npz
from dor.adapters.ivideogpt_vp2 import (
    CONTEXT_LENGTH, decoded_ground_truth, frame_rewards,
    sample_rollout, tokenize_ground_truth,
)
from dor.grpo import set_determinism
from dor.metrics import Metrics


def propagation_curve(rewards):
    rewards = np.asarray(rewards, dtype=np.float64)
    G, H, K = rewards.shape
    rho_k = np.full(H, np.nan)
    for k in range(H):
        vals = [np.corrcoef(rewards[g, t], rewards[g, t + k])[0, 1]
                for g in range(G) for t in range(H - k)
                if rewards[g, t].std() > 1e-9 and rewards[g, t + k].std() > 1e-9]
        if vals:
            rho_k[k] = float(np.mean(vals))
    return rho_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_determinism(args.seed)
    device = torch.device(args.device)
    total = int(np.load(args.sample, allow_pickle=False)["image"].shape[0])
    length = CONTEXT_LENGTH + args.horizon
    starts = list(range(0, total - length + 1, max(1, args.stride)))
    if not starts:
        raise ValueError(f"sample too short ({total}) for horizon {args.horizon}")
    windows = [load_robonet_window_npz(args.sample, start=s, horizon=args.horizon, device=device)
               for s in starts]
    print(f"[preview] {len(windows)} windows from 1 RoboNet trajectory, horizon={args.horizon}", flush=True)
    tokenizer, model = load_robonet_ivideogpt(args.upstream, args.checkpoint,
                                              horizon=args.horizon, device=device)
    metrics = Metrics(device)

    H = args.horizon
    lp = np.zeros(H); ss = np.zeros(H); rc = []
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
        rc.append(frame_rewards(metrics, rollout, window, reachable)["rc"].T)
        print(f"[preview gen {ci+1}/{len(windows)}]", flush=True)
    lp /= len(windows); ss /= len(windows)
    rewards = np.stack(rc)
    rho_k = propagation_curve(rewards)
    spread = float(np.mean(rewards.std(axis=2)))

    print("\n=== RoboNet preview (1 trajectory, %d windows x K=%d) ===" % (len(windows), args.K))
    print("horizon |  LPIPS   SSIM")
    for h in range(H):
        print("  h=%2d  |  %.4f  %.4f" % (h + 1, lp[h], ss[h]))
    print("\nheadroom: LPIPS %.4f->%.4f (growth %+.4f)  SSIM %.3f->%.3f" %
          (lp[0], lp[-1], lp[-1] - lp[0], ss[0], ss[-1]))
    print("spread=%.5f ; propagation lag-1 rho=%+.3f" %
          (spread, rho_k[1] if H > 1 else float("nan")))
    room = (lp[-1] - lp[0] > 0.02) and spread > 0.005
    print("\nPRELIMINARY: %s (grow>0.02 & spread>0.005 => worth downloading more RoboNet)" %
          ("GO" if room else "MARGINAL/NO"))
    print("ROBONET_PREVIEW_OK", flush=True)


if __name__ == "__main__":
    main()
