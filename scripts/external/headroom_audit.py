"""Zero-training headroom audit: is the base world model imperfect enough, with
horizon-growing error, for temporal credit to have room?

Reports per-horizon LPIPS/SSIM/MSE of base-model candidates against raw GT.
A good C2 platform looks like RT-1 (LPIPS clearly > 0 and growing with horizon),
not like RoboDesk open_drawer (LPIPS ~0.006, flat = near-ceiling, no room).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dor.adapters.ivideogpt_vp2 import (
    CONTEXT_LENGTH,
    decoded_ground_truth,
    load_ivideogpt,
    load_vp2_window_npz,
    sample_rollout,
    tokenize_ground_truth,
)
from dor.grpo import set_determinism
from dor.metrics import Metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--contexts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    set_determinism(args.seed)
    device = torch.device(args.device)
    tokenizer, model = load_ivideogpt(
        args.upstream, args.checkpoint, horizon=args.horizon,
        action_dim=args.action_dim, device=device,
    )
    metrics = Metrics(device)
    entries = json.loads(Path(args.manifest).read_text())["entries"][: args.contexts]

    H = args.horizon
    lp_mean = np.zeros(H); ss_mean = np.zeros(H); mse_mean = np.zeros(H)
    lp_best = np.zeros(H)
    n = 0
    for ci, entry in enumerate(entries):
        window = load_vp2_window_npz(entry["window_npz"], action_dim=args.action_dim, device=device)
        gt_tokens = tokenize_ground_truth(tokenizer, window)
        rollout = sample_rollout(
            tokenizer, model, gt_tokens, window.actions,
            horizon=H, group_size=args.K, seed=args.seed + ci * 100_003,
        )
        pred = rollout.decoded[:, CONTEXT_LENGTH:]          # [K,H,3,h,w]
        gt = window.frames[CONTEXT_LENGTH:]                 # [H,3,h,w] raw GT
        for h in range(H):
            q = metrics.eval_batch(pred[:, h], gt[h])
            lp = np.asarray(q["lpips"], float)
            lp_mean[h] += lp.mean(); lp_best[h] += lp.min()
            ss_mean[h] += np.asarray(q["ssim"], float).mean() if "ssim" in q else 0.0
            mse_mean[h] += np.asarray(q["mse"], float).mean()
        n += 1
        print(f"[headroom {ci+1}/{len(entries)}] context={entry.get('episode','?')}", flush=True)

    lp_mean /= n; ss_mean /= n; mse_mean /= n; lp_best /= n
    print("\n=== Base-model headroom (per horizon, mean over %d contexts x K=%d) ===" % (n, args.K))
    print("horizon |  LPIPS(mean)  LPIPS(best)   SSIM     MSE")
    for h in range(H):
        print("  h=%d   |   %.5f      %.5f    %.4f   %.6f" %
              (h + 1, lp_mean[h], lp_best[h], ss_mean[h], mse_mean[h]))
    growth = lp_mean[-1] - lp_mean[0]
    late = float(lp_mean[(2 * H) // 3:].mean())  # late-third mean, where temporal credit acts
    print("\nLPIPS horizon-growth (last - first) = %+.5f ; late-third mean = %.5f" % (growth, late))
    # Room for temporal credit needs non-trivial LATE-horizon error AND clear growth,
    # not a high overall mean (early frames are easy on every platform).
    verdict = "ROOM" if (late > 0.03 and growth > 0.02) else "NEAR-CEILING"
    print("headroom verdict: %s  (late-third LPIPS %.5f, growth %+.5f)" % (verdict, late, growth))
    report = {"checkpoint": args.checkpoint, "manifest": args.manifest, "contexts": n, "K": args.K,
              "lpips_mean_by_h": lp_mean.tolist(), "lpips_best_by_h": lp_best.tolist(),
              "ssim_by_h": ss_mean.tolist(), "mse_by_h": mse_mean.tolist(),
              "lpips_growth": float(growth), "verdict": verdict}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("saved", args.out)
    print("HEADROOM_AUDIT_OK")


if __name__ == "__main__":
    main()
