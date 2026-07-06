"""Block 0 analysis (matrix §4 Fig 2/3): from `cache_reward_spaces.py` output,
compute each reward arm's WITHIN-GROUP rank-preservation against the held-out
DINOv2 reference quality. This is the de-risk metric of the whole thesis:

  story_spine §5: GRPO consumes only the intra-group ranking of the reward, so the
  quantity that decides training is spearman( reward_arm , true-quality ) inside a
  group. The reward-noise floor corrupts it in pixel space; pre-decode spaces
  (phi, code) should preserve it better.

Reference quality (held-out, never a reward arm): q_ref = -dino_cos (higher==closer).

Outputs:
  - a sorted table: per-arm mean +/- s.e. of intra-group Spearman vs reference
  - reward-noise floor phi_tok = mean(floor_lpips)
  - a json with per-window correlations (for Fig 2 bars / Fig 3 scatter)

PASS/FAIL heuristic (printed): the thesis is on track iff pre-decode arms (code, phi)
have materially higher rank-preservation than pixel/floor-calibrated-pixel.
"""
import argparse
import json
import os

import numpy as np

from dor.metrics import spearman


def build_arms(d, phi_hat):
    """Map cached raw quantities -> reward arms, all 'higher == better'. Shapes [N,K]."""
    lpips, mse, psnr = d["lpips"], d["mse"], d["psnr"]
    ssim = d["ssim"] if "ssim" in d else np.full_like(lpips, np.nan)
    arms = {
        "A0 pixel(-LPIPS)": -lpips,
        "A1 -MSE": -mse,
        "A3 floor-cal pixel": -np.maximum(lpips - phi_hat, 0.0),
        "A5 phi(-RMS)": -d["phi_rms"],
        "A6 code(-RMS) [DoR]": -d["code_rms"],
    }
    if np.isfinite(ssim).all():
        arms["A2 SSIM"] = ssim
        # A4 ToolRL-style multi: mean of within-group z-scored components
        def zg(x):  # per-group z-score along K
            m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True) + 1e-6
            return (x - m) / s
        arms["A4 multi(ToolRL)"] = (zg(-lpips) + zg(psnr) + zg(ssim)) / 3.0
    return arms


def rank_pres(reward, q_ref):
    """Per-window Spearman(reward[i], q_ref[i]); returns [N] (nan where degenerate)."""
    N = reward.shape[0]
    return np.array([spearman(reward[i], q_ref[i]) for i in range(N)], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="reward_spaces.npz from cache_reward_spaces.py")
    ap.add_argument("--phi_hat", type=float, default=None,
                    help="floor estimate for the floor-cal arm; default = mean(floor_lpips)")
    ap.add_argument("--out", default=None, help="json out (default alongside cache)")
    args = ap.parse_args()

    d = dict(np.load(args.cache))
    phi_hat = args.phi_hat if args.phi_hat is not None else float(np.mean(d["floor_lpips"]))
    q_ref = -d["dino_cos"]  # held-out true-quality, higher == closer to GT

    arms = build_arms(d, phi_hat)
    rows = []
    for name, r in arms.items():
        rp = rank_pres(r, q_ref)
        rp = rp[np.isfinite(rp)]
        rows.append((name, float(rp.mean()), float(rp.std(ddof=1) / np.sqrt(len(rp))), len(rp)))
    rows.sort(key=lambda x: -x[1])

    print(f"\n=== reward-noise floor phi_tok (LPIPS) = {phi_hat:.4f} ===")
    print(f"=== within-group rank-preservation vs DINOv2 reference (N windows) ===")
    print(f"{'arm':24s}{'spearman':>12s}{'s.e.':>10s}{'N':>7s}")
    for name, m, se, n in rows:
        print(f"{name:24s}{m:12.4f}{se:10.4f}{n:7d}")

    pre = {k: v for k, v in arms.items() if ("code" in k or "phi" in k)}
    pix = {k: v for k, v in arms.items() if ("pixel" in k or "MSE" in k or "SSIM" in k)}
    best_pre = max(np.nanmean(rank_pres(v, q_ref)) for v in pre.values())
    best_pix = max(np.nanmean(rank_pres(v, q_ref)) for v in pix.values())
    verdict = "ON-TRACK" if best_pre > best_pix + 0.02 else "WEAK -- revisit thesis before training"
    print(f"\n[verdict] best pre-decode={best_pre:.4f}  best pixel={best_pix:.4f}  -> {verdict}")

    out = args.out or os.path.join(os.path.dirname(args.cache), "rank_preservation.json")
    payload = {
        "phi_tok_lpips": phi_hat,
        "arms": {name: {"mean": m, "se": se, "n": n} for name, m, se, n in rows},
        "verdict": verdict,
        "per_window": {name: rank_pres(r, q_ref).tolist() for name, r in arms.items()},
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] saved {out}")
    print("RANK_PRESERVATION_OK")


if __name__ == "__main__":
    main()
