"""Stratified diagnostic on a cached reward_spaces npz (no recompute).

The aggregate rank-preservation verdict can wash out the mechanism. story_spine
predicts code's advantage over pixel grows as the within-group TRUE-quality spread
sigma_star shrinks (floor-dominated regime). Test that directly:

  * per-window rp_pix = spearman(-lpips, q_ref), rp_code = spearman(-code_rms, q_ref)
  * delta = rp_code - rp_pix
  * sigma_star proxy = within-group std of q_ref (spread of true quality among the K)
  * stratify windows into terciles by sigma_star (and by motion); report rp per arm
  * correlate delta with sigma_star and motion

CAVEAT printed up top: q_ref = -dino_cos is computed on DECODED images, so it shares
the tokenizer floor with pixel arms -> this test is biased TOWARD pixel. Read deltas
as a lower bound on code's true advantage.
"""
import argparse

import numpy as np

from dor.metrics import spearman


def per_window_rp(reward, q_ref):
    return np.array([spearman(reward[i], q_ref[i]) for i in range(reward.shape[0])], float)


def tercile_table(label, key, rp_pix, rp_code, mask):
    k = key[mask]
    dp, dc = rp_pix[mask], rp_code[mask]
    qs = np.quantile(k, [1 / 3, 2 / 3])
    bins = [k <= qs[0], (k > qs[0]) & (k <= qs[1]), k > qs[1]]
    names = [f"low {label}", f"mid {label}", f"high {label}"]
    print(f"\n--- stratified by {label} (low = floor-dominated regime) ---")
    print(f"{'bin':16s}{'n':>5s}{'rp_pixel':>11s}{'rp_code':>10s}{'delta':>9s}")
    for nm, b in zip(names, bins):
        bb = b & np.isfinite(dp) & np.isfinite(dc)
        if bb.sum() == 0:
            continue
        mp, mc = dp[bb].mean(), dc[bb].mean()
        print(f"{nm:16s}{bb.sum():5d}{mp:11.4f}{mc:10.4f}{mc - mp:+9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    d = dict(np.load(args.cache))

    q_ref = -d["dino_cos"]                      # [N,K] post-decode reference (caveat above)
    rp_pix = per_window_rp(-d["lpips"], q_ref)
    rp_code = per_window_rp(-d["code_rms"], q_ref)
    delta = rp_code - rp_pix

    sigma_star = q_ref.std(axis=1)              # within-group true-quality spread proxy
    motion = d["motion"]

    fin = np.isfinite(rp_pix) & np.isfinite(rp_code)
    print("=== CAVEAT: q_ref is post-decode (DINO on decoded imgs); biased toward pixel. ===")
    print(f"windows usable: {fin.sum()}/{len(rp_pix)}")
    print(f"aggregate: rp_pixel={np.nanmean(rp_pix):.4f}  rp_code={np.nanmean(rp_code):.4f}  "
          f"delta={np.nanmean(delta[fin]):+.4f}")

    # mechanism test: does code win more where sigma_star is small?
    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return spearman(a[m], b[m])
    print(f"\nspearman(delta, sigma_star) = {corr(delta, sigma_star):+.4f}   "
          "(thesis predicts NEGATIVE: code wins as spread shrinks)")
    print(f"spearman(delta, motion)     = {corr(delta, motion):+.4f}   "
          "(thesis predicts NEGATIVE: code wins in low-motion/static windows)")

    tercile_table("sigma_star", sigma_star, rp_pix, rp_code, fin)
    tercile_table("motion", motion, rp_pix, rp_code, fin)

    # reference informativeness vs spread: if ref is floor-dominated at low spread,
    # EVERYTHING decorrelates there -> shows the proxy itself degrades.
    print("\n--- reference informativeness (mean |rp_pixel|) by sigma_star tercile ---")
    qs = np.quantile(sigma_star[fin], [1 / 3, 2 / 3])
    for nm, b in zip(["low", "mid", "high"],
                     [sigma_star <= qs[0], (sigma_star > qs[0]) & (sigma_star <= qs[1]), sigma_star > qs[1]]):
        bb = b & fin
        print(f"  {nm:5s} sigma_star  n={bb.sum():4d}  mean|rp_pixel|={np.abs(rp_pix[bb]).mean():.4f}")
    print("STRATIFIED_OK")


if __name__ == "__main__":
    main()
