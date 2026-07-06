"""B2 mechanism evidence (story_spine decision #2): WITHIN-GROUP rank-flip rate,
REFERENCE-FREE. This replaces the dead "rank-pres vs DINO" aggregate figure, whose
post-decode reference is itself floor-corrupted in exactly the regime of interest.

Idea: the pre-decode code reward has ~zero reward-noise floor (no lossy decode), so
its within-group ordering is the clean signal R*. A post-decode reward (pixel -LPIPS,
-MSE, SSIM) is R* + floor noise eta. The floor's damage = how often, within a group,
the post-decode reward FLIPS a pairwise ordering relative to code.

Under a joint-Gaussian (R*, measured) model the per-pair flip probability has the
closed form P_flip = arccos(rho)/pi, rho = within-group corr(code, measured).
We report the empirical flip rate AND this bound, and how flips grow as the clean
within-group spread sigma_star shrinks (the floor-dominated regime).

This is NOT circular: code-as-R* is justified by construction (zero decode floor) and
independently by B1 (training on code yields better INDEPENDENT LPIPS/PSNR than pixel).
"""
import argparse
import json

import numpy as np

from dor.metrics import pearson


def zgroup(x):
    m = x.mean(1, keepdims=True)
    s = x.std(1, keepdims=True) + 1e-9
    return (x - m) / s


def pair_flip_rate(clean, meas):
    """Per-window fraction of candidate pairs whose order disagrees. clean/meas [N,K]."""
    N, K = clean.shape
    iu = np.triu_indices(K, k=1)
    out = np.empty(N)
    for i in range(N):
        dc = np.subtract.outer(clean[i], clean[i])[iu]
        dm = np.subtract.outer(meas[i], meas[i])[iu]
        valid = (dc != 0) & (dm != 0)
        out[i] = np.mean(np.sign(dc[valid]) != np.sign(dm[valid])) if valid.any() else np.nan
    return out


def per_window_rho(clean, meas):
    return np.array([pearson(clean[i], meas[i]) for i in range(clean.shape[0])], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = dict(np.load(args.cache))

    clean = -d["code_rms"]                      # pre-decode, ~zero floor = R* proxy
    sigma_star = clean.std(axis=1)              # within-group clean spread
    measured = {
        "pixel(-LPIPS)": -d["lpips"],
        "-MSE": -d["mse"],
    }
    if np.isfinite(d.get("ssim", np.array([np.nan]))).all():
        measured["SSIM"] = d["ssim"]

    print(f"=== within-group rank-flip vs clean pre-decode (code) ordering, N={clean.shape[0]} K={clean.shape[1]} ===")
    print(f"{'measured arm':18s}{'emp_flip':>10s}{'bound':>9s}{'|gap|':>8s}{'rho':>8s}")
    payload = {"n_windows": int(clean.shape[0]), "K": int(clean.shape[1]), "arms": {}}
    for name, m in measured.items():
        flip = pair_flip_rate(zgroup(clean), zgroup(m))
        rho = per_window_rho(clean, m)
        bound = np.arccos(np.clip(rho, -1, 1)) / np.pi   # Gaussian-pair flip prob
        fin = np.isfinite(flip) & np.isfinite(bound)
        ef, bf, rf = flip[fin].mean(), bound[fin].mean(), np.nanmean(rho)
        print(f"{name:18s}{ef:10.4f}{bf:9.4f}{abs(ef - bf):8.4f}{rf:8.4f}")
        payload["arms"][name] = {"emp_flip": float(ef), "bound": float(bf), "rho": float(rf),
                                 "flip_per_window": flip.tolist()}

    # floor-dominated regime: flip rate by sigma_star tercile (pixel arm)
    flip_pix = pair_flip_rate(zgroup(clean), zgroup(measured["pixel(-LPIPS)"]))
    fin = np.isfinite(flip_pix)
    qs = np.quantile(sigma_star[fin], [1 / 3, 2 / 3])
    print("\n--- pixel flip rate by clean spread sigma_star (low = floor-dominated) ---")
    print(f"{'bin':16s}{'n':>5s}{'flip_rate':>11s}{'mean_sigma':>12s}")
    for nm, b in zip(["low sigma_star", "mid sigma_star", "high sigma_star"],
                     [sigma_star <= qs[0], (sigma_star > qs[0]) & (sigma_star <= qs[1]), sigma_star > qs[1]]):
        bb = b & fin
        print(f"{nm:16s}{bb.sum():5d}{flip_pix[bb].mean():11.4f}{sigma_star[bb].mean():12.4f}")
    from dor.metrics import spearman
    print(f"\nspearman(flip_rate, sigma_star) = {spearman(flip_pix[fin], sigma_star[fin]):+.4f}"
          "   (thesis predicts NEGATIVE: more flips when signal is weak)")

    out = args.out or args.cache.replace(".npz", "_rankflip.json")
    json.dump(payload, open(out, "w"), indent=2)
    print(f"[done] saved {out}")
    print("RANK_FLIP_OK")


if __name__ == "__main__":
    main()
