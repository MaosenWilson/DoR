"""Aggregate multi-seed pixel-vs-code GRPO pilot curves: mean +/- std, paired test,
error-bar overlap. Independent metrics only (LPIPS/PSNR); code_rms is code's own
objective and is excluded from the headline.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/grpo/pilot_pixel_code_s*.json")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    print(f"files: {[os.path.basename(f) for f in files]}")
    a0 = json.load(open(files[0]))["args"]
    print("config:", {k: a0[k] for k in ("steps", "K", "rewards", "modes", "kl", "lr")})

    arms = ["pixel-gt_only", "code-gt_only"]
    agg = {a: {k: [] for k in ("fin_lp", "fin_ps", "best_lp", "best_ps")} for a in arms}
    for f in files:
        runs = json.load(open(f))["runs"]
        for a in arms:
            r = runs[a]
            agg[a]["fin_lp"].append(r["eval_lpips"][-1])
            agg[a]["fin_ps"].append(r["eval_psnr"][-1])
            agg[a]["best_lp"].append(min(r["eval_lpips"]))
            agg[a]["best_ps"].append(max(r["eval_psnr"]))
    for a in arms:
        for k in agg[a]:
            agg[a][k] = np.array(agg[a][k], float)

    def fmt(x):
        return f"{x.mean():.4f}+/-{x.std(ddof=1):.4f}"

    px, cd = agg["pixel-gt_only"], agg["code-gt_only"]
    n = len(files)
    print(f"\n=== final (last eval) over {n} seeds ===")
    print(f"  LPIPS  pixel {fmt(px['fin_lp'])}    code {fmt(cd['fin_lp'])}   (lower better)")
    print(f"  PSNR   pixel {fmt(px['fin_ps'])}    code {fmt(cd['fin_ps'])}   (higher better)")

    dlp = cd["fin_lp"] - px["fin_lp"]
    dps = cd["fin_ps"] - px["fin_ps"]
    print("\n=== paired (code - pixel), per seed ===")
    print("  dLPIPS:", [round(x, 4) for x in dlp], "(neg=code better)")
    print("  dPSNR :", [round(x, 3) for x in dps], "(pos=code better)")

    def paired_t(d):
        m = d.mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        return m, m / se, len(d) - 1

    for nm, d in [("LPIPS", dlp), ("PSNR", dps)]:
        m, t, df = paired_t(d)
        win = int(np.sum(d < 0)) if nm == "LPIPS" else int(np.sum(d > 0))
        print(f"  {nm}: mean_delta={m:+.4f}  paired_t={t:+.2f} (df={df})  code wins {win}/{len(d)} seeds")

    print("\n=== error-bar overlap (mean +/- 1 std, final) ===")
    for nm, key, lower in [("LPIPS", "fin_lp", True), ("PSNR", "fin_ps", False)]:
        p, c = px[key], cd[key]
        pm, ps, cm, cs = p.mean(), p.std(ddof=1), c.mean(), c.std(ddof=1)
        disjoint = (pm - ps > cm + cs) if lower else (cm - cs > pm + ps)
        print(f"  {nm}: {'DISJOINT' if disjoint else 'overlap'}  "
              f"(pixel {pm:.4f}+/-{ps:.4f}, code {cm:.4f}+/-{cs:.4f})")
    print("AGG_OK")


if __name__ == "__main__":
    main()
