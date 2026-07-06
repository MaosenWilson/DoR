"""Table 1 aggregator: per-arm reconstruction (LPIPS/PSNR) AND dynamics (RAFT flow /
frame-delta cosine) mean +/- std over seeds, paired test vs the pixel (A0) baseline.
Reads pilot_pixel_code_s*.json (pixel, code) and sweep_{arm}_{mode}_s{seed}.json.
Seed is parsed from the filename. Runs without a flow field show '--' (trained before
the dynamics eval was added). code_rms (code's own objective) is not a headline column.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ARM_LABEL = {
    "pixel": "A0 pixel(-LPIPS)", "mse": "A1 -MSE", "ssim": "A2 SSIM",
    "floorpc": "A3 floor(per-cand)", "floor": "A3 floor(const)",
    "multi": "A4 multi(ToolRL)", "phi": "A5 phi(-RMS)", "code": "A6 code [DoR]",
    "hybrid": "hybrid",
}
KEYS = ("eval_lpips", "eval_psnr", "eval_ssim", "eval_mse", "eval_mae", "eval_flow", "eval_dmotion")
# each arm's "home" metric (the one it optimizes) -> mark to avoid circularity confusion
HOME = {"pixel": "eval_lpips", "mse": "eval_mse", "ssim": "eval_ssim",
        "floor": "eval_lpips", "floorpc": "eval_lpips", "multi": "eval_lpips"}


def iter_runs(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "pilot_*.json"))
                   + glob.glob(os.path.join(out_dir, "sweep_*.json")))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        runs = d.get("runs") or d.get("run") or {}
        m = re.search(r"_s(\d+)\.json$", os.path.basename(f))
        seed = int(m.group(1)) if m else d.get("args", {}).get("seed", -1)
        for name, log in runs.items():
            arm, _, mode = name.partition("-")
            if mode != "gt_only" or not log.get("eval_lpips"):
                continue
            yield arm, seed, {k: (float(log[k][-1]) if log.get(k) else np.nan) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/grpo")
    args = ap.parse_args()

    data = {}  # arm -> seed -> {key: val}
    for arm, seed, vals in iter_runs(args.out_dir):
        data.setdefault(arm, {})[seed] = vals
    if "pixel" not in data:
        raise SystemExit("no pixel (A0) baseline found")

    def col(arm, k):
        return np.array([data[arm][s][k] for s in sorted(data[arm])])

    def ms(x):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return "      --      "
        return f"{x.mean():.4f}+/-{x.std(ddof=1) if len(x) > 1 else 0:.4f}"

    def paired_t(arm, k):
        pix = {s: data["pixel"][s][k] for s in data["pixel"]}
        d = np.array([data[arm][s][k] - pix[s] for s in sorted(data[arm])
                      if s in pix and np.isfinite(data[arm][s][k]) and np.isfinite(pix[s])])
        if len(d) < 2:
            return np.nan
        return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)

    order = [a for a in ARM_LABEL if a in data]
    show = [("LPIPS", "eval_lpips"), ("PSNR", "eval_psnr"), ("SSIM", "eval_ssim"),
            ("MSE", "eval_mse"), ("flow", "eval_flow")]
    hdr = f"{'arm':20s}{'n':>3s}" + "".join(f"{name:>17s}" for name, _ in show) + f"{'t:LPIPS':>9s}{'t:flow':>8s}"
    print(hdr)
    for arm in order:
        n = len(data[arm])
        cells = ""
        for _, k in show:
            v = ms(col(arm, k))
            cells += f"{(v + '*') if HOME.get(arm) == k else v:>17s}"  # * marks the arm's home metric
        tl = paired_t(arm, "eval_lpips")
        tf = paired_t(arm, "eval_flow")
        tl_s = f"{tl:9.2f}" if np.isfinite(tl) else f"{'--':>9s}"
        tf_s = f"{tf:8.2f}" if np.isfinite(tf) else f"{'--':>8s}"
        print(f"{ARM_LABEL[arm]:20s}{n:3d}{cells}{tl_s}{tf_s}")
    print("\nLPIPS/MSE lower=better; PSNR/SSIM/flow higher=better. LPIPS=vgg (RLVR-World Evaluator).")
    print("'*' = arm's own reward metric (home advantage; read code's wins on NON-* columns).")
    print("t = paired (arm-pixel) over shared seeds, |t|>2.78 ~ p<0.05@df4. TABLE1_OK")


if __name__ == "__main__":
    main()
