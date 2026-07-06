"""Compare final eval metrics across arms in one or more sweep dirs.
Usage: python scripts/compare_arms.py LABEL=dir [LABEL=dir ...]
Prints per (label, arm): n seeds, mean+/-std of final eval LPIPS/PSNR/SSIM/flow.
"""
import glob
import json
import re
import sys

import numpy as np

K = [("LPIPS", "eval_lpips", "lo"), ("PSNR", "eval_psnr", "hi"),
     ("SSIM", "eval_ssim", "hi"), ("flow", "eval_flow", "hi")]


def collect(d):
    arms = {}
    for f in sorted(glob.glob(f"{d}/sweep_*.json")):
        doc = json.load(open(f))
        runs = doc.get("run") or doc.get("runs") or {}
        for name, log in runs.items():
            arm = name.split("-")[0]
            arms.setdefault(arm, {k: [] for _, k, _ in K})
            for _, k, _ in K:
                v = log.get(k)
                arms[arm][k].append(float(v[-1]) if v else np.nan)
    return arms


def ms(x):
    x = np.array(x); x = x[np.isfinite(x)]
    return f"{x.mean():.4f}+/-{x.std(ddof=1) if len(x)>1 else 0:.4f}" if len(x) else "  --  "


print(f"{'config/arm':28s}{'n':>3s}" + "".join(f"{n:>18s}" for n, _, _ in K))
for spec in sys.argv[1:]:
    label, d = spec.split("=", 1)
    for arm, m in collect(d).items():
        n = len(m["eval_lpips"])
        print(f"{label+'/'+arm:28s}{n:3d}" + "".join(f"{ms(m[k]):>18s}" for _, k, _ in K))
print("\nLPIPS lo=better; PSNR/SSIM/flow hi=better. COMPARE_OK")
