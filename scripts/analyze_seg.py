"""Paired analysis for single-step segment-level GRPO (seg_grpo) validation.

Reads outputs/seg_<grid>_l<lambda>/sweep_*_s*.json produced by
  train_grpo.py --adv_estimator seg_grpo --seg_grid <grid> --seg_lambda <lambda>
Baseline for each grid = same grid at lambda=0 (spatial credit off, same reward pooling),
so the ONLY variable vs treatment is spatial credit assignment.

For each (grid, lambda>0) it reports, paired by seed:
  - per-config metric mean +- std over seeds
  - paired delta (treatment - grid's lambda=0 baseline) + per-seed sign test
on flow/dmotion (dynamics; the thing spatial credit should help) and LPIPS/PSNR/SSIM
(fidelity; should not drop).

Usage:  python scripts/analyze_seg.py [outputs_root]   (default: outputs)
        python scripts/analyze_seg.py --last1           (final-step instead of last-3 mean)
"""
import glob
import json
import os
import re
import statistics as st
import sys

METRICS = ["eval_flow", "eval_dmotion", "eval_lpips", "eval_psnr", "eval_ssim"]
HIGHER_BETTER = {"eval_flow": True, "eval_dmotion": True, "eval_psnr": True,
                 "eval_ssim": True, "eval_lpips": False}
SHORT = {"eval_flow": "flow", "eval_dmotion": "dmot", "eval_lpips": "LPIPS",
         "eval_psnr": "PSNR", "eval_ssim": "SSIM"}


def read_run(path, last3=True):
    d = json.load(open(path))
    run = d["run"]
    r = run[list(run.keys())[0]]
    out = {}
    for m in METRICS:
        v = r.get(m)
        if v:
            out[m] = st.mean(v[-3:]) if last3 else v[-1]
    return out


def load_config(d, last3=True):
    """dir -> {seed: {metric: value}}"""
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "sweep_*_s*.json"))):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            res[int(m.group(1))] = read_run(f, last3)
    return res


def ms(x):
    return (st.mean(x), st.pstdev(x) if len(x) > 1 else 0.0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    last3 = "--last1" not in sys.argv
    root = args[0] if args else "outputs"

    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
                  and glob.glob(os.path.join(d, "sweep_*_s*.json")))
    if not dirs:
        raise SystemExit(f"no dirs with sweep_*_s*.json under {root!r}")

    # group by full run-family prefix (e.g. "seg_2x2" vs "segpd_2x2" vs "segcf_2x2") so
    # different seg_reward sweeps never get mixed. Dirs without an _l<lambda> suffix
    # (e.g. segcf_1x1_anchor) get lam=None: shown in the per-config table, excluded
    # from paired tests (no lambda axis to pair on).
    configs = {}  # (prefix, lam_or_None) -> {seed: {metric: val}}
    for d in dirs:
        name = os.path.basename(d)
        m = re.match(r"(.+)_l([0-9]+\.[0-9]+)$", name)
        prefix, lam = (m.group(1), float(m.group(2))) if m else (name, None)
        configs[(prefix, lam)] = load_config(d, last3)

    def _key(t):
        return (t[0], -1.0 if t[1] is None else t[1])

    grids = sorted({g for g, lam in configs if lam is not None})
    print(f"readout = {'mean of last 3 evals' if last3 else 'final step'} per seed\n")

    # per-config summary
    print("=== per-config (mean +- std over seeds) ===")
    hdr = f"{'config':16s} {'n':>2s}  " + "  ".join(f"{SHORT[m]:>14s}" for m in METRICS)
    print(hdr)
    for (grid, lam) in sorted(configs, key=_key):
        rows = configs[(grid, lam)]
        cells = []
        for m in METRICS:
            vals = [rows[s][m] for s in rows if m in rows[s]]
            cells.append(f"{ms(vals)[0]:.4f}±{ms(vals)[1]:.4f}" if vals else " " * 14)
        label = grid if lam is None else f"{grid}_l{lam}"
        print(f"{label:<18s} {len(rows):>2d}  " + "  ".join(f"{c:>14s}" for c in cells))

    # paired deltas vs same-grid lambda=0
    print("\n=== paired delta (treatment - same-grid lambda=0), per-seed sign test ===")
    for grid in grids:
        base = configs.get((grid, 0.0))
        if not base:
            print(f"[{grid}] no lambda=0 baseline -> skip paired test")
            continue
        for (g, lam) in sorted(configs, key=_key):
            if g != grid or lam is None or lam == 0.0:
                continue
            treat = configs[(g, lam)]
            seeds = sorted(set(base) & set(treat))
            print(f"\n-- grid {grid}: lambda={lam} vs lambda=0  (paired seeds={seeds}) --")
            for m in METRICS:
                deltas = []
                for s in seeds:
                    if m in base[s] and m in treat[s]:
                        d = treat[s][m] - base[s][m]
                        deltas.append(d if HIGHER_BETTER[m] else -d)  # sign: +ve = better
                if not deltas:
                    continue
                mean_d, sd = ms(deltas)
                wins = sum(1 for x in deltas if x > 0)
                arrow = "better" if mean_d > 0 else "worse "
                raw = [round((treat[s][m] - base[s][m]), 4) for s in seeds if m in treat[s] and m in base[s]]
                n = len(deltas)
                if n > 1:
                    sample_sd = st.stdev(deltas)
                    se = sample_sd / (n ** 0.5)
                    t_stat = mean_d / se if se > 0 else float("nan")
                    tstr = f"  paired_t={t_stat:+.2f}(df={n - 1})"
                else:
                    tstr = ""
                print(f"  {SHORT[m]:>5s}: {arrow} meanΔ(dir)={mean_d:+.4f}±{sd:.4f}  "
                      f"wins={wins}/{n}{tstr}  rawΔ={raw}")
    print("\nGreen light = flow/dmot better with wins>=4/5 AND LPIPS/PSNR/SSIM not clearly worse.")
    print("SEG_ANALYSIS_OK")


if __name__ == "__main__":
    main()
