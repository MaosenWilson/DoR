"""Paired pilot decision for conservative adaptive temporal redistribution."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


METRICS = ("eval_lpips", "eval_lpips_last", "eval_mse", "eval_psnr", "eval_ssim", "eval_latent_rms")
LOWER_IS_BETTER = {"eval_lpips", "eval_lpips_last", "eval_mse", "eval_latent_rms"}


def _load(pattern: str, reward: str, credit: str) -> dict[int, dict]:
    rows = {}
    for path in sorted(glob.glob(pattern)):
        payload = json.loads(Path(path).read_text())
        args = payload["args"]
        rewards = {value.strip() for value in str(args["rewards"]).split(",")}
        credits = {value.strip() for value in str(args["credits"]).split(",")}
        if reward not in rewards or credit not in credits:
            raise ValueError(f"unexpected arm in {path}: {args['rewards']}/{args['credits']}")
        if not Path(path).name.startswith(f"sweep_{reward}_{credit}_s"):
            raise ValueError(f"filename does not identify {reward}/{credit}: {path}")
        match = re.search(r"_s(-?\d+)\.json$", path)
        if match:
            seed = int(match.group(1))
        else:
            values = [value for value in str(args["seeds"]).split(",") if value]
            if len(values) != 1:
                raise ValueError(f"cannot infer individual seed from {path}")
            seed = int(values[0])
        run = payload["run"]
        rows[seed] = {metric: float(run[metric][-1]) for metric in METRICS}
    if not rows:
        raise FileNotFoundError(pattern)
    return rows


def _bootstrap(values, rounds, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(rounds, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catr", required=True)
    parser.add_argument("--seq_raw", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=27183)
    parser.add_argument("--lpips_margin", type=float, default=0.00015)
    parser.add_argument("--mse_margin", type=float, default=0.00003)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidate = _load(args.catr, "catr", "adaptive")
    baseline = _load(args.seq_raw, "raw", "seq")
    seeds = sorted(set(candidate) & set(baseline))
    if len(seeds) < 3:
        raise ValueError(f"paired pilot requires at least three seeds, got {seeds}")

    report = {
        "protocol": "CATR paired VP2 pilot v1",
        "seeds": seeds,
        "primary": "final raw-GT full-rollout LPIPS versus sequence-raw",
        "pilot_thresholds": {
            "lpips_noninferiority_margin": args.lpips_margin,
            "mse_noninferiority_margin": args.mse_margin,
            "lpips_last_direction": "mean < 0 and at least 2/3 paired wins",
        },
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        delta = np.asarray([
            candidate[seed][metric] - baseline[seed][metric] for seed in seeds
        ])
        wins = int(np.sum(delta < 0.0)) if metric in LOWER_IS_BETTER else int(np.sum(delta > 0.0))
        report["metrics"][metric] = {
            "candidate_minus_seq_raw": delta.tolist(),
            "mean": float(delta.mean()),
            "ci95": _bootstrap(delta, args.bootstrap, args.seed + metric_index),
            "wins": wins,
            "n": len(seeds),
        }
    lpips = report["metrics"]["eval_lpips"]
    lpips_last = report["metrics"]["eval_lpips_last"]
    mse = report["metrics"]["eval_mse"]
    green = bool(
        lpips["mean"] <= args.lpips_margin
        and mse["mean"] <= args.mse_margin
        and lpips_last["mean"] < 0.0
        and lpips_last["wins"] >= int(np.ceil(2.0 * len(seeds) / 3.0))
    )
    report["verdict"] = "PROVISIONAL-GREEN" if green else "RED"
    report["scope"] = "pilot admission only; green permits fixed ten-seed confirmation"
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("=== CATR VP2 Paired Pilot ===")
    for metric, row in report["metrics"].items():
        print(
            f"{metric:>16s} delta={row['mean']:+.8f} "
            f"CI95=[{row['ci95'][0]:+.8f},{row['ci95'][1]:+.8f}] "
            f"wins={row['wins']}/{row['n']}"
        )
    print(f"[verdict] {report['verdict']}")
    print(f"saved {output}\nCATR_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
