"""Paired analysis of raw, encoder-RC, MRRT, and matched-random training arms."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
from scipy import stats


METRICS = (
    "eval_lpips", "eval_mse", "eval_psnr", "eval_ssim",
    "eval_flow", "eval_dmotion",
)
PROTOCOL = (
    "modes", "steps", "K", "batch_windows", "train_windows", "eval_windows",
    "lr", "eval_every", "deterministic",
)
ARM_REWARD = {
    "raw": "a0faithful",
    "encoder_rc": "a0faithful_tok",
    "mrrt": "mrrt",
    "random": "mrrt_random",
}


def _load(pattern, arm):
    reward = ARM_REWARD[arm]
    rows, protocol = {}, None
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"_s(\d+)\.json$", os.path.basename(path))
        if not match:
            raise ValueError(f"cannot parse seed from {path!r}")
        payload = json.load(open(path))
        args = payload["args"]
        if args["rewards"] != "a0faithful,a0faithful_tok,mrrt,mrrt_random":
            raise ValueError(f"unexpected reward sweep in {path}")
        if args["modes"] != "gt_only":
            raise ValueError(f"MRRT comparison must use gt_only in {path}")
        current = {name: args[name] for name in PROTOCOL}
        if protocol is None:
            protocol = current
        elif protocol != current:
            raise ValueError(f"within-arm protocol mismatch in {path}")
        run = payload["run"][f"{reward}-gt_only"]
        seed = int(match.group(1))
        rows[seed] = {metric: float(run[metric][-1]) for metric in METRICS}
    if not rows:
        raise FileNotFoundError(f"no files match {pattern!r}")
    return rows, protocol


def _paired(delta, lower_is_better):
    delta = np.asarray(delta, dtype=np.float64)
    sd = delta.std(ddof=1)
    t = delta.mean() / (sd / math.sqrt(len(delta))) if sd > 0 else float("inf")
    p = 2 * stats.t.sf(abs(t), len(delta) - 1) if np.isfinite(t) else 0.0
    wins = delta < 0 if lower_is_better else delta > 0
    return {
        "values": delta.tolist(), "mean": float(delta.mean()),
        "sample_sd": float(sd), "t": float(t), "p": float(p),
        "mrrt_wins": int(np.sum(wins)),
    }


def main():
    parser = argparse.ArgumentParser()
    for arm in ARM_REWARD:
        parser.add_argument(f"--{arm}", required=True)
    parser.add_argument("--expected_n", type=int, default=5)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    arms, protocols = {}, []
    for arm in ARM_REWARD:
        arms[arm], protocol = _load(getattr(cli, arm), arm)
        protocols.append(protocol)
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("the four MRRT arms do not share the fixed protocol")
    seeds = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if len(seeds) != cli.expected_n:
        raise RuntimeError(f"expected {cli.expected_n} paired seeds, found {seeds}")

    report = {"protocol": protocols[0], "paired_seeds": seeds, "metrics": {}}
    for metric in METRICS:
        lower = metric in ("eval_lpips", "eval_mse")
        values = {
            arm: np.asarray([arms[arm][seed][metric] for seed in seeds])
            for arm in arms
        }
        report["metrics"][metric] = {
            "arm_means": {arm: float(value.mean()) for arm, value in values.items()},
            "mrrt_minus_encoder_rc": _paired(values["mrrt"] - values["encoder_rc"], lower),
            "mrrt_minus_raw": _paired(values["mrrt"] - values["raw"], lower),
            "mrrt_minus_random": _paired(values["mrrt"] - values["random"], lower),
        }

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\n=== MRRT Four-Arm Paired Analysis ===")
    for metric, rows in report["metrics"].items():
        print(f"\n{metric}")
        for name in ("mrrt_minus_encoder_rc", "mrrt_minus_raw", "mrrt_minus_random"):
            row = rows[name]
            print(
                f"  {name:25s} delta={row['mean']:+.6f} "
                f"wins={row['mrrt_wins']}/{len(seeds)} t={row['t']:+.2f} p={row['p']:.4f}"
            )
    print(f"saved {cli.out}\nMRRT_ANALYSIS_OK")


if __name__ == "__main__":
    main()
