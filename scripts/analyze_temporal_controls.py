"""Paired analysis for temporal-return correspondence controls."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
from scipy import stats


METRICS = ("eval_lpips", "eval_lpips_last", "eval_mse")
PROTOCOL = (
    "T", "K", "steps", "batch_windows", "train_windows", "eval_windows",
    "lr", "kl", "kl_type", "temporal_gamma", "horizon_kl_alpha",
    "eval_every", "deterministic", "which",
)


def _load(pattern, expected_mode, expected_horizon):
    rows, protocol = {}, None
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"_s(\d+)\.json$", os.path.basename(path))
        if not match:
            raise ValueError(f"cannot parse seed from {path!r}")
        payload = json.load(open(path))
        args = payload["args"]
        if args["rewards"] != "rc":
            raise ValueError(f"control must use only RC reward: {path}")
        if args["adv_temporal"] != expected_mode:
            raise ValueError(f"wrong temporal mode in {path}")
        if int(args["return_horizon"]) != expected_horizon:
            raise ValueError(f"wrong return horizon in {path}")
        current = {name: args[name] for name in PROTOCOL}
        if protocol is None:
            protocol = current
        elif protocol != current:
            raise ValueError(f"within-arm protocol mismatch in {path}")
        run = payload["run"]["rc-msp"]
        seed = int(match.group(1))
        if seed in rows:
            raise ValueError(f"duplicate seed {seed} in {pattern!r}")
        rows[seed] = {metric: float(run[metric][-1]) for metric in METRICS}
    if not rows:
        raise FileNotFoundError(f"no files match {pattern!r}")
    return rows, protocol


def _paired(delta):
    delta = np.asarray(delta, dtype=np.float64)
    sd = delta.std(ddof=1)
    t = delta.mean() / (sd / math.sqrt(len(delta))) if sd > 0 else float("inf")
    p = 2 * stats.t.sf(abs(t), len(delta) - 1) if np.isfinite(t) else 0.0
    return {
        "values": delta.tolist(), "mean": float(delta.mean()),
        "sample_sd": float(sd), "t": float(t), "p": float(p),
        "full_better": int(np.sum(delta < 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunc1", required=True)
    parser.add_argument("--trunc3", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--expected_n", type=int, default=5)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    specs = {
        "trunc1": (cli.trunc1, "return", 1),
        "trunc3": (cli.trunc3, "return", 3),
        "full": (cli.full, "return", 0),
        "shuffled": (cli.shuffled, "shuffled_return", 0),
    }
    arms, protocols = {}, []
    for name, (pattern, mode, horizon) in specs.items():
        arms[name], protocol = _load(pattern, mode, horizon)
        protocols.append(protocol)
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("control arms do not share the fixed protocol")
    seeds = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if len(seeds) != cli.expected_n:
        raise RuntimeError(f"expected {cli.expected_n} paired seeds, found {seeds}")

    report = {"protocol": protocols[0], "paired_seeds": seeds, "metrics": {}}
    for metric in METRICS:
        full = np.asarray([arms["full"][seed][metric] for seed in seeds])
        report["metrics"][metric] = {}
        for control in ("trunc1", "trunc3", "shuffled"):
            other = np.asarray([arms[control][seed][metric] for seed in seeds])
            report["metrics"][metric][f"full_minus_{control}"] = _paired(full - other)

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\n=== Temporal Correspondence Controls ===")
    for metric, rows in report["metrics"].items():
        print(f"\n{metric} (negative means full aligned return is better)")
        for name, row in rows.items():
            print(
                f"  {name:24s} delta={row['mean']:+.6f} "
                f"wins={row['full_better']}/{len(seeds)} t={row['t']:+.2f} p={row['p']:.4f}"
            )
    print(f"saved {cli.out}\nTEMPORAL_CONTROLS_OK")


if __name__ == "__main__":
    main()
