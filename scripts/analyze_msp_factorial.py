"""Paired 2x2 analysis of verifier target and temporal credit assignment."""

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
CORE_PROTOCOL = ("T", "K", "steps", "batch_windows", "train_windows", "eval_windows",
                 "lr", "kl", "kl_type", "temporal_gamma", "return_horizon",
                 "horizon_kl_alpha", "eval_every", "deterministic", "which")


def _load(pattern, expected_adv, expected_reward):
    rows, protocol = {}, None
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"_s(\d+)\.json$", os.path.basename(path))
        if not match:
            raise ValueError(f"cannot parse seed from {path!r}")
        payload = json.load(open(path))
        run_args = payload["args"]
        requested_rewards = {x.strip() for x in run_args["rewards"].split(",")}
        if run_args["adv_temporal"] != expected_adv or expected_reward not in requested_rewards:
            raise ValueError(
                f"wrong arm in {path}: adv={run_args['adv_temporal']!r}, "
                f"reward={run_args['rewards']!r}"
            )
        current = {name: run_args[name] for name in CORE_PROTOCOL}
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(f"protocol mismatch in {path}")
        run_name, run = next(iter(payload["run"].items()))
        if run_name != f"{expected_reward}-msp":
            raise ValueError(f"wrong run key in {path}: {run_name!r}")
        seed = int(match.group(1))
        if seed in rows:
            raise ValueError(f"duplicate seed {seed} matched by {pattern!r}")
        rows[seed] = {name: float(run[name][-1]) for name in METRICS}
    if not rows:
        raise FileNotFoundError(f"no files match {pattern!r}")
    return rows, protocol


def _paired(values):
    values = np.asarray(values, dtype=np.float64)
    sd = values.std(ddof=1)
    t = values.mean() / (sd / math.sqrt(len(values))) if sd > 0 else float("inf")
    p = 2.0 * stats.t.sf(abs(t), df=len(values) - 1) if np.isfinite(t) else 0.0
    return {"values": values.tolist(), "mean": float(values.mean()),
            "sample_sd": float(sd), "t": float(t), "p": float(p),
            "negative": int(np.sum(values < 0)), "positive": int(np.sum(values > 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_raw", required=True)
    parser.add_argument("--seq_rc", required=True)
    parser.add_argument("--return_raw", required=True)
    parser.add_argument("--return_rc", required=True)
    parser.add_argument("--expected_n", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    arms, protocols = {}, []
    arm_spec = {
        "seq_raw": ("seq", "raw"),
        "seq_rc": ("seq", "rc"),
        "return_raw": ("return", "raw"),
        "return_rc": ("return", "rc"),
    }
    for name, (adv, reward) in arm_spec.items():
        arms[name], protocol = _load(getattr(args, name), adv, reward)
        protocols.append(protocol)
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("the four arms do not share the fixed core protocol")
    seeds = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if len(seeds) != args.expected_n:
        raise RuntimeError(f"expected {args.expected_n} paired seeds, found {seeds}")

    report = {"protocol": protocols[0], "paired_seeds": seeds, "metrics": {}}
    for metric in METRICS:
        sr = np.asarray([arms["seq_raw"][s][metric] for s in seeds])
        sc = np.asarray([arms["seq_rc"][s][metric] for s in seeds])
        rr = np.asarray([arms["return_raw"][s][metric] for s in seeds])
        rc = np.asarray([arms["return_rc"][s][metric] for s in seeds])
        report["metrics"][metric] = {
            "rc_effect_under_seq": _paired(sc - sr),
            "rc_effect_under_return": _paired(rc - rr),
            "return_effect_under_raw": _paired(rr - sr),
            "return_effect_under_rc": _paired(rc - sc),
            "interaction": _paired((rc - sc) - (rr - sr)),
            "arm_means": {"seq_raw": float(sr.mean()), "seq_rc": float(sc.mean()),
                          "return_raw": float(rr.mean()), "return_rc": float(rc.mean())},
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Verifier x Temporal-Credit Factorial Analysis ===")
    for metric, rows in report["metrics"].items():
        print(f"\n{metric} (negative delta is better)")
        for name in ("rc_effect_under_seq", "rc_effect_under_return",
                     "return_effect_under_raw", "return_effect_under_rc", "interaction"):
            row = rows[name]
            print(f"  {name:24s} delta={row['mean']:+.6f} "
                  f"t={row['t']:+.2f} p={row['p']:.4f}")
    print(f"saved {args.out}\nMSP_FACTORIAL_OK")


if __name__ == "__main__":
    main()
