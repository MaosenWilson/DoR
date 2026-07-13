"""Pre-registered paired analysis for the fixed n=10 temporal-return extension."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats


METRICS = ("eval_lpips", "eval_lpips_last", "eval_mse")
PROTOCOL = ("T", "K", "steps", "batch_windows", "train_windows", "eval_windows",
            "lr", "kl", "temporal_gamma", "horizon_kl_alpha", "eval_every",
            "deterministic", "which")


def _seed(path):
    match = re.search(r"_s(\d+)\.json$", os.path.basename(path))
    if not match:
        raise ValueError(f"cannot parse seed from {path!r}")
    return int(match.group(1))


def _load(pattern, expected_adv, expected_reward):
    rows = {}
    protocol = None
    for path in sorted(glob.glob(pattern)):
        payload = json.load(open(path))
        run_args = payload["args"]
        requested_rewards = {x.strip() for x in run_args["rewards"].split(",")}
        if run_args["adv_temporal"] != expected_adv or expected_reward not in requested_rewards:
            raise ValueError(
                f"wrong arm in {path}: adv={run_args['adv_temporal']!r}, "
                f"reward={run_args['rewards']!r}"
            )
        current = {name: run_args[name] for name in PROTOCOL}
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(f"protocol mismatch in {path}: {current} != {protocol}")
        run_name, run = next(iter(payload["run"].items()))
        if run_name != f"{expected_reward}-msp":
            raise ValueError(f"wrong run key in {path}: {run_name!r}")
        seed = _seed(path)
        if seed in rows:
            raise ValueError(f"duplicate seed {seed} matched by {pattern!r}")
        rows[seed] = {name: float(run[name][-1]) for name in METRICS}
    if not rows:
        raise FileNotFoundError(f"no files match {pattern!r}")
    return rows, protocol


def _holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def _compare(a, b, seeds):
    # Delta is b-a; all registered metrics are lower-is-better.
    report = {}
    pvalues = []
    for metric in METRICS:
        delta = np.asarray([b[s][metric] - a[s][metric] for s in seeds])
        test = stats.ttest_rel([b[s][metric] for s in seeds],
                               [a[s][metric] for s in seeds])
        report[metric] = {
            "delta": delta.tolist(),
            "mean_delta": float(delta.mean()),
            "sd_delta": float(delta.std(ddof=1)),
            "wins": int(np.sum(delta < 0)),
            "n": len(delta),
            "paired_t": float(test.statistic),
            "p_two_sided": float(test.pvalue),
        }
        pvalues.append(float(test.pvalue))
    for metric, adjusted in zip(METRICS, _holm(pvalues)):
        report[metric]["p_holm"] = adjusted
    return report


def _arm_summary(rows, seeds):
    return {
        metric: {
            "mean": float(np.mean([rows[s][metric] for s in seeds])),
            "sample_sd": float(np.std([rows[s][metric] for s in seeds], ddof=1)),
        }
        for metric in METRICS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True)
    parser.add_argument("--return", dest="ret", required=True)
    parser.add_argument("--raw", default="")
    parser.add_argument("--expected_n", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seq, seq_protocol = _load(args.seq, "seq", "rc")
    ret, ret_protocol = _load(args.ret, "return", "rc")
    if seq_protocol != ret_protocol:
        raise ValueError(f"seq/return protocols differ: {seq_protocol} != {ret_protocol}")
    seeds = sorted(set(seq) & set(ret))
    if len(seeds) != args.expected_n:
        raise RuntimeError(f"expected {args.expected_n} paired seeds, found {seeds}")
    temporal = _compare(seq, ret, seeds)
    payload = {
        "protocol": seq_protocol,
        "paired_seeds": seeds,
        "primary_metric": "eval_lpips",
        "arms": {"seq_rc": _arm_summary(seq, seeds),
                 "return_rc": _arm_summary(ret, seeds)},
        "return_minus_seq": temporal,
    }
    if args.raw:
        raw, raw_protocol = _load(args.raw, "seq", "raw")
        if raw_protocol != seq_protocol:
            raise ValueError("raw/seq protocols differ")
        raw_seeds = sorted(set(raw) & set(seq))
        if len(raw_seeds) != args.expected_n:
            raise RuntimeError(f"expected {args.expected_n} raw/RC seeds, found {raw_seeds}")
        payload["arms"]["seq_raw"] = _arm_summary(raw, raw_seeds)
        payload["rc_minus_raw"] = _compare(raw, seq, raw_seeds)

    primary = temporal["eval_lpips"]
    payload["temporal_primary_pass"] = bool(
        primary["mean_delta"] < 0
        and primary["wins"] >= 7
        and primary["p_two_sided"] < 0.05
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\n=== Fixed n=10 Temporal-Return Analysis ===")
    for metric, row in temporal.items():
        print(f"{metric:18s} delta={row['mean_delta']:+.6f} "
              f"wins={row['wins']}/{row['n']} t={row['paired_t']:+.2f} "
              f"p={row['p_two_sided']:.4f} holm={row['p_holm']:.4f}")
    print(f"primary_pass={payload['temporal_primary_pass']}")
    print(f"saved {args.out}\nMSP_N10_ANALYSIS_OK")


if __name__ == "__main__":
    main()
