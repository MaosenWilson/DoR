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
CORE_PROTOCOL = {
    "msp": ("T", "K", "steps", "batch_windows", "train_windows", "eval_windows",
            "lr", "kl", "kl_type", "temporal_gamma", "return_horizon",
            "horizon_kl_alpha", "eval_every", "deterministic", "which"),
    "vp2": ("checkpoint", "train_manifest", "eval_manifest", "horizon", "K", "eval_K",
            "steps", "batch_windows", "lr", "kl", "kl_type", "gamma", "data_seed",
            "grad_clip", "eval_every", "eval_seed", "deterministic"),
}


def _load(pattern, expected_adv, expected_reward, platform="msp"):
    rows, protocol = {}, None
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"_s(\d+)\.json$", os.path.basename(path))
        if not match:
            raise ValueError(f"cannot parse seed from {path!r}")
        payload = json.load(open(path))
        run_args = payload["args"]
        requested_rewards = {x.strip() for x in run_args["rewards"].split(",")}
        credit_key = "adv_temporal" if platform == "msp" else "credits"
        requested_credits = {x.strip() for x in str(run_args[credit_key]).split(",")}
        if expected_adv not in requested_credits or expected_reward not in requested_rewards:
            raise ValueError(
                f"wrong arm in {path}: credit={run_args[credit_key]!r}, "
                f"reward={run_args['rewards']!r}"
            )
        current = {name: run_args[name] for name in CORE_PROTOCOL[platform]}
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(f"protocol mismatch in {path}")
        if platform == "msp":
            run_name, run = next(iter(payload["run"].items()))
            if run_name != f"{expected_reward}-msp":
                raise ValueError(f"wrong run key in {path}: {run_name!r}")
        else:
            run = payload["run"]
            expected_prefix = f"sweep_{expected_reward}_{expected_adv}_s"
            if not os.path.basename(path).startswith(expected_prefix):
                raise ValueError(
                    f"wrong VP2 filename identity in {path}: expected prefix "
                    f"{expected_prefix!r}"
                )
        seed = int(match.group(1))
        if seed in rows:
            raise ValueError(f"duplicate seed {seed} matched by {pattern!r}")
        rows[seed] = {name: float(run[name][-1]) for name in METRICS}
    if not rows:
        raise FileNotFoundError(f"no files match {pattern!r}")
    return rows, protocol


def _sign_flip_p_less(values):
    """Exact one-sided randomization p-value for a negative paired mean."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) > 20:
        raise ValueError("exact sign-flip inference is limited to at most 20 seeds")
    observed = float(values.mean())
    masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
    bits = (masks >> np.arange(len(values), dtype=np.uint64)) & 1
    signs = 2.0 * bits.astype(np.float64) - 1.0
    null = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(null <= observed + 1e-15))


def _paired(values, bootstrap=20000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    sd = values.std(ddof=1)
    t = values.mean() / (sd / math.sqrt(len(values))) if sd > 0 else float("inf")
    p = 2.0 * stats.t.sf(abs(t), df=len(values) - 1) if np.isfinite(t) else 0.0
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
    return {"values": values.tolist(), "mean": float(values.mean()),
            "sample_sd": float(sd), "t": float(t), "p": float(p),
            "negative": int(np.sum(values < 0)), "positive": int(np.sum(values > 0)),
            "bootstrap95": [float(np.quantile(boot, 0.025)),
                            float(np.quantile(boot, 0.975))],
            "p_sign_flip_less": _sign_flip_p_less(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("msp", "vp2"), default="msp")
    parser.add_argument("--seq_raw", required=True)
    parser.add_argument("--seq_rc", required=True)
    parser.add_argument("--return_raw", required=True)
    parser.add_argument("--return_rc", required=True)
    parser.add_argument("--expected_n", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4701)
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
        arms[name], protocol = _load(getattr(args, name), adv, reward, args.platform)
        protocols.append(protocol)
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("the four arms do not share the fixed core protocol")
    seeds = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if len(seeds) != args.expected_n:
        raise RuntimeError(f"expected {args.expected_n} paired seeds, found {seeds}")

    report = {
        "platform": args.platform,
        "protocol": protocols[0],
        "paired_seeds": seeds,
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        sr = np.asarray([arms["seq_raw"][s][metric] for s in seeds])
        sc = np.asarray([arms["seq_rc"][s][metric] for s in seeds])
        rr = np.asarray([arms["return_raw"][s][metric] for s in seeds])
        rc = np.asarray([arms["return_rc"][s][metric] for s in seeds])
        comparisons = {
            "rc_effect_under_seq": sc - sr,
            "rc_effect_under_return": rc - rr,
            "return_effect_under_raw": rr - sr,
            "return_effect_under_rc": rc - sc,
            "interaction": (rc - sc) - (rr - sr),
        }
        report["metrics"][metric] = {
            name: _paired(values, args.bootstrap, args.seed + metric_index * 10 + offset)
            for offset, (name, values) in enumerate(comparisons.items())
        }
        report["metrics"][metric].update({
            "arm_means": {"seq_raw": float(sr.mean()), "seq_rc": float(sc.mean()),
                          "return_raw": float(rr.mean()), "return_rc": float(rc.mean())},
        })

    primary = report["metrics"]["eval_lpips"]
    interaction = primary["interaction"]
    full_is_best = primary["arm_means"]["return_rc"] == min(
        primary["arm_means"].values()
    )
    provisional = bool(
        interaction["mean"] < 0
        and interaction["negative"] >= 4
        and full_is_best
    )
    formal = bool(
        args.expected_n >= 10
        and interaction["mean"] < 0
        and interaction["negative"] >= 7
        and interaction["p_sign_flip_less"] < 0.05
        and interaction["bootstrap95"][1] < 0
        and full_is_best
    )
    mse = report["metrics"]["eval_mse"]
    pilot = bool(
        args.expected_n >= 3
        and primary["return_effect_under_rc"]["mean"] < 0
        and primary["return_effect_under_rc"]["negative"] >= math.ceil(2 * len(seeds) / 3)
        and primary["rc_effect_under_return"]["mean"] < 0
        and primary["rc_effect_under_return"]["negative"] >= math.ceil(2 * len(seeds) / 3)
        and interaction["mean"] < 0
        and interaction["negative"] >= math.ceil(2 * len(seeds) / 3)
        and mse["return_effect_under_rc"]["mean"] <= 0
        and mse["rc_effect_under_return"]["mean"] <= 0
        and full_is_best
    )
    report["c3_training_gate"] = {
        "primary_metric": "eval_lpips",
        "full_stack_has_lowest_mean": bool(full_is_best),
        "pilot": pilot,
        "provisional_n5": provisional,
        "formal_n10": formal,
        "note": "C3 additionally requires the zero-training coupling mechanism gate.",
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
                  f"CI95=[{row['bootstrap95'][0]:+.6f},{row['bootstrap95'][1]:+.6f}] "
                  f"signflip-p<={row['p_sign_flip_less']:.4f} "
                  f"t={row['t']:+.2f} p2={row['p']:.4f}")
    gate = report["c3_training_gate"]
    print(f"\n[C3 training gate] full-best={gate['full_stack_has_lowest_mean']} "
          f"pilot={gate['pilot']} provisional-n5={gate['provisional_n5']} "
          f"formal-n10={gate['formal_n10']}")
    print(f"saved {args.out}\nMSP_FACTORIAL_OK")


if __name__ == "__main__":
    main()
