"""Paired analysis of episode-disjoint frozen-checkpoint evaluation."""

import argparse
import itertools
import json
import os

import numpy as np
from scipy import stats


METRICS = ("lpips", "lpips_last", "mse", "psnr", "ssim")
LOWER_IS_BETTER = {"lpips", "lpips_last", "mse"}


def _load(directory, arm, seed):
    path = os.path.join(directory, f"{arm}_s{seed}.json")
    payload = json.load(open(path))
    return payload


def _sign_flip_p(values, alternative):
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(float(np.mean(values * np.asarray(signs))))
    null = np.asarray(null)
    if alternative == "less":
        return float(np.mean(null <= observed + 1e-15))
    return float(np.mean(null >= observed - 1e-15))


def _paired(values, lower, bootstrap, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(bootstrap)]
    )
    test = stats.ttest_1samp(values, 0.0)
    favorable = values < 0 if lower else values > 0
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "wins": int(favorable.sum()),
        "n": int(len(values)),
        "paired_t": float(test.statistic),
        "two_sided_p": float(test.pvalue),
        "exact_one_sided_sign_flip_p": _sign_flip_p(
            values, "less" if lower else "greater"
        ),
        "seed_bootstrap95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    arms = ("seq_raw", "seq_rc", "return_raw", "return_rc")
    payloads = {
        arm: [_load(args.eval_dir, arm, seed) for seed in seeds] for arm in arms
    }
    hashes = {
        payload["protocol"]["manifest_sha256"]
        for arm_payloads in payloads.values()
        for payload in arm_payloads
    }
    if len(hashes) != 1:
        raise RuntimeError(f"manifest mismatch: {sorted(hashes)}")

    report = {
        "paired_seeds": seeds,
        "manifest_sha256": next(iter(hashes)),
        "aggregation": "window_macro per checkpoint; paired training seed inference",
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        lower = metric in LOWER_IS_BETTER
        values = {
            arm: np.asarray(
                [
                    payload["aggregate"]["window_macro"][metric]
                    for payload in arm_payloads
                ]
            )
            for arm, arm_payloads in payloads.items()
        }
        comparisons = {
            "rc_effect_under_seq": values["seq_rc"] - values["seq_raw"],
            "return_effect_under_raw": values["return_raw"] - values["seq_raw"],
            "return_effect_under_rc": values["return_rc"] - values["seq_rc"],
            "interaction": (
                values["return_rc"]
                - values["seq_rc"]
                - values["return_raw"]
                + values["seq_raw"]
            ),
        }
        report["metrics"][metric] = {
            "arm_mean": {arm: float(array.mean()) for arm, array in values.items()},
            **{
                name: _paired(
                    delta,
                    lower,
                    args.bootstrap,
                    2027 + 100 * metric_index + comparison_index,
                )
                for comparison_index, (name, delta) in enumerate(comparisons.items())
            },
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    for metric in METRICS:
        row = report["metrics"][metric]["return_effect_under_rc"]
        print(
            f"[return_rc - seq_rc] {metric:>10s} delta={row['mean']:+.8f} "
            f"wins={row['wins']}/{row['n']} t={row['paired_t']:+.2f} "
            f"p={row['two_sided_p']:.4f} exact={row['exact_one_sided_sign_flip_p']:.5f} "
            f"CI95=[{row['seed_bootstrap95'][0]:+.8f},{row['seed_bootstrap95'][1]:+.8f}]"
        )
    print(f"saved {args.out}\nMSP_REEVAL_ANALYSIS_OK")


if __name__ == "__main__":
    main()
