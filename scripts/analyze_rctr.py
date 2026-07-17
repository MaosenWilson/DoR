"""Paired pilot analysis for the admitted RCTR training arm."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np


METRICS = {
    "lpips": ("eval_lpips", True),
    "lpips_last": ("eval_lpips_last", True),
    "mse": ("eval_mse", True),
    "psnr": ("eval_psnr", False),
    "ssim": ("eval_ssim", False),
    "latent": ("eval_latent_rms", True),
}
CORE_PROTOCOL = (
    "checkpoint", "train_manifest", "eval_manifest", "horizon", "K", "eval_K",
    "steps", "batch_windows", "lr", "kl", "kl_type", "gamma", "data_seed",
    "grad_clip", "eval_every", "eval_seed", "deterministic",
)


def _load(pattern: str, expected_reward: str, expected_credit: str):
    result = {}
    protocol = None
    prefix = f"sweep_{expected_reward}_{expected_credit}_s"
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        if not path.name.startswith(prefix):
            raise ValueError(f"wrong arm filename {path.name!r}; expected {prefix!r}")
        match = re.search(r"_s(\d+)\.json$", path.name)
        if match is None:
            raise ValueError(f"cannot parse seed from {path}")
        payload = json.loads(path.read_text())
        args = payload["args"]
        if expected_reward not in args["rewards"].split(","):
            raise ValueError(f"reward mismatch in {path}")
        if expected_credit not in args["credits"].split(","):
            raise ValueError(f"credit mismatch in {path}")
        current = {key: args[key] for key in CORE_PROTOCOL}
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(f"protocol mismatch within arm {path}")
        run = payload["run"]
        seed = int(match.group(1))
        result[seed] = {
            name: float(run[key][-1]) for name, (key, _) in METRICS.items()
        }
    if not result:
        raise FileNotFoundError(pattern)
    return result, protocol


def _summary(delta: np.ndarray, lower: bool, rounds: int, rng) -> dict:
    boot = delta[rng.integers(0, len(delta), size=(rounds, len(delta)))].mean(axis=1)
    wins = delta < 0.0 if lower else delta > 0.0
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "wins": int(wins.sum()),
        "n": int(len(delta)),
        "per_seed": delta.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rctr", required=True)
    parser.add_argument("--seq_raw", required=True)
    parser.add_argument("--return_raw", required=True)
    parser.add_argument("--return_rc", required=True)
    parser.add_argument("--expected_n", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=19331)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    loaded = {
        "rctr": _load(args.rctr, "rctr", "return"),
        "seq_raw": _load(args.seq_raw, "raw", "seq"),
        "return_raw": _load(args.return_raw, "raw", "return"),
        "return_rc": _load(args.return_rc, "rc", "return"),
    }
    arms = {name: value[0] for name, value in loaded.items()}
    protocols = [value[1] for value in loaded.values()]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("RCTR and all baselines must share the fixed protocol")
    seeds = sorted(set.intersection(*(set(values) for values in arms.values())))
    if len(seeds) != args.expected_n:
        raise RuntimeError(f"expected {args.expected_n} paired seeds, found {seeds}")
    report = {
        "analysis_protocol": "RCTR paired analysis v2",
        "training_protocol": protocols[0],
        "seeds": seeds,
        "comparisons": {},
    }
    rng = np.random.default_rng(args.seed)
    for baseline in ("seq_raw", "return_raw", "return_rc"):
        rows = {}
        for metric, (_, lower) in METRICS.items():
            candidate = np.asarray([arms["rctr"][s][metric] for s in seeds])
            control = np.asarray([arms[baseline][s][metric] for s in seeds])
            rows[metric] = _summary(candidate - control, lower, args.bootstrap, rng)
            rows[metric]["rctr_mean"] = float(candidate.mean())
            rows[metric]["baseline_mean"] = float(control.mean())
        report["comparisons"][f"rctr_minus_{baseline}"] = rows
    primary = report["comparisons"]["rctr_minus_seq_raw"]
    required = math.ceil(2 * len(seeds) / 3)
    common = bool(
        primary["lpips"]["mean"] < 0.0
        and primary["mse"]["ci95"][0] <= 0.0
        and primary["ssim"]["ci95"][1] >= 0.0
        and all(
            report["comparisons"][f"rctr_minus_{baseline}"]["lpips"]["mean"] < 0.0
            for baseline in ("return_raw", "return_rc")
        )
    )
    if len(seeds) >= 10:
        green = bool(
            common
            and primary["lpips"]["wins"] >= 8
            and primary["lpips"]["ci95"][1] < 0.0
        )
    else:
        green = bool(common and primary["lpips"]["wins"] >= required)
    report["decision"] = {
        "primary": "final raw-GT LPIPS against seq-raw",
        "required_wins": required,
        "verdict": "GREEN" if green else "RED",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("=== RCTR VP2 Paired Pilot ===")
    for comparison, rows in report["comparisons"].items():
        print(comparison)
        for metric in ("lpips", "lpips_last", "mse", "psnr", "ssim", "latent"):
            value = rows[metric]
            print(
                f"  {metric:>10s} delta={value['mean']:+.6g} "
                f"CI95=[{value['ci95'][0]:+.6g},{value['ci95'][1]:+.6g}] "
                f"wins={value['wins']}/{value['n']}"
            )
    print(f"[verdict] {report['decision']['verdict']}")
    print(f"saved {output}\nRCTR_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
