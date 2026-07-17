"""Paired RA-RC analysis with one metric contract across all platforms."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


ARMS = ("raw", "rc", "ra_rc")
METRICS = ("lpips", "mse", "psnr", "ssim", "latent")
LOWER_IS_BETTER = {"lpips": True, "mse": True, "psnr": False, "ssim": False, "latent": True}


def _arm_name(platform: str, arm: str) -> str:
    if platform == "rt1":
        return {"raw": "a0faithful", "rc": "a0faithful_tok", "ra_rc": "ra_rc"}[arm]
    return arm


def _pattern(platform: str, arm: str, credit: str) -> re.Pattern:
    escaped = re.escape(_arm_name(platform, arm))
    if platform == "rt1":
        return re.compile(rf"^sweep_{escaped}_gt_only_s(?P<seed>\d+)\.json$")
    if platform == "vp2":
        return re.compile(rf"^sweep_{escaped}_{re.escape(credit)}_s(?P<seed>\d+)\.json$")
    return re.compile(rf"^sweep_{escaped}_s(?P<seed>\d+)\.json$")


def _run_payload(payload: dict, platform: str, arm: str) -> dict:
    run = payload["run"]
    if platform == "rt1":
        key = f"{_arm_name(platform, arm)}-gt_only"
        return run[key]
    return run


def _final_metrics(run: dict, platform: str) -> dict[str, float]:
    latent_key = "eval_code_rms" if platform == "rt1" else "eval_latent_rms"
    keys = {
        "lpips": "eval_lpips",
        "mse": "eval_mse",
        "psnr": "eval_psnr",
        "ssim": "eval_ssim",
        "latent": latent_key,
    }
    values = {}
    for metric, key in keys.items():
        series = run.get(key)
        if not isinstance(series, list) or not series:
            raise ValueError(f"missing non-empty metric series {key}")
        value = float(series[-1])
        if not math.isfinite(value):
            raise ValueError(f"non-finite final metric {key}")
        values[metric] = value
    return values


def _load(directory: Path, platform: str, credit: str) -> dict[str, dict[int, dict]]:
    result = {arm: {} for arm in ARMS}
    for arm in ARMS:
        pattern = _pattern(platform, arm, credit)
        for path in directory.glob("sweep_*.json"):
            match = pattern.match(path.name)
            if not match:
                continue
            seed = int(match.group("seed"))
            payload = json.loads(path.read_text())
            result[arm][seed] = _final_metrics(
                _run_payload(payload, platform, arm), platform
            )
    common = sorted(set.intersection(*(set(result[arm]) for arm in ARMS)))
    if not common:
        raise ValueError("no seeds have complete raw/RC/RA-RC triples")
    return {
        arm: {seed: result[arm][seed] for seed in common}
        for arm in ARMS
    }


def _paired_summary(delta: np.ndarray, lower_is_better: bool, rounds: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(rounds, len(delta)))].mean(axis=1)
    improved = delta < 0.0 if lower_is_better else delta > 0.0
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "wins": int(improved.sum()),
        "n": int(len(delta)),
        "per_seed_delta": delta.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("rt1", "vp2", "iris"), required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--credit", default="seq", help="VP2 filename credit suffix")
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--expected_n",
        type=int,
        default=None,
        help="fail rather than analyze an incomplete paired sweep",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = _load(Path(args.input_dir), args.platform, args.credit)
    seeds = sorted(runs["ra_rc"])
    expected_n = args.expected_n or (3 if args.stage == "pilot" else 10)
    if len(seeds) != expected_n:
        raise ValueError(
            f"expected {expected_n} complete raw/RC/RA-RC seed triples, found {len(seeds)}: {seeds}"
        )
    report = {
        "protocol": "RA-RC paired cross-platform analysis v1",
        "platform": args.platform,
        "stage": args.stage,
        "seeds": seeds,
        "comparisons": {},
    }
    for comparator in ("raw", "rc"):
        rows = {}
        for offset, metric in enumerate(METRICS):
            candidate = np.asarray([runs["ra_rc"][seed][metric] for seed in seeds])
            baseline = np.asarray([runs[comparator][seed][metric] for seed in seeds])
            rows[metric] = _paired_summary(
                candidate - baseline,
                LOWER_IS_BETTER[metric],
                args.bootstrap,
                args.seed + 100 * offset + (0 if comparator == "raw" else 1000),
            )
            rows[metric]["candidate_mean"] = float(candidate.mean())
            rows[metric]["baseline_mean"] = float(baseline.mean())
            rows[metric]["relative_mean_delta"] = float(
                (candidate.mean() - baseline.mean()) / max(abs(baseline.mean()), 1e-20)
            )
        report["comparisons"][f"ra_rc_minus_{comparator}"] = rows

    raw_lpips = report["comparisons"]["ra_rc_minus_raw"]["lpips"]
    raw_latent = report["comparisons"]["ra_rc_minus_raw"]["latent"]
    raw_mse = report["comparisons"]["ra_rc_minus_raw"]["mse"]
    raw_ssim = report["comparisons"]["ra_rc_minus_raw"]["ssim"]
    rc_lpips = report["comparisons"]["ra_rc_minus_rc"]["lpips"]
    if args.stage == "pilot":
        required_wins = math.ceil(2 * len(seeds) / 3)
        green = (
            len(seeds) >= 3
            and raw_lpips["mean_delta"] < 0.0
            and raw_lpips["wins"] >= required_wins
            and raw_latent["mean_delta"] < 0.0
            and raw_latent["wins"] >= required_wins
            and rc_lpips["mean_delta"] <= 0.0
            and raw_mse["relative_mean_delta"] <= 0.05
            and raw_ssim["mean_delta"] >= -0.01
        )
    else:
        required_wins = math.ceil(0.8 * len(seeds))
        green = (
            len(seeds) >= 10
            and raw_lpips["mean_delta"] < 0.0
            and raw_lpips["wins"] >= required_wins
            and raw_lpips["ci95"][1] < 0.0
            and raw_latent["mean_delta"] < 0.0
            and raw_latent["wins"] >= required_wins
            and raw_latent["ci95"][1] < 0.0
            and rc_lpips["mean_delta"] <= 0.0
            and raw_mse["ci95"][0] <= 0.0
            and raw_ssim["ci95"][1] >= 0.0
        )
    report["decision"] = {
        "required_wins": required_wins,
        "verdict": "GREEN" if green else "RED",
        "note": (
            "Pilot non-catastrophe bounds are MSE relative delta <=5% and SSIM delta >=-0.01; "
            "full runs require no significant MSE/SSIM degradation."
        ),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n=== RA-RC {args.platform.upper()} {args.stage.upper()} ===")
    for comparison, metrics in report["comparisons"].items():
        print(comparison)
        for metric, row in metrics.items():
            print(
                f"  {metric:>7s} delta={row['mean_delta']:+.6g} "
                f"CI95=[{row['ci95'][0]:+.6g},{row['ci95'][1]:+.6g}] "
                f"wins={row['wins']}/{row['n']}"
            )
    print(f"[verdict] {report['decision']['verdict']}")
    print(f"saved {output}\nRA_RC_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
