"""Pre-registered three-arm decision for the VP2 UATR pilot."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


METRICS = (
    "eval_lpips",
    "eval_lpips_last",
    "eval_mse",
    "eval_psnr",
    "eval_ssim",
    "eval_latent_rms",
)
LOWER_IS_BETTER = {
    "eval_lpips", "eval_lpips_last", "eval_mse", "eval_latent_rms"
}
PROTOCOL_KEYS = (
    "train_manifest", "eval_manifest", "horizon", "K", "eval_K",
    "eval_draws", "eval_aggregation", "steps", "batch_windows", "lr",
    "kl", "kl_type", "gamma", "data_seed", "eval_seed",
)


def _seed(path: str) -> int:
    match = re.search(r"_s(-?\d+)\.json$", path)
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def _load(pattern: str, reward: str, credit: str) -> dict[int, dict]:
    rows = {}
    for path in sorted(glob.glob(pattern)):
        payload = json.loads(Path(path).read_text())
        args = payload["args"]
        rewards = {value.strip() for value in str(args["rewards"]).split(",")}
        credits = {value.strip() for value in str(args["credits"]).split(",")}
        if reward not in rewards or credit not in credits:
            raise ValueError(f"unexpected arm in {path}")
        if not Path(path).name.startswith(f"sweep_{reward}_{credit}_s"):
            raise ValueError(f"filename does not identify {reward}/{credit}: {path}")
        run = payload["run"]
        if not run["step"] or int(run["step"][-1]) != int(args["steps"]):
            raise ValueError(f"{path} lacks its fixed final-step evaluation")
        rows[_seed(path)] = {
            "metrics": {name: float(run[name][-1]) for name in METRICS},
            "args": args,
            "protocol": run["protocol"],
            "min_raw_progress": float(min(
                run.get("train_projected_progress_ratio", [1.0])
            )),
        }
    if not rows:
        raise FileNotFoundError(pattern)
    return rows


def _protocol_signature(row: dict) -> tuple:
    args = row["args"]
    protocol = row["protocol"]
    return (
        *(json.dumps(args.get(key), sort_keys=True) for key in PROTOCOL_KEYS),
        protocol.get("train_manifest_sha256"),
        protocol.get("eval_manifest_sha256"),
        protocol.get("context_schedule_sha256"),
        json.dumps(protocol.get("train_episodes"), sort_keys=True),
        json.dumps(protocol.get("eval_episodes"), sort_keys=True),
    )


def _bootstrap(values: np.ndarray, rounds: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(rounds, len(values)))
    ].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _comparison(candidate, control, seeds, rounds, seed):
    report = {}
    for index, metric in enumerate(METRICS):
        delta = np.asarray([
            candidate[value]["metrics"][metric] - control[value]["metrics"][metric]
            for value in seeds
        ], dtype=np.float64)
        favorable = delta < 0.0 if metric in LOWER_IS_BETTER else delta > 0.0
        report[metric] = {
            "delta": delta.tolist(),
            "mean": float(delta.mean()),
            "wins": int(favorable.sum()),
            "n": len(seeds),
            "bootstrap95": _bootstrap(delta, rounds, seed + index),
        }
    return report


def analyze(
    raw,
    aligned,
    shuffled,
    *,
    bootstrap,
    seed,
    lpips_margin,
    mse_margin,
    protocol_name="VP2 UATR three-arm final-checkpoint pilot v1",
):
    paired = sorted(set(raw) & set(aligned) & set(shuffled))
    if len(paired) < 3:
        raise ValueError(f"pilot requires at least three paired seeds, got {paired}")
    signatures = {
        _protocol_signature(rows[value])
        for rows in (raw, aligned, shuffled)
        for value in paired
    }
    if len(signatures) != 1:
        raise ValueError("three arms do not share one frozen training/evaluation protocol")
    adaptive_hashes = {
        rows[value]["protocol"].get("adaptive_config_sha256")
        for rows in (aligned, shuffled)
        for value in paired
    }
    if len(adaptive_hashes) != 1 or None in adaptive_hashes:
        raise ValueError("aligned and shuffled arms do not share one frozen adaptive config")
    first = aligned[paired[0]]
    if first["args"].get("eval_aggregation") != "episode_macro":
        raise ValueError("pilot requires episode_macro evaluation")
    if int(first["args"].get("eval_draws", 1)) < 2:
        raise ValueError("pilot requires at least two evaluation draws")
    coefficients = np.asarray(
        first["protocol"]["adaptive_coefficients"], dtype=np.float64
    )
    active_blocks = int(np.count_nonzero(coefficients > 0.0))
    aligned_raw = _comparison(aligned, raw, paired, bootstrap, seed)
    aligned_shuffled = _comparison(
        aligned, shuffled, paired, bootstrap, seed + 100
    )
    min_progress = min(
        aligned[value]["min_raw_progress"] for value in paired
    )
    min_shuffled_progress = min(
        shuffled[value]["min_raw_progress"] for value in paired
    )
    lpips = aligned_raw["eval_lpips"]
    last = aligned_raw["eval_lpips_last"]
    mse = aligned_raw["eval_mse"]
    temporal = aligned_shuffled["eval_lpips_last"]
    green = bool(
        active_blocks >= 2
        and lpips["mean"] < 0.0 and lpips["wins"] >= 2
        and lpips["bootstrap95"][1] <= lpips_margin
        and last["mean"] < 0.0 and last["wins"] >= 2
        and mse["mean"] <= mse_margin
        and temporal["mean"] < 0.0 and temporal["wins"] >= 2
        and min_progress >= 1.0 - 1e-5
        and min_shuffled_progress >= 1.0 - 1e-5
    )
    return {
        "protocol": protocol_name,
        "paired_seeds": paired,
        "active_blocks": active_blocks,
        "coefficients": coefficients.tolist(),
        "uatr_minus_seq_raw": aligned_raw,
        "uatr_minus_shuffled": aligned_shuffled,
        "minimum_projected_raw_progress": min_progress,
        "minimum_shuffled_raw_progress": min_shuffled_progress,
        "margins": {
            "lpips_bootstrap_upper": lpips_margin,
            "mse_mean": mse_margin,
        },
        "verdict": "PROVISIONAL-GREEN" if green else "RED",
        "scope": "pilot admission only; fixed final checkpoint, no best-step selection",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--uatr", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--uatr_reward", default="uatr")
    parser.add_argument("--shuffled_reward", default="uatr_shuffled")
    parser.add_argument(
        "--protocol_name",
        default="VP2 UATR three-arm final-checkpoint pilot v1",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=27183)
    parser.add_argument("--lpips_margin", type=float, default=0.00015)
    parser.add_argument("--mse_margin", type=float, default=0.00003)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = analyze(
        _load(args.raw, "raw", "seq"),
        _load(args.uatr, args.uatr_reward, "adaptive"),
        _load(args.shuffled, args.shuffled_reward, "adaptive"),
        bootstrap=args.bootstrap,
        seed=args.seed,
        lpips_margin=args.lpips_margin,
        mse_margin=args.mse_margin,
        protocol_name=args.protocol_name,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("=== VP2 UATR Pilot ===")
    for comparison in ("uatr_minus_seq_raw", "uatr_minus_shuffled"):
        print(comparison)
        for metric, row in report[comparison].items():
            print(
                f"  {metric:>18s} delta={row['mean']:+.8f} "
                f"CI95=[{row['bootstrap95'][0]:+.8f},{row['bootstrap95'][1]:+.8f}] "
                f"wins={row['wins']}/{row['n']}"
            )
    print(
        f"active_blocks={report['active_blocks']} "
        f"minRawProgress={report['minimum_projected_raw_progress']:.6f} "
        f"[verdict] {report['verdict']}"
    )
    print(f"saved {output}\nVP2_UATR_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
