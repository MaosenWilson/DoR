"""Diagnose cross-horizon reward scale before adapting VP2 temporal credit.

This script is deliberately training-free.  It consumes a P1-v2 cache and asks
whether standard reward-to-go is distorted by horizon-dependent reward scale,
using the frozen tokenizer's post-quant latent return as a decoder-free readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dor.rank_stats import rowwise_spearman


def _returns(values: np.ndarray, gamma: float, *, equalize: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(f"expected [context,draw,horizon,candidate], got {values.shape}")
    source = values
    if equalize:
        source = (source - source.mean(axis=-1, keepdims=True)) / (
            source.std(axis=-1, keepdims=True) + 1e-6
        )
    output = np.zeros_like(source)
    horizon = source.shape[2]
    for start in range(horizon):
        discount = 1.0
        for future in range(start, horizon):
            output[:, :, start] += discount * source[:, :, future]
            discount *= gamma
    return output


def _episode_bootstrap(values: np.ndarray, episodes: np.ndarray, rounds: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes).astype(str)
    unique = np.unique(episodes)
    per_episode = np.asarray([
        np.nanmean(values[episodes == item]) for item in unique
    ])
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        np.nanmean(per_episode[rng.integers(0, len(per_episode), size=len(per_episode))])
        for _ in range(rounds)
    ])
    return {
        "mean": float(np.nanmean(per_episode)),
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "episodes": int(len(unique)),
    }


def _std_profile(values: np.ndarray) -> dict:
    group_std = np.std(values, axis=-1)
    mean = np.mean(group_std, axis=(0, 1))
    positive = mean[mean > 1e-12]
    span = float(np.max(positive) / np.min(positive)) if len(positive) else float("nan")
    return {"mean_group_std": mean.tolist(), "max_min_ratio": span}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=94117)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        raw = np.asarray(cache["raw_reward"])
        rc = np.asarray(cache["rc_reward"])
        reference = np.asarray(cache["post_quant_reward"])
        episodes = np.asarray(cache["episode"]).astype(str)
    if raw.shape != rc.shape or raw.shape != reference.shape:
        raise ValueError("raw, RC, and post-quant arrays must share shape")
    if raw.shape[2] < 2:
        raise ValueError("temporal-credit diagnosis requires horizon >= 2")

    oracle = _returns(reference, args.gamma, equalize=True)
    rows = {}
    correlations = {}
    for name, reward in (("raw", raw), ("rc", rc)):
        standard = rowwise_spearman(_returns(reward, args.gamma, equalize=False), oracle)
        equalized = rowwise_spearman(_returns(reward, args.gamma, equalize=True), oracle)
        correlations[name] = {
            "standard": _episode_bootstrap(standard, episodes, args.bootstrap, args.seed),
            "equalized": _episode_bootstrap(equalized, episodes, args.bootstrap, args.seed + 1),
            "equalized_minus_standard": _episode_bootstrap(
                equalized - standard, episodes, args.bootstrap, args.seed + 2
            ),
        }
        rows[name] = (standard, equalized)

    standard_rc_delta = rows["rc"][0] - rows["raw"][0]
    equalized_rc_delta = rows["rc"][1] - rows["raw"][1]
    raw_green = correlations["raw"]["equalized_minus_standard"]["ci90"][0] > 0.0
    rc_green = correlations["rc"]["equalized_minus_standard"]["ci90"][0] > 0.0
    report = {
        "protocol": "VP2 cross-horizon reward-scale diagnosis v1",
        "cache": str(Path(args.cache).resolve()),
        "gamma": args.gamma,
        "shape": list(raw.shape),
        "oracle": "discounted return of per-horizon standardized post-quant latent rewards",
        "reward_scale": {
            "raw": _std_profile(raw),
            "rc": _std_profile(rc),
            "post_quant": _std_profile(reference),
        },
        "return_rank_agreement": correlations,
        "rc_minus_raw": {
            "standard_return": _episode_bootstrap(
                standard_rc_delta, episodes, args.bootstrap, args.seed + 10
            ),
            "scale_equalized_return": _episode_bootstrap(
                equalized_rc_delta, episodes, args.bootstrap, args.seed + 11
            ),
        },
        "adaptation_verdict": "GREEN" if raw_green and rc_green else "RED",
        "decision_rule": "GREEN only if scale equalization improves return-rank agreement for both raw and RC with episode-bootstrap CI90 above zero",
        "scope_note": "offline admission gate only; a GREEN verdict still requires paired policy training",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("=== VP2 Temporal-Credit Scale Diagnosis ===")
    for name in ("raw", "rc", "post_quant"):
        profile = report["reward_scale"][name]
        print(f"{name:>10s} std={profile['mean_group_std']} span={profile['max_min_ratio']:.3f}")
    for name in ("raw", "rc"):
        value = correlations[name]["equalized_minus_standard"]
        print(
            f"{name:>10s} equalized-standard dSpearman={value['mean']:+.5f} "
            f"CI90=[{value['ci90'][0]:+.5f},{value['ci90'][1]:+.5f}]"
        )
    print(f"[verdict] {report['adaptation_verdict']}")
    print(f"saved {output}\nVP2_CREDIT_DIAGNOSIS_OK", flush=True)


if __name__ == "__main__":
    main()
