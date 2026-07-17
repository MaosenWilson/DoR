"""Audit whether verifier residual interactions accumulate through temporal returns.

The input is the frozen-candidate cache produced by
``scripts/cache_temporal_reliability.py``. No world-model training is performed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _pearson_rows(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.zeros(len(a), dtype=np.float64)
    valid = den > 0
    out[valid] = (a[valid] * b[valid]).sum(axis=1) / den[valid]
    return out


def _flip_rows(score, reference):
    ds = score[:, :, None] - score[:, None, :]
    dr = reference[:, :, None] - reference[:, None, :]
    upper = np.triu(np.ones(score.shape[1], dtype=bool), 1)
    valid = upper & (ds != 0) & (dr != 0)
    return ((ds * dr < 0) & valid).sum(axis=(1, 2)) / np.maximum(
        valid.sum(axis=(1, 2)), 1
    )


def _discounted_from_start(reward, start, gamma):
    """reward [M,H,K] -> discounted return [M,K] from ``start``."""
    powers = gamma ** np.arange(reward.shape[1] - start, dtype=np.float64)
    return np.einsum("h,mhk->mk", powers, reward[:, start:])


def _cluster_boot(values, episodes, rounds, seed):
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    index = {episode: np.flatnonzero(episodes == episode) for episode in unique}
    rng = np.random.default_rng(seed)
    result = np.empty(rounds, dtype=np.float64)
    for bi in range(rounds):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[episode] for episode in chosen])
        result[bi] = values[idx].mean()
    return result


def _summary(values, episodes, rounds, seed):
    boot = _cluster_boot(values, episodes, rounds, seed)
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def audit(cache, gamma=0.95, bootstrap=2000, seed=5701):
    data = np.load(cache, allow_pickle=False)
    raw = data["raw_reward"].astype(np.float64)
    rc = data["rc_reward"].astype(np.float64)
    reference = data["code_reward"].astype(np.float64)
    if raw.shape != rc.shape or raw.shape != reference.shape or raw.ndim != 4:
        raise ValueError("raw/rc/code rewards must share shape [repeat,window,horizon,K]")
    repetitions, windows, horizons, group = raw.shape
    episode = np.asarray(data["episode"])
    if episode.shape != (windows,):
        raise ValueError("episode must have one label per window")

    raw = raw.reshape(repetitions * windows, horizons, group)
    rc = rc.reshape(repetitions * windows, horizons, group)
    reference = reference.reshape(repetitions * windows, horizons, group)
    episodes = np.broadcast_to(episode[None, :], (repetitions, windows)).reshape(-1)
    horizon_labels = np.asarray(data["horizon"]).astype(int)

    starts = []
    all_drho, all_dflip, all_episodes = [], [], []
    residual_dispersion = []
    for start in range(horizons):
        raw_return = _discounted_from_start(raw, start, gamma)
        rc_return = _discounted_from_start(rc, start, gamma)
        ref_return = _discounted_from_start(reference, start, gamma)
        interaction_return = _discounted_from_start(raw - rc, start, gamma)
        drho = _pearson_rows(rc_return, ref_return) - _pearson_rows(raw_return, ref_return)
        dflip = _flip_rows(rc_return, ref_return) - _flip_rows(raw_return, ref_return)
        dispersion = interaction_return.std(axis=1)
        starts.append({
            "start_horizon": int(horizon_labels[start]),
            "remaining_rewards": int(horizons - start),
            "delta_rho": _summary(drho, episodes, bootstrap, seed + 10 * start),
            "delta_flip": _summary(dflip, episodes, bootstrap, seed + 10 * start + 1),
            "residual_dispersion": _summary(
                dispersion, episodes, bootstrap, seed + 10 * start + 2
            ),
        })
        all_drho.append(drho)
        all_dflip.append(dflip)
        all_episodes.append(episodes)
        residual_dispersion.append(dispersion)

    aggregate_drho = np.concatenate(all_drho)
    aggregate_dflip = np.concatenate(all_dflip)
    aggregate_episodes = np.concatenate(all_episodes)
    # Prefix labels prevent horizons/repetitions from being treated as independent clusters.
    aggregate_episodes = np.asarray([str(value) for value in aggregate_episodes])
    early_minus_late = residual_dispersion[0] - residual_dispersion[-1]
    aggregate = {
        "delta_rho": _summary(aggregate_drho, aggregate_episodes, bootstrap, seed + 1000),
        "delta_flip": _summary(aggregate_dflip, aggregate_episodes, bootstrap, seed + 1001),
        "early_minus_late_residual_dispersion": _summary(
            early_minus_late, episodes, bootstrap, seed + 1002
        ),
    }
    mechanism_pass = bool(
        aggregate["delta_rho"]["ci95"][0] > 0
        and aggregate["delta_flip"]["ci95"][1] < 0
        and aggregate["early_minus_late_residual_dispersion"]["ci95"][0] > 0
    )
    return {
        "cache": os.path.abspath(cache),
        "shape": {"repetitions": repetitions, "windows": windows,
                  "horizons": horizons, "group": group},
        "gamma": gamma,
        "bootstrap": bootstrap,
        "cluster": "episode",
        "starts": starts,
        "aggregate": aggregate,
        "mechanism_pass": mechanism_pass,
        "scope": "diagnostic only; C3 also requires the fixed-n=10 training interaction",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5701)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(args.cache, args.gamma, args.bootstrap, args.seed)
    for row in report["starts"]:
        print(
            f"[h={row['start_horizon']}] terms={row['remaining_rewards']} "
            f"drho={row['delta_rho']['mean']:+.4f} "
            f"dflip={row['delta_flip']['mean']:+.4f} "
            f"resid-sd={row['residual_dispersion']['mean']:.5f}"
        )
    aggregate = report["aggregate"]
    print(
        "[aggregate] "
        f"drho={aggregate['delta_rho']['mean']:+.4f} "
        f"CI={aggregate['delta_rho']['ci95']} | "
        f"dflip={aggregate['delta_flip']['mean']:+.4f} "
        f"CI={aggregate['delta_flip']['ci95']} | "
        f"early-late={aggregate['early_minus_late_residual_dispersion']['mean']:+.5f} "
        f"CI={aggregate['early_minus_late_residual_dispersion']['ci95']}"
    )
    print(f"[verdict] {'GREEN' if report['mechanism_pass'] else 'RED'}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved {args.out}\nCALIBRATION_CREDIT_COUPLING_OK")


if __name__ == "__main__":
    main()
