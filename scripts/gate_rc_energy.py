"""Two-group zero-training admission gate for RC-Energy."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dor.energy_verifier import combine_block_distances, episode_bootstrap, rowwise_correlation


def _take_rows(values, indices):
    return values[np.arange(len(values)), indices]


def _direction_report(cache, group, selection, episodes, rounds, seed):
    partner = 1 - group
    scales = cache["block_scales"].astype(np.float64)
    reach = combine_block_distances(cache["reach_blocks"][group, selection], scales)
    raw = combine_block_distances(cache["raw_blocks"][group, selection], scales)
    within = combine_block_distances(cache["within_blocks"][group, selection], scales)
    cross = combine_block_distances(cache["cross_blocks"][group, selection], scales)
    k = reach.shape[1]
    pair_contribution = within.sum(axis=2) / (k - 1)
    rc = -reach
    rce = rc + pair_contribution
    utility = -raw + cross.mean(axis=2)
    reversed_reward = rc - pair_contribution
    shuffled_reward = rc + np.roll(pair_contribution, 1, axis=0)

    correlations = {}
    for method in ("pearson", "spearman"):
        base = rowwise_correlation(rc, utility, method)
        full = rowwise_correlation(rce, utility, method)
        reversed_corr = rowwise_correlation(reversed_reward, utility, method)
        shuffled_corr = rowwise_correlation(shuffled_reward, utility, method)
        delta = full - base
        correlations[method] = {
            "rc": float(np.nanmean(base)),
            "rc_energy": float(np.nanmean(full)),
            "delta": episode_bootstrap(delta, episodes, rounds, seed + (method == "spearman")),
            "gain_over_reversed": episode_bootstrap(
                full - reversed_corr, episodes, rounds, seed + 10 + (method == "spearman")
            ),
            "gain_over_shuffled": episode_bootstrap(
                full - shuffled_corr, episodes, rounds, seed + 20 + (method == "spearman")
            ),
        }

    rc_top = np.argmax(rc, axis=1)
    rce_top = np.argmax(rce, axis=1)
    raw_lpips = cache["raw_lpips"][group, selection].astype(np.float64)
    raw_mse = cache["raw_mse"][group, selection].astype(np.float64)
    lpips_delta = _take_rows(raw_lpips, rce_top) - _take_rows(raw_lpips, rc_top)
    mse_rc = _take_rows(raw_mse, rc_top)
    mse_relative_delta = (
        _take_rows(raw_mse, rce_top) - mse_rc
    ) / np.maximum(mse_rc, 1e-12)
    return {
        "group": int(group),
        "partner": int(partner),
        "correlations": correlations,
        "top_candidate": {
            "lpips_delta": episode_bootstrap(lpips_delta, episodes, rounds, seed + 30),
            "mse_relative_delta": episode_bootstrap(
                mse_relative_delta, episodes, rounds, seed + 31
            ),
            "same_choice_fraction": float(np.mean(rc_top == rce_top)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--lpips_margin", type=float, default=0.002)
    parser.add_argument("--mse_relative_margin", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=False)
    if cache["reach_blocks"].shape[0] != 2:
        raise ValueError("the pre-registered gate expects exactly two candidate groups")
    is_scale = np.asarray(cache["is_scale"], dtype=bool)
    selection = ~is_scale
    episodes = np.asarray(cache["episode"])[selection]
    if len(np.unique(episodes)) < 3:
        raise ValueError("gate split has fewer than three episodes")
    reports = [
        _direction_report(
            cache, group, selection, episodes, args.bootstrap, args.seed + 100 * group
        )
        for group in range(2)
    ]
    direction_green = []
    for report in reports:
        pearson = report["correlations"]["pearson"]
        spearman = report["correlations"]["spearman"]
        top = report["top_candidate"]
        green = (
            pearson["delta"]["q05"] > 0.0
            and spearman["delta"]["mean"] > 0.0
            and pearson["gain_over_reversed"]["q05"] > 0.0
            and pearson["gain_over_shuffled"]["q05"] > 0.0
            and top["lpips_delta"]["q95"] <= args.lpips_margin
            and top["mse_relative_delta"]["q95"] <= args.mse_relative_margin
        )
        direction_green.append(bool(green))
        print(
            f"[group {report['group']}->{report['partner']}] "
            f"pearson {pearson['rc']:+.3f}->{pearson['rc_energy']:+.3f} "
            f"delta={pearson['delta']['mean']:+.4f} "
            f"CI90=[{pearson['delta']['q05']:+.4f},{pearson['delta']['q95']:+.4f}] | "
            f"LPIPS top delta={top['lpips_delta']['mean']:+.5f} "
            f"q95={top['lpips_delta']['q95']:+.5f} | "
            f"MSE rel={top['mse_relative_delta']['mean']:+.3%} "
            f"q95={top['mse_relative_delta']['q95']:+.3%} | "
            f"{'GREEN' if green else 'RED'}",
            flush=True,
        )

    verdict = "GREEN" if all(direction_green) else "RED"
    payload = {
        "protocol": {
            "cache": os.path.abspath(args.cache),
            "bootstrap": args.bootstrap,
            "cluster": "episode",
            "scale_windows": int(is_scale.sum()),
            "gate_windows": int(selection.sum()),
            "primary": "cross-group Pearson gain, both directions",
            "lpips_margin": args.lpips_margin,
            "mse_relative_margin": args.mse_relative_margin,
        },
        "directions": reports,
        "direction_green": direction_green,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[verdict] {verdict}\nsaved {args.out}", flush=True)
    print("RC_ENERGY_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
