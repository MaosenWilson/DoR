"""Exploratory zero-training gate for fidelity-constrained group rewards.

This script is deliberately labelled PROVISIONAL: the current RC-Energy cache
was already inspected while designing these candidates. A positive result must
be repeated on a cache with new generation seeds before any GRPO training.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dor.energy_verifier import (
    combine_block_distances,
    episode_bootstrap,
    pair_uncertainty_threshold,
    project_certified_order,
    project_reliable_order,
    radial_residual_reward,
    rowwise_correlation,
    top_safe_energy_reward,
)


def _take_rows(values, indices):
    return values[np.arange(len(values)), indices]


def _bootstrap_delta(values, episodes, rounds, seed):
    return episode_bootstrap(values, episodes, rounds=rounds, seed=seed)


def _report_method(
    name,
    reward,
    rc,
    utility,
    raw_lpips,
    raw_mse,
    episodes,
    rounds,
    seed,
    controls=None,
):
    rc_top = np.argmax(rc, axis=1)
    method_top = np.argmax(reward, axis=1)
    lpips_delta = _take_rows(raw_lpips, method_top) - _take_rows(raw_lpips, rc_top)
    mse_base = _take_rows(raw_mse, rc_top)
    mse_delta = (_take_rows(raw_mse, method_top) - mse_base) / np.maximum(mse_base, 1e-12)
    correlations = {}
    for offset, method in enumerate(("pearson", "spearman")):
        base = rowwise_correlation(rc, utility, method)
        candidate = rowwise_correlation(reward, utility, method)
        correlations[method] = {
            "rc": float(np.nanmean(base)),
            "candidate": float(np.nanmean(candidate)),
            "delta": _bootstrap_delta(candidate - base, episodes, rounds, seed + offset),
        }
        if controls:
            correlations[method]["gain_over_controls"] = {
                control_name: _bootstrap_delta(
                    candidate - rowwise_correlation(control_reward, utility, method),
                    episodes,
                    rounds,
                    seed + 30 + 10 * control_index + offset,
                )
                for control_index, (control_name, control_reward) in enumerate(controls.items())
            }
    return {
        "name": name,
        "correlations": correlations,
        "top_candidate": {
            "lpips_delta": _bootstrap_delta(lpips_delta, episodes, rounds, seed + 10),
            "mse_relative_delta": _bootstrap_delta(mse_delta, episodes, rounds, seed + 11),
            "same_choice_fraction": float(np.mean(rc_top == method_top)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument(
        "--protocol_role",
        choices=("exploratory", "confirmation"),
        default="exploratory",
    )
    parser.add_argument("--uncertainty_quantile", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--lpips_margin", type=float, default=0.002)
    parser.add_argument("--mse_relative_margin", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=False)
    is_scale = np.asarray(cache["is_scale"], dtype=bool)
    gate = ~is_scale
    scales = np.asarray(cache["block_scales"], dtype=np.float64)
    episodes = np.asarray(cache["episode"])[gate]
    raw_scale, reach_scale = [], []
    for group in range(2):
        raw_scale.append(combine_block_distances(cache["raw_blocks"][group, is_scale], scales))
        reach_scale.append(
            combine_block_distances(cache["reach_blocks"][group, is_scale], scales)
        )
    threshold = pair_uncertainty_threshold(
        np.concatenate(raw_scale),
        np.concatenate(reach_scale),
        quantile=args.uncertainty_quantile,
    )

    directions = []
    method_green = {
        name: []
        for name in ("certified_rank_safe", "rank_safe", "radial_residual", "top_safe")
    }
    for group in range(2):
        reach = combine_block_distances(cache["reach_blocks"][group, gate], scales)
        raw = combine_block_distances(cache["raw_blocks"][group, gate], scales)
        within = combine_block_distances(cache["within_blocks"][group, gate], scales)
        cross = combine_block_distances(cache["cross_blocks"][group, gate], scales)
        rc = -reach
        pair = within.sum(axis=2) / (reach.shape[1] - 1)
        energy = rc + pair
        utility = -raw + cross.mean(axis=2)
        candidates = {
            "certified_rank_safe": project_certified_order(energy, rc, raw, reach),
            "rank_safe": project_reliable_order(energy, rc, threshold),
            "radial_residual": radial_residual_reward(rc, pair, reach),
        }
        candidates["top_safe"], coefficients = top_safe_energy_reward(rc, pair)
        controls = {
            "certified_rank_safe": {
                "reversed": project_certified_order(rc - pair, rc, raw, reach),
                "shuffled": project_certified_order(
                    rc + np.roll(pair, 1, axis=0), rc, raw, reach
                ),
            }
        }
        reports = []
        for index, (name, reward) in enumerate(candidates.items()):
            report = _report_method(
                name,
                reward,
                rc,
                utility,
                cache["raw_lpips"][group, gate].astype(np.float64),
                cache["raw_mse"][group, gate].astype(np.float64),
                episodes,
                args.bootstrap,
                args.seed + 100 * group + 20 * index,
                controls=controls.get(name),
            )
            pearson = report["correlations"]["pearson"]["delta"]
            spearman = report["correlations"]["spearman"]["delta"]
            top = report["top_candidate"]
            green = (
                pearson["q05"] > 0.0
                and spearman["q05"] > 0.0
                and top["lpips_delta"]["q95"] <= args.lpips_margin
                and top["mse_relative_delta"]["q95"] <= args.mse_relative_margin
            )
            if name == "certified_rank_safe":
                control_gains = report["correlations"]["pearson"]["gain_over_controls"]
                green = green and all(
                    value["q05"] > 0.0 for value in control_gains.values()
                )
            method_green[name].append(bool(green))
            report["provisional_green"] = bool(green)
            if name == "top_safe":
                report["coefficient"] = {
                    "mean": float(coefficients.mean()),
                    "median": float(np.median(coefficients)),
                    "q05": float(np.quantile(coefficients, 0.05)),
                    "q95": float(np.quantile(coefficients, 0.95)),
                }
            control_text = ""
            if name == "certified_rank_safe":
                gains = report["correlations"]["pearson"]["gain_over_controls"]
                control_text = (
                    f" ctrlQ05=rev:{gains['reversed']['q05']:+.4f}"
                    f"/shuf:{gains['shuffled']['q05']:+.4f}"
                )
            positive_label = "GREEN" if args.protocol_role == "confirmation" else "PROVISIONAL-GREEN"
            print(
                f"[{name} {group}->{1-group}] "
                f"dPearson={pearson['mean']:+.4f} "
                f"CI90=[{pearson['q05']:+.4f},{pearson['q95']:+.4f}] "
                f"dSpearman={spearman['mean']:+.4f} "
                f"CI90=[{spearman['q05']:+.4f},{spearman['q95']:+.4f}] "
                f"LPIPSq95={top['lpips_delta']['q95']:+.5f} "
                f"MSEq95={top['mse_relative_delta']['q95']:+.2%} "
                f"same={top['same_choice_fraction']:.3f}{control_text} "
                f"=> {positive_label if green else 'RED'}",
                flush=True,
            )
            reports.append(report)
        directions.append({"group": group, "reports": reports})

    positive_label = "GREEN" if args.protocol_role == "confirmation" else "PROVISIONAL-GREEN"
    verdicts = {
        name: positive_label if all(values) else "RED"
        for name, values in method_green.items()
    }
    payload = {
        "protocol": {
            "cache": os.path.abspath(args.cache),
            "protocol_role": args.protocol_role,
            "status": (
                "independent_candidate_seed_confirmation"
                if args.protocol_role == "confirmation"
                else "exploratory_posthoc_requires_independent_confirmation"
            ),
            "uncertainty_quantile": args.uncertainty_quantile,
            "uncertainty_threshold": threshold,
            "bootstrap": args.bootstrap,
            "cluster": "episode",
            "scale_windows": int(is_scale.sum()),
            "gate_windows": int(gate.sum()),
            "lpips_margin": args.lpips_margin,
            "mse_relative_margin": args.mse_relative_margin,
        },
        "directions": directions,
        "verdicts": verdicts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[verdicts] {verdicts}\nsaved {args.out}", flush=True)
    print("CONSTRAINED_ENERGY_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
