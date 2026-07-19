"""Offline admission gate for candidate-specific VP2 branch-value credit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dor.branch_credit import cluster_bootstrap, heldout_branch_rows


def _prefixes(cache) -> list[int]:
    values = []
    for name in cache.files:
        if name.startswith("future_p"):
            values.append(int(name.removeprefix("future_p")))
    return sorted(set(values))


def analyze(cache_path, *, gamma=0.95, bootstrap=2000, seed=2027, min_horizons=2):
    with np.load(cache_path, allow_pickle=False) as cache:
        prefixes = _prefixes(cache)
        if len(prefixes) < 2:
            raise ValueError("cache needs at least two consecutive prefix horizons")
        rows = {}
        green_horizons = []
        for horizon in prefixes[1:]:
            if horizon - 1 not in prefixes:
                continue
            required = (
                f"immediate_p{horizon}", f"future_p{horizon}",
                f"future_p{horizon - 1}", f"context_p{horizon}",
                f"episode_p{horizon}",
            )
            missing = [name for name in required if name not in cache]
            if missing:
                raise KeyError(f"cache lacks {missing}")
            context = np.asarray(cache[f"context_p{horizon}"]).astype(str)
            episode = np.asarray(cache[f"episode_p{horizon}"]).astype(str)
            previous_context = np.asarray(cache[f"context_p{horizon - 1}"]).astype(str)
            if not np.array_equal(context, previous_context):
                raise ValueError("candidate identity/order changes between prefix horizons")
            heldout = heldout_branch_rows(
                cache[f"immediate_p{horizon}"],
                cache[f"future_p{horizon}"],
                cache[f"future_p{horizon - 1}"],
                context,
                gamma=gamma,
                seed=seed + horizon,
            )
            # heldout rows repeat one context for every continuation draw. Map each
            # context to its episode so draws never become independent samples.
            context_episode = {
                name: episode[np.flatnonzero(context == name)[0]] for name in np.unique(context)
            }
            clusters = np.asarray([context_episode[name] for name in heldout["context"]])
            report = {
                metric: cluster_bootstrap(
                    heldout[metric], clusters, bootstrap, seed + horizon * 101 + index
                )
                for index, metric in enumerate((
                    "split_rho", "immediate_rho", "delta_rho",
                    "selection_gain", "aligned_minus_shuffled",
                ))
            }
            is_green = (
                report["split_rho"]["q05"] > 0.0
                and report["selection_gain"]["q05"] > 0.0
                and report["aligned_minus_shuffled"]["q05"] > 0.0
            )
            report["verdict"] = "GREEN" if is_green else "RED"
            rows[str(horizon)] = report
            if is_green:
                green_horizons.append(horizon)
        verdict = "GREEN" if len(green_horizons) >= int(min_horizons) else "RED"
        return {
            "protocol": "VP2 candidate-specific branch-value credit gate v1",
            "cache": str(Path(cache_path).resolve()),
            "gamma": float(gamma),
            "bootstrap": int(bootstrap),
            "minimum_green_horizons": int(min_horizons),
            "green_horizons": green_horizons,
            "rows": rows,
            "verdict": verdict,
            "scope": "offline admission only; not a downstream training claim",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min_horizons", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = analyze(
        args.cache, gamma=args.gamma, bootstrap=args.bootstrap,
        seed=args.seed, min_horizons=args.min_horizons,
    )
    print("\n=== VP2 Candidate-Specific Branch Credit Gate ===")
    for horizon, row in report["rows"].items():
        print(
            f"h={horizon} split-rho={row['split_rho']['mean']:+.3f} "
            f"d-rho={row['delta_rho']['mean']:+.3f} "
            f"selection={row['selection_gain']['mean']:+.5f} "
            f"aligned-shuffled={row['aligned_minus_shuffled']['mean']:+.5f} "
            f"=> {row['verdict']}"
        )
    print(f"[verdict] {report['verdict']} green={report['green_horizons']}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {output}\nVP2_BRANCH_CREDIT_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
