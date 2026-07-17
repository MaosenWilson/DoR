"""Recompute external RC rank closure from frozen caches at 95% confidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dor.rank_stats import pair_flip_fraction, rowwise_spearman


def _bootstrap(values, clusters, rounds, rng):
    unique = np.unique(clusters)
    per_cluster = np.asarray([np.nanmean(values[clusters == item]) for item in unique])
    draws = np.asarray([
        np.nanmean(per_cluster[rng.integers(0, len(per_cluster), len(per_cluster))])
        for _ in range(rounds)
    ])
    return {
        "mean": float(np.nanmean(per_cluster)),
        "ci90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "clusters": int(len(unique)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference_key", default="post_quant_reward")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as payload:
        raw = np.asarray(payload["raw_reward"])
        rc = np.asarray(payload["rc_reward"])
        reference = np.asarray(payload[args.reference_key])
        clusters = np.asarray(payload["episode"])
    raw_s, rc_s = rowwise_spearman(raw, reference), rowwise_spearman(rc, reference)
    raw_f, rc_f = pair_flip_fraction(raw, reference), pair_flip_fraction(rc, reference)
    raw_top, rc_top = np.argmax(raw, -1), np.argmax(rc, -1)
    top_delta = (
        np.take_along_axis(reference, rc_top[..., None], -1)[..., 0]
        - np.take_along_axis(reference, raw_top[..., None], -1)[..., 0]
    )
    rng = np.random.default_rng(args.seed)
    report = {
        "protocol": "frozen-cache external rank closure, 95% cluster bootstrap",
        "cache": str(Path(args.cache).resolve()),
        "reference_key": args.reference_key,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "delta_spearman": _bootstrap(rc_s - raw_s, clusters, args.bootstrap, rng),
        "delta_flip": _bootstrap(rc_f - raw_f, clusters, args.bootstrap, rng),
        "delta_reference_top": _bootstrap(top_delta, clusters, args.bootstrap, rng),
        "raw_spearman": _bootstrap(raw_s, clusters, args.bootstrap, rng),
        "rc_spearman": _bootstrap(rc_s, clusters, args.bootstrap, rng),
        "raw_flip": _bootstrap(raw_f, clusters, args.bootstrap, rng),
        "rc_flip": _bootstrap(rc_f, clusters, args.bootstrap, rng),
    }
    mechanism = (
        report["delta_spearman"]["ci95"][0] > 0
        and report["delta_flip"]["ci95"][1] < 0
    )
    selection = report["delta_reference_top"]["ci95"][0] > 0
    report["mechanism_verdict"] = "GREEN" if mechanism else "RED"
    report["selection_verdict"] = "GREEN" if selection else "RED"
    report["verdict"] = "GREEN" if mechanism and selection else "RED"
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"=== External RC Rank Closure: {Path(args.cache).parent.name} ===")
    for key in ("delta_spearman", "delta_flip", "delta_reference_top"):
        value = report[key]
        print(f"{key:>24s} {value['mean']:+.5f} CI95=[{value['ci95'][0]:+.5f},{value['ci95'][1]:+.5f}]")
    print(f"[verdict] {report['verdict']}\nsaved {output}\nEXTERNAL_RANK_REANALYSIS_OK")


if __name__ == "__main__":
    main()
