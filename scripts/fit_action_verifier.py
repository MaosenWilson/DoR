"""Fit the grouped ridge verifier and run the real-transition Gate A/D1."""
import argparse
import json
import os

import numpy as np

from dor.action_verifier import ARM_MOTION_DIMS, fit_grouped_ridge, save_payload
from dor.constants import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{ROOT}/outputs/rcav/action_transitions.npz")
    ap.add_argument("--model_out", default=f"{ROOT}/outputs/rcav/action_verifier.npz")
    ap.add_argument("--report_out", default=f"{ROOT}/outputs/rcav/action_verifier_gate_d1.json")
    ap.add_argument("--split_seed", type=int, default=2027)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--permutation_seed", type=int, default=7301)
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=False)
    names = data["episode_names"].astype(str)
    rows = names[data["episode_id"].astype(int)]
    payload, report = fit_grouped_ridge(
        data["features"], data["actions"], rows,
        action_dims=ARM_MOTION_DIMS,
        split_seed=args.split_seed,
        permutations=args.permutations,
        permutation_seed=args.permutation_seed,
    )
    metadata = {
        "feature_pool_hw": data["pool_hw"].astype(int).tolist(),
        "feature_stride": int(data["stride"].item()),
        "source_cache": os.path.abspath(args.cache),
    }
    save_payload(args.model_out, payload, metadata)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    with open(args.report_out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== RCAV Gate A / D1: Real Transitions ===")
    print(f"split_counts={report['split_counts']} alpha={report['selected_alpha']}")
    print(f"R2={report['test_mean_r2']:+.4f} per_dim="
          + ",".join(f"{x:+.3f}" for x in report["test_r2_per_dim"]))
    print(f"positive_dims={report['test_positive_dims']}/6 "
          f"NRMSE={report['test_mean_nrmse']:.4f} "
          f"mean_baseline={report['mean_baseline_nrmse']:.4f}")
    print(f"permutation_p={report['permutation_p']:.5f} "
          f"null_q95={report['permutation_q95']:+.4f}")
    print(f"[verdict] {'GREEN' if report['green'] else 'RED'}")
    print(f"saved {args.model_out} and {args.report_out}\nACTION_VERIFIER_D1_OK", flush=True)


if __name__ == "__main__":
    main()
