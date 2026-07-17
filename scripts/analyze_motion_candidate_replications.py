"""Combine two locked candidate-generation replications at context level."""
import argparse
import json
import os

import numpy as np

from dor.constants import ROOT


def episode_bootstrap(values, episodes, n_boot, seed):
    values = np.asarray(values, np.float64)
    episodes = np.asarray(episodes, dtype=str)
    unique = np.unique(episodes)
    means = np.asarray([values[episodes == ep].mean() for ep in unique])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(int(n_boot), len(means)))].mean(1)
    return {
        "mean": float(means.mean()),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
        "episode_means": means.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep1", default=f"{ROOT}/outputs/rcav/motion_action_candidate_gate.npz")
    ap.add_argument("--rep2", default=f"{ROOT}/outputs/rcav/motion_action_candidate_gate_rep2.npz")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/motion_action_candidate_gate_combined.json")
    args = ap.parse_args()
    a = np.load(args.rep1, allow_pickle=False)
    b = np.load(args.rep2, allow_pickle=False)
    for key in ("episode", "start"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"replications do not share identical {key} rows")
    episodes = a["episode"].astype(str)
    primary = ("rho_dmotion", "delta_rho_command", "top_bottom_dmotion")
    combined = {}
    replicate_means = {}
    for offset, key in enumerate(primary):
        av = a[key].astype(np.float64)
        bv = b[key].astype(np.float64)
        replicate_means[key] = [float(av.mean()), float(bv.mean())]
        combined[key] = episode_bootstrap(
            0.5 * (av + bv), episodes, args.bootstrap, args.seed + offset
        )
    nondegenerate = [
        float(np.mean(a["action_std"] > 1e-4)),
        float(np.mean(b["action_std"] > 1e-4)),
    ]
    median_abs_state = float(np.median(np.abs(np.concatenate((a["rho_state"], b["rho_state"])))))
    directional = bool(
        all(x > 0 for x in replicate_means["rho_dmotion"])
        and all(x > 0 for x in replicate_means["delta_rho_command"])
    )
    green = bool(
        min(nondegenerate) >= 0.80
        and median_abs_state < 0.90
        and directional
        and combined["rho_dmotion"]["q05"] > 0
        and combined["delta_rho_command"]["q05"] > 0
    )
    secondary = {}
    for key in ("rho_raw", "rho_flow", "top_bottom_lpips", "top_bottom_mse"):
        secondary[key] = {
            "replicate_means": [float(np.mean(a[key])), float(np.mean(b[key]))],
            "combined_mean": float(np.mean(np.concatenate((a[key], b[key])))),
        }
    report = {
        "n_contexts": int(len(episodes)),
        "n_episodes": int(len(np.unique(episodes))),
        "replicate_means": replicate_means,
        "combined_context_mean_bootstrap90": combined,
        "nondegenerate_fraction_by_rep": nondegenerate,
        "median_abs_rho_state_pooled": median_abs_state,
        "directionally_consistent": directional,
        "secondary": secondary,
        "green": green,
        "stopping_rule": "no third generation seed after combined RED",
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== RCAV Gate B2: Two-Replication Combined Decision ===")
    print("rho reps=" + ",".join(f"{x:+.3f}" for x in replicate_means["rho_dmotion"]))
    rho = combined["rho_dmotion"]
    print(f"rho combined={rho['mean']:+.3f} CI90=[{rho['q05']:+.3f},{rho['q95']:+.3f}]")
    print("delta reps=" + ",".join(f"{x:+.3f}" for x in replicate_means["delta_rho_command"]))
    delta = combined["delta_rho_command"]
    print(f"delta combined={delta['mean']:+.3f} CI90=[{delta['q05']:+.3f},{delta['q95']:+.3f}]")
    print(f"directionally_consistent={directional} median|rho(action,RC)|={median_abs_state:.3f}")
    print(f"[verdict] {'GREEN' if green else 'RED'}")
    print(f"saved {out}\nMOTION_ACTION_REPLICATION_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
