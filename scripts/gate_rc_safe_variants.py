"""Zero-training gate for conservative extensions of RC verification.

The eligible variants are fixed before reading outcomes:

* ``snr_shrink`` retains raw supervision when the candidate-dependent decoder
  interaction is small and approaches RC as residual interaction dominates;
* ``orthogonal_rc`` adds only the component of the RC advantage correction
  orthogonal to the raw-GT advantage, then restores GRPO unit variance.

Fixed-half blending and a hard RIR switch are reported as controls only.  A
variant must retain rank repair on both tokenizer caches and improve the raw-GT
top-candidate joint distortion relative to full RC before any policy training.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


ELIGIBLE = ("snr_shrink", "orthogonal_rc")
CONTROLS = ("half", "hard_rir")


def _zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + eps
    )


def _pearson_rows(left, right, eps=1e-12):
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    den = np.sqrt((left * left).sum(axis=1) * (right * right).sum(axis=1))
    out = np.zeros(len(left), dtype=np.float64)
    valid = den > eps
    out[valid] = (left[valid] * right[valid]).sum(axis=1) / den[valid]
    return out


def _variants(raw, rc):
    raw = np.asarray(raw, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    rir = np.std(raw - rc, axis=1) / (np.std(rc, axis=1) + 1e-8)
    weight = rir * rir / (1.0 + rir * rir)
    variants = {
        "raw": raw,
        "rc": rc,
        "snr_shrink": (1.0 - weight[:, None]) * raw + weight[:, None] * rc,
        "half": 0.5 * raw + 0.5 * rc,
        "hard_rir": np.where((rir >= 1.0)[:, None], rc, raw),
    }

    primary = _zscore(raw)
    corrected = _zscore(rc)
    correction = corrected - primary
    dot = (primary * correction).sum(axis=1)
    norm = (primary * primary).sum(axis=1) + 1e-12
    coefficient = np.minimum(dot / norm, 0.0)
    projected = correction - coefficient[:, None] * primary
    projected_norm = np.sqrt((projected * projected).sum(axis=1))
    cap = np.minimum(1.0, np.sqrt(norm) / (projected_norm + 1e-12))
    variants["orthogonal_rc"] = _zscore(primary + cap[:, None] * projected)
    return variants, rir, weight


def _cluster_boot(values, episodes, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    grouped = np.asarray([values[episodes == episode].mean() for episode in unique])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        choice = rng.integers(0, len(grouped), size=len(grouped))
        boot[index] = grouped[choice].mean()
    return {
        "mean": float(grouped.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "episodes": int(len(grouped)),
    }


def _audit_rank(raw, rc, ref, episodes, draws, seed):
    variants, rir, weight = _variants(raw, rc)
    rho = {name: _pearson_rows(score, ref) for name, score in variants.items()}
    rows = {}
    for offset, name in enumerate(ELIGIBLE + CONTROLS):
        rows[name] = {
            "delta_rho_vs_raw": _cluster_boot(
                rho[name] - rho["raw"], episodes, draws, seed + offset
            ),
            "delta_rho_vs_rc": _cluster_boot(
                rho[name] - rho["rc"], episodes, draws, seed + 20 + offset
            ),
            "repair_retention": float(
                np.mean(rho[name] - rho["raw"])
                / (np.mean(rho["rc"] - rho["raw"]) + 1e-12)
            ),
        }
    return rows, variants, {
        "rir_mean": float(rir.mean()),
        "snr_weight_mean": float(weight.mean()),
        "snr_weight_q10_q50_q90": [
            float(x) for x in np.quantile(weight, [0.1, 0.5, 0.9])
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_cache", default="outputs/rankcal/calibration_spatial.npz")
    parser.add_argument(
        "--multi_cache", default="outputs/analysis/temporal_reliability_cache.npz"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--min_repair_retention", type=float, default=0.50)
    parser.add_argument("--out", default="outputs/analysis/rc_safe_variants_gate.json")
    args = parser.parse_args()

    single = np.load(args.single_cache, allow_pickle=False)
    single_raw = -(single["raw_lpips"] + single["raw_mse"]).astype(np.float64)
    single_rc = -(single["reach_lpips"] + single["reach_mse"]).astype(np.float64)
    single_rank, single_scores, single_weight = _audit_rank(
        single_raw,
        single_rc,
        single["code"].astype(np.float64),
        single["episode"],
        args.bootstrap,
        args.seed,
    )

    multi = np.load(args.multi_cache, allow_pickle=False)
    generation, windows, horizons, group = multi["raw_reward"].shape
    multi_raw = multi["raw_reward"].astype(np.float64).reshape(-1, group)
    multi_rc = multi["rc_reward"].astype(np.float64).reshape(-1, group)
    multi_ref = multi["code_reward"].astype(np.float64).reshape(-1, group)
    multi_episodes = np.broadcast_to(
        multi["episode"][None, :, None], (generation, windows, horizons)
    ).reshape(-1)
    multi_rank, _, multi_weight = _audit_rank(
        multi_raw,
        multi_rc,
        multi_ref,
        multi_episodes,
        args.bootstrap,
        args.seed + 100,
    )

    raw_joint = single["raw_lpips"].astype(np.float64) + single["raw_mse"].astype(np.float64)
    rc_top = np.argmax(single_scores["rc"], axis=1)
    row_index = np.arange(len(raw_joint))
    fidelity = {}
    verdicts = {}
    for offset, name in enumerate(ELIGIBLE + CONTROLS):
        candidate_top = np.argmax(single_scores[name], axis=1)
        joint_delta = raw_joint[row_index, candidate_top] - raw_joint[row_index, rc_top]
        fidelity[name] = {
            "raw_joint_top_delta_vs_rc": _cluster_boot(
                joint_delta, single["episode"], args.bootstrap, args.seed + 200 + offset
            ),
            "same_top_as_rc": float(np.mean(candidate_top == rc_top)),
        }
        single_gain = single_rank[name]["delta_rho_vs_raw"]
        multi_gain = multi_rank[name]["delta_rho_vs_raw"]
        top_delta = fidelity[name]["raw_joint_top_delta_vs_rc"]
        eligible = name in ELIGIBLE
        green = (
            eligible
            and single_gain["ci95"][0] > 0.0
            and multi_gain["ci95"][0] > 0.0
            and single_rank[name]["repair_retention"] >= args.min_repair_retention
            and multi_rank[name]["repair_retention"] >= args.min_repair_retention
            and top_delta["ci95"][1] < 0.0
        )
        verdicts[name] = "GREEN" if green else ("CONTROL" if not eligible else "RED")

    payload = {
        "args": vars(args),
        "single_rank": single_rank,
        "multi_rank": multi_rank,
        "single_weight": single_weight,
        "multi_weight": multi_weight,
        "single_fidelity": fidelity,
        "verdicts": verdicts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== Conservative RC Variant Gate ===")
    for name in ELIGIBLE + CONTROLS:
        sg = single_rank[name]["delta_rho_vs_raw"]
        mg = multi_rank[name]["delta_rho_vs_raw"]
        fd = fidelity[name]["raw_joint_top_delta_vs_rc"]
        print(
            f"[{name}] single dRho={sg['mean']:+.4f} CI={sg['ci95']} "
            f"ret={single_rank[name]['repair_retention']:.2f} | "
            f"multi dRho={mg['mean']:+.4f} CI={mg['ci95']} "
            f"ret={multi_rank[name]['repair_retention']:.2f} | "
            f"rawTop-vs-RC={fd['mean']:+.6f} CI={fd['ci95']} "
            f"=> {verdicts[name]}",
            flush=True,
        )
    print(f"saved {args.out}\nRC_SAFE_VARIANTS_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
