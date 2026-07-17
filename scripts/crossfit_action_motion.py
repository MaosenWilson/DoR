"""Nested episode cross-fit for command-aligned RAFT action observability."""
import argparse
import json
import os
import time

import numpy as np
import torch

from dor.action_observability import (
    ALPHA_GRID,
    RidgeDesign,
    action_targets,
    direction_balanced_accuracy,
    effect_indices,
    episode_bootstrap,
    episode_folds,
    retrieval_rows,
    r2_per_dim,
    select_alpha,
)
from dor.constants import ROOT


def csv_ints(value):
    return tuple(int(x) for x in value.split(",") if x)


def csv_strings(value):
    return tuple(x.strip() for x in value.split(",") if x.strip())


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{ROOT}/outputs/rcav/action_motion_oracle.npz")
    ap.add_argument("--horizons", default="1,2,3,4")
    ap.add_argument("--pools", default="4x5,8x10")
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=2027)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--permutation_seed", type=int, default=7301)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/action_motion_command_crossfit_a24.json")
    args = ap.parse_args()
    horizons = csv_ints(args.horizons)
    pools = csv_strings(args.pools)
    if not horizons or not pools:
        raise ValueError("horizons and pools cannot be empty")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    raw = np.load(args.cache, allow_pickle=False)
    actions = raw["actions"].astype(np.float64)
    frame_episode = raw["episode_id"].astype(int)
    frame_step = raw["step"].astype(int)
    episode_names = raw["episode_names"].astype(str)
    folds = episode_folds(len(episode_names), args.outer_folds, args.split_seed)

    feature_cache = {}
    for horizon in horizons:
        starts, _, episodes, steps = effect_indices(frame_episode, frame_step, horizon)
        if not np.array_equal(starts, raw[f"starts_h{horizon}"].astype(int)):
            raise ValueError(f"motion cache alignment mismatch at horizon {horizon}")
        y = action_targets(actions, starts, frame_episode, horizon, 0, "mean")
        for pool in pools:
            prefix = f"h{horizon}_{pool}"
            x = raw[f"features_{prefix}"].astype(np.float64)
            state_dim = int(raw[f"state_dim_{prefix}"].item())
            feature_cache[(horizon, pool)] = {
                "x": x,
                "y": y,
                "episodes": episodes,
                "steps": steps,
                "state_dim": state_dim,
            }

    fold_reports = []
    fold_states = []
    oof_y, oof_pred, oof_state_pred = [], [], []
    oof_episode, oof_retrieval, oof_gain = [], [], []
    direction_rows = []
    t0 = time.time()
    print(
        f"[setup] episodes={len(episode_names)} outer_folds={len(folds)} "
        f"configs/fold={len(horizons) * len(pools)} device={args.device}",
        flush=True,
    )
    all_episode_ids = np.arange(len(episode_names), dtype=int)
    for fi, test_ids in enumerate(folds):
        remaining = np.setdiff1d(all_episode_ids, test_ids)
        inner = np.random.default_rng(args.split_seed + 1009 * (fi + 1)).permutation(remaining)
        n_cal = max(1, int(round(0.25 * len(inner))))
        cal_ids = np.sort(inner[:n_cal])
        train_ids = np.sort(inner[n_cal:])
        candidates = []
        for horizon in horizons:
            for pool in pools:
                data = feature_cache[(horizon, pool)]
                train = np.isin(data["episodes"], train_ids)
                cal = np.isin(data["episodes"], cal_ids)
                design = RidgeDesign.from_array(data["x"][train], args.device)
                best, path = select_alpha(
                    design,
                    data["y"][train],
                    data["x"][cal],
                    data["y"][cal],
                    ALPHA_GRID,
                )
                candidates.append({
                    "horizon": int(horizon),
                    "pool": pool,
                    "alpha": float(best["alpha"]),
                    "calibration_mean_r2": float(best["mean_r2"]),
                    "alpha_path": path,
                })
                del design
        winner = max(candidates, key=lambda row: (row["calibration_mean_r2"], -row["alpha"]))
        data = feature_cache[(winner["horizon"], winner["pool"])]
        train = np.isin(data["episodes"], train_ids)
        cal = np.isin(data["episodes"], cal_ids)
        fit = train | cal
        test = np.isin(data["episodes"], test_ids)
        design = RidgeDesign.from_array(data["x"][fit], args.device)
        pred = design.fit_predict(data["y"][fit], data["x"][test], winner["alpha"])

        state_x = data["x"][:, :data["state_dim"]]
        state_train_design = RidgeDesign.from_array(state_x[train], args.device)
        state_best, state_path = select_alpha(
            state_train_design,
            data["y"][train],
            state_x[cal],
            data["y"][cal],
            ALPHA_GRID,
        )
        state_design = RidgeDesign.from_array(state_x[fit], args.device)
        state_pred = state_design.fit_predict(
            data["y"][fit], state_x[test], state_best["alpha"]
        )

        scale = np.maximum(data["y"][fit].std(0), 1e-6)
        full_error = np.mean(((pred - data["y"][test]) / scale) ** 2, axis=1)
        state_error = np.mean(((state_pred - data["y"][test]) / scale) ** 2, axis=1)
        retrieval = retrieval_rows(
            data["y"][fit], data["y"][test], pred, data["episodes"][test]
        )
        direction = direction_balanced_accuracy(data["y"][fit], data["y"][test], pred)
        oof_y.append(data["y"][test])
        oof_pred.append(pred)
        oof_state_pred.append(state_pred)
        oof_episode.append(data["episodes"][test])
        oof_retrieval.append(retrieval)
        oof_gain.append(state_error - full_error)
        direction_rows.append((int(test.sum()), direction))
        fold_reports.append({
            "fold": fi,
            "test_episode_ids": test_ids.tolist(),
            "test_episode_names": episode_names[test_ids].tolist(),
            "train_episode_ids": train_ids.tolist(),
            "calibration_episode_ids": cal_ids.tolist(),
            "selected": winner,
            "state_selected_alpha": float(state_best["alpha"]),
            "state_alpha_path": state_path,
            "test_mean_r2": float(np.mean(r2_per_dim(data["y"][test], pred))),
            "test_retrieval_mean": float(np.nanmean(retrieval)),
            "test_transition_gain": float(np.mean(state_error - full_error)),
        })
        fold_states.append({
            "design": design,
            "x_test": data["x"][test],
            "y_fit": data["y"][fit],
            "y_test": data["y"][test],
            "fit_episodes": data["episodes"][fit],
            "fit_steps": data["steps"][fit],
            "alpha": winner["alpha"],
        })
        elapsed = time.time() - t0
        eta = elapsed / (fi + 1) * (len(folds) - fi - 1)
        print(
            f"[fold {fi + 1}/{len(folds)}] h={winner['horizon']} pool={winner['pool']} "
            f"calR2={winner['calibration_mean_r2']:+.3f} "
            f"testR2={fold_reports[-1]['test_mean_r2']:+.3f} "
            f"retrieval={fold_reports[-1]['test_retrieval_mean']:.3f} "
            f"elapsed={hms(elapsed)} eta={hms(eta)}",
            flush=True,
        )

    y_all = np.concatenate(oof_y)
    pred_all = np.concatenate(oof_pred)
    state_pred_all = np.concatenate(oof_state_pred)
    episode_all = np.concatenate(oof_episode)
    retrieval_all = np.concatenate(oof_retrieval)
    gain_all = np.concatenate(oof_gain)
    r2 = r2_per_dim(y_all, pred_all)
    state_r2 = r2_per_dim(y_all, state_pred_all)
    retrieval_boot = episode_bootstrap(
        retrieval_all, episode_all, args.bootstrap, args.split_seed + 1
    )
    gain_boot = episode_bootstrap(gain_all, episode_all, args.bootstrap, args.split_seed + 2)
    direction = float(
        sum(n * score for n, score in direction_rows) / sum(n for n, _ in direction_rows)
    )

    rng = np.random.default_rng(args.permutation_seed)
    null = np.empty(args.permutations, dtype=np.float64)
    for pi in range(args.permutations):
        null_y, null_pred = [], []
        for state in fold_states:
            permuted = state["y_fit"].copy()
            for episode in np.unique(state["fit_episodes"]):
                loc = np.flatnonzero(state["fit_episodes"] == episode)
                loc = loc[np.argsort(state["fit_steps"][loc])]
                shift = int(rng.integers(1, len(loc))) if len(loc) > 1 else 0
                permuted[loc] = np.roll(state["y_fit"][loc], shift, axis=0)
            pred_null = state["design"].fit_predict(
                permuted, state["x_test"], state["alpha"]
            )
            null_y.append(state["y_test"])
            null_pred.append(pred_null)
        null[pi] = float(np.mean(r2_per_dim(np.concatenate(null_y), np.concatenate(null_pred))))
        if args.permutations >= 20 and (pi + 1) % max(1, args.permutations // 10) == 0:
            print(f"[null] {pi + 1}/{args.permutations}", flush=True)
    observed = float(np.mean(r2))
    p_perm = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    green = bool(
        observed > 0
        and p_perm < 0.05
        and retrieval_boot["q05"] is not None and retrieval_boot["q05"] > 0.5
        and gain_boot["q05"] is not None and gain_boot["q05"] > 0
    )
    report = clean_json({
        "protocol": {
            "outer_folds": args.outer_folds,
            "horizons": list(horizons),
            "pools": list(pools),
            "offset": 0,
            "target": "mean",
            "split_seed": args.split_seed,
            "selection_metric": "inner_calibration_mean_r2",
        },
        "folds": fold_reports,
        "oof": {
            "n_episodes": len(episode_names),
            "n_rows": len(y_all),
            "mean_r2": observed,
            "r2_per_dim": r2.tolist(),
            "positive_dims": int(np.sum(r2 > 0)),
            "state_only_mean_r2": float(np.mean(state_r2)),
            "direction_balanced_accuracy": direction,
            "retrieval": retrieval_boot,
            "transition_gain_over_state": gain_boot,
            "permutation_p": p_perm,
            "permutation_q95": float(np.quantile(null, 0.95)),
        },
        "gate": {
            "green": green,
            "criteria": {
                "oof_mean_r2_gt": 0.0,
                "permutation_p_lt": 0.05,
                "retrieval_q05_gt": 0.5,
                "transition_gain_q05_gt": 0.0,
            },
        },
    })
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("\n=== RCAV Gate A2.4: Nested Episode Cross-Fit ===")
    print(f"OOF R2={observed:+.4f} per_dim=" + ",".join(f"{v:+.3f}" for v in r2))
    print(f"positive_dims={np.sum(r2 > 0)}/6 directionBA={direction:.3f}")
    print(
        f"retrieval={retrieval_boot['mean']:.3f} "
        f"CI90=[{retrieval_boot['q05']:.3f},{retrieval_boot['q95']:.3f}]"
    )
    print(
        f"transition>state gain={gain_boot['mean']:+.4f} "
        f"CI90=[{gain_boot['q05']:+.4f},{gain_boot['q95']:+.4f}]"
    )
    print(f"permutation p={p_perm:.5f} null_q95={np.quantile(null, 0.95):+.4f}")
    print(f"[verdict] {'GREEN' if green else 'RED'}")
    print(f"saved {out}\nACTION_MOTION_CROSSFIT_OK", flush=True)


if __name__ == "__main__":
    main()
