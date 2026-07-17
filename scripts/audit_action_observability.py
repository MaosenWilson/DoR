"""Pre-registered RCAV Gate A-v2 action-observability audit."""
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
    effect_indices,
    episode_bootstrap,
    evaluate_prediction,
    pooled_effect_features,
    r2_per_dim,
    select_alpha,
    split_masks,
)
from dor.constants import ROOT


PROTOCOLS = ("random", "blocked", "episode")


def csv_ints(value):
    return tuple(int(x) for x in value.split(",") if x)


def csv_strings(value):
    return tuple(x.strip() for x in value.split(",") if x.strip())


def pool_sizes(value):
    result = []
    for item in csv_strings(value):
        h, w = item.lower().split("x")
        result.append((int(h), int(w)))
    return tuple(result)


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


def config_id(row):
    return (
        f"h{row['horizon']}_{row['pool']}_{row['target']}_"
        f"off{row['offset']:+d}_{row['protocol']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{ROOT}/outputs/rcav/action_observability_codes.npz")
    ap.add_argument("--source", choices=("fsq", "motion"), default="fsq")
    ap.add_argument("--horizons", default="1,2,3,4")
    ap.add_argument("--offsets", default="-1,0,1")
    ap.add_argument("--targets", default="first,mean")
    ap.add_argument("--pools", default="4x5,8x10")
    ap.add_argument("--split_seed", type=int, default=2027)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--permutation_seed", type=int, default=7301)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/action_observability_gate_a2.json")
    args = ap.parse_args()

    horizons = csv_ints(args.horizons)
    offsets = csv_ints(args.offsets)
    targets = csv_strings(args.targets)
    pools = pool_sizes(args.pools)
    if not horizons or not offsets or not targets or not pools:
        raise ValueError("the audit grid cannot be empty")
    if any(x not in (-1, 0, 1) for x in offsets):
        raise ValueError("offsets are pre-registered as -1,0,+1")
    if any(x not in ("first", "mean") for x in targets):
        raise ValueError("targets are pre-registered as first,mean")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    raw = np.load(args.cache, allow_pickle=False)
    actions = raw["actions"].astype(np.float64)
    frame_episode = raw["episode_id"].astype(int)
    frame_step = raw["step"].astype(int)
    episode_names = raw["episode_names"].astype(str)
    if not (len(actions) == len(frame_episode) == len(frame_step)):
        raise ValueError("cache arrays have inconsistent lengths")
    codes = raw["codes"] if args.source == "fsq" else None
    if codes is not None and len(codes) != len(actions):
        raise ValueError("FSQ code count does not match frame metadata")

    grid = []
    feature_cache = {}
    total_designs = len(horizons) * len(pools) * len(PROTOCOLS)
    done = 0
    t0 = time.time()
    print(
        f"[setup] source={args.source} frames={len(actions)} episodes={len(episode_names)} "
        f"designs={total_designs} configs={total_designs * len(offsets) * len(targets)} "
        f"device={args.device}",
        flush=True,
    )

    for horizon in horizons:
        starts, ends, row_episode, row_step = effect_indices(
            frame_episode, frame_step, horizon
        )
        for pool in pools:
            pool_name = f"{pool[0]}x{pool[1]}"
            if args.source == "fsq":
                x = pooled_effect_features(codes, starts, ends, pool)
                state_dim = x.shape[1] // 3
            else:
                prefix = f"h{horizon}_{pool_name}"
                x = raw[f"features_{prefix}"].astype(np.float64)
                saved_starts = raw[f"starts_h{horizon}"].astype(int)
                if not np.array_equal(saved_starts, starts):
                    raise ValueError(f"motion feature rows do not match h={horizon} metadata")
                state_dim = int(raw[f"state_dim_{prefix}"].item())
                if not (0 < state_dim < x.shape[1]):
                    raise ValueError(f"invalid state-only width for {prefix}")
            feature_cache[(horizon, pool_name)] = {
                "x": x,
                "starts": starts,
                "episodes": row_episode,
                "steps": row_step,
                "state_dim": state_dim,
            }
            for protocol in PROTOCOLS:
                masks = split_masks(
                    protocol, row_episode, row_step, horizon, episode_names, args.split_seed
                )
                if any(mask.sum() < 2 for mask in masks.values()):
                    raise RuntimeError(
                        f"insufficient rows for {protocol}, h={horizon}: "
                        + str({k: int(v.sum()) for k, v in masks.items()})
                    )
                design = RidgeDesign.from_array(x[masks["train"]], args.device)
                for offset in offsets:
                    for target in targets:
                        y = action_targets(
                            actions, starts, frame_episode, horizon, offset, target
                        )
                        best, alpha_path = select_alpha(
                            design,
                            y[masks["train"]],
                            x[masks["calibration"]],
                            y[masks["calibration"]],
                            ALPHA_GRID,
                        )
                        cal_pred = design.fit_predict(
                            y[masks["train"]], x[masks["calibration"]], best["alpha"]
                        )
                        cal = evaluate_prediction(
                            y[masks["train"]],
                            y[masks["calibration"]],
                            cal_pred,
                            row_episode[masks["calibration"]],
                            n_boot=min(200, args.bootstrap),
                            seed=args.split_seed,
                        )
                        grid.append({
                            "protocol": protocol,
                            "horizon": int(horizon),
                            "pool": pool_name,
                            "offset": int(offset),
                            "target": target,
                            "split_counts": {k: int(v.sum()) for k, v in masks.items()},
                            "selected_alpha": float(best["alpha"]),
                            "alpha_path": alpha_path,
                            "calibration": cal,
                        })
                del design
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total_designs - done)
                filled = 20 * done // total_designs
                local = [r for r in grid if r["protocol"] == protocol and
                         r["horizon"] == horizon and r["pool"] == pool_name]
                winner = max(local, key=lambda row: row["calibration"]["mean_r2"])
                print(
                    f"[audit {'#' * filled:<20}] {done}/{total_designs} "
                    f"{protocol} h={horizon} pool={pool_name} "
                    f"best-cal-R2={winner['calibration']['mean_r2']:+.3f} "
                    f"elapsed={hms(elapsed)} eta={hms(eta)}",
                    flush=True,
                )

    def evaluate_config(row):
        cached = feature_cache[(row["horizon"], row["pool"])]
        x = cached["x"]
        starts = cached["starts"]
        episodes = cached["episodes"]
        steps = cached["steps"]
        masks = split_masks(
            row["protocol"], episodes, steps, row["horizon"], episode_names, args.split_seed
        )
        y = action_targets(
            actions, starts, frame_episode, row["horizon"], row["offset"], row["target"]
        )
        fit = masks["train"] | masks["calibration"]
        design = RidgeDesign.from_array(x[fit], args.device)
        pred = design.fit_predict(y[fit], x[masks["test"]], row["selected_alpha"])
        metrics = evaluate_prediction(
            y[fit], y[masks["test"]], pred, episodes[masks["test"]],
            n_boot=args.bootstrap, seed=args.split_seed + 1,
        )
        return {
            "config": {k: row[k] for k in
                       ("protocol", "horizon", "pool", "offset", "target", "selected_alpha")},
            "config_id": config_id(row),
            "calibration_mean_r2": row["calibration"]["mean_r2"],
            "test": metrics,
        }, (design, x, y, episodes, steps, masks, fit, cached["state_dim"])

    selected = {}
    diagnostic_by_pool = {}
    formal_state = None
    for protocol in PROTOCOLS:
        candidates = [r for r in grid if r["protocol"] == protocol]
        winner = max(candidates, key=lambda row: (row["calibration"]["mean_r2"],
                                                  -row["selected_alpha"]))
        result, state = evaluate_config(winner)
        selected[protocol] = result
        if protocol == "episode":
            formal_state = state
        for pool in (f"{h}x{w}" for h, w in pools):
            pool_rows = [r for r in candidates if r["pool"] == pool]
            pool_winner = max(pool_rows, key=lambda row: row["calibration"]["mean_r2"])
            pool_result, _ = evaluate_config(pool_winner)
            diagnostic_by_pool[f"{protocol}:{pool}"] = pool_result

    formal = selected["episode"]
    design, x, y, episodes, steps, masks, fit, state_dim = formal_state
    state_x = x[:, :state_dim]
    state_train_design = RidgeDesign.from_array(state_x[masks["train"]], args.device)
    state_best, state_alpha_path = select_alpha(
        state_train_design,
        y[masks["train"]],
        state_x[masks["calibration"]],
        y[masks["calibration"]],
        ALPHA_GRID,
    )
    state_design = RidgeDesign.from_array(state_x[fit], args.device)
    state_pred = state_design.fit_predict(
        y[fit], state_x[masks["test"]], state_best["alpha"]
    )
    full_pred = design.fit_predict(
        y[fit], x[masks["test"]], formal["config"]["selected_alpha"]
    )
    target_scale = np.maximum(y[fit].std(0), 1e-6)
    full_error = np.mean(((full_pred - y[masks["test"]]) / target_scale) ** 2, axis=1)
    state_error = np.mean(((state_pred - y[masks["test"]]) / target_scale) ** 2, axis=1)
    transition_gain = episode_bootstrap(
        state_error - full_error,
        episodes[masks["test"]],
        n_boot=args.bootstrap,
        seed=args.split_seed + 2,
    )
    formal["state_only_control"] = {
        "selected_alpha": float(state_best["alpha"]),
        "alpha_path": state_alpha_path,
        "test": evaluate_prediction(
            y[fit], y[masks["test"]], state_pred, episodes[masks["test"]],
            n_boot=args.bootstrap, seed=args.split_seed + 2,
        ),
    }
    formal["transition_gain_over_state_only"] = transition_gain
    rng = np.random.default_rng(args.permutation_seed)
    fit_episodes = episodes[fit]
    fit_steps = steps[fit]
    test_y = y[masks["test"]]
    null = np.empty(args.permutations, dtype=np.float64)
    y_fit = y[fit]
    for pi in range(args.permutations):
        permuted = y_fit.copy()
        for episode in np.unique(fit_episodes):
            loc = np.flatnonzero(fit_episodes == episode)
            loc = loc[np.argsort(fit_steps[loc])]
            shift = int(rng.integers(1, len(loc))) if len(loc) > 1 else 0
            permuted[loc] = np.roll(y_fit[loc], shift, axis=0)
        pred_null = design.fit_predict(
            permuted, x[masks["test"]], formal["config"]["selected_alpha"]
        )
        null[pi] = float(np.mean(r2_per_dim(test_y, pred_null)))
        if args.permutations >= 20 and (pi + 1) % max(1, args.permutations // 10) == 0:
            print(f"[null] {pi + 1}/{args.permutations}", flush=True)
    observed = formal["test"]["mean_r2"]
    permutation_p = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    formal["permutation"] = {
        "n": int(args.permutations),
        "unit": "within_episode_circular_shift_fit_labels",
        "p": permutation_p,
        "q95": float(np.quantile(null, 0.95)),
    }
    retrieval_q05 = formal["test"]["retrieval"]["q05"]
    green = bool(
        observed > 0
        and permutation_p < 0.05
        and retrieval_q05 is not None
        and retrieval_q05 > 0.5
        and transition_gain["q05"] is not None
        and transition_gain["q05"] > 0
    )
    random_r2 = selected["random"]["test"]["mean_r2"]
    blocked_r2 = selected["blocked"]["test"]["mean_r2"]
    if green:
        diagnosis = "EPISODE_GENERALIZABLE_ACTION_SIGNAL"
    elif (observed > 0 and permutation_p < 0.05
          and transition_gain["q05"] is not None and transition_gain["q05"] > 0):
        diagnosis = "EPISODE_SIGNAL_RETRIEVAL_CI_NEAR_MISS"
    elif random_r2 > 0 and blocked_r2 <= 0:
        diagnosis = "TRANSITION_RANDOM_SHORTCUT_ONLY"
    elif blocked_r2 > 0 and observed <= 0:
        diagnosis = "CROSS_EPISODE_DOMAIN_SHIFT"
    elif observed > 0:
        diagnosis = "WEAK_OR_NON_RETRIEVABLE_EPISODE_SIGNAL"
    else:
        diagnosis = "FSQ_SIGNAL_NOT_ESTABLISHED_RUN_MOTION_ORACLE"

    report = {
        "protocol": {
            "horizons": list(horizons),
            "offsets": list(offsets),
            "targets": list(targets),
            "pools": [f"{h}x{w}" for h, w in pools],
            "alpha_grid": list(ALPHA_GRID),
            "split_seed": args.split_seed,
            "selection_metric": "calibration_mean_r2",
            "formal_split": "episode_disjoint",
            "feature_source": args.source,
        },
        "n_frames": int(len(actions)),
        "n_episodes": int(len(episode_names)),
        "calibration_grid": grid,
        "selected_test_once": selected,
        "diagnostic_pool_winners": diagnostic_by_pool,
        "formal_gate": {
            "green": green,
            "criteria": {
                "test_mean_r2_gt": 0.0,
                "permutation_p_lt": 0.05,
                "retrieval_bootstrap_q05_gt": 0.5,
                "transition_gain_over_state_bootstrap_q05_gt": 0.0,
            },
            "diagnosis": diagnosis,
        },
    }
    report = clean_json(report)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("\n=== RCAV Gate A-v2: Action Observability ===")
    for protocol in PROTOCOLS:
        row = selected[protocol]
        test = row["test"]
        print(
            f"{protocol:8s} {row['config_id']:40s} "
            f"R2={test['mean_r2']:+.3f} pos={test['positive_dims']}/6 "
            f"dirBA={test['direction_balanced_accuracy']:.3f} "
            f"retrieval={test['retrieval']['mean']:.3f} "
            f"CI90=[{test['retrieval']['q05']:.3f},{test['retrieval']['q95']:.3f}]"
        )
    print(f"episode permutation p={permutation_p:.5f} null_q95={np.quantile(null, 0.95):+.3f}")
    print(
        f"transition>state gain={transition_gain['mean']:+.4f} "
        f"CI90=[{transition_gain['q05']:+.4f},{transition_gain['q95']:+.4f}]"
    )
    print(f"[diagnosis] {diagnosis}")
    print(f"[verdict] {'GREEN' if green else 'RED'}")
    print(f"saved {out}\nACTION_OBSERVABILITY_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
