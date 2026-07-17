"""Fit five episode-cross-fitted h1 motion-action verifier payloads."""
import argparse
import json
import os

import numpy as np

from dor.action_observability import (
    ALPHA_GRID,
    RidgeDesign,
    action_targets,
    effect_indices,
    episode_folds,
    r2_per_dim,
    select_alpha,
)
from dor.constants import ROOT
from dor.motion_action import save_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{ROOT}/outputs/rcav/action_motion_oracle.npz")
    ap.add_argument("--crossfit", default=f"{ROOT}/outputs/rcav/action_motion_command_crossfit_a24.json")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--pool", default="8x10")
    ap.add_argument("--split_seed", type=int, default=2027)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/motion_action_payload_h1_8x10.npz")
    args = ap.parse_args()

    crossfit = json.load(open(args.crossfit))
    if not crossfit.get("gate", {}).get("green", False):
        raise RuntimeError("refusing to fit payload: A2.4 cross-fit gate is not GREEN")
    raw = np.load(args.cache, allow_pickle=False)
    actions = raw["actions"].astype(np.float64)
    frame_episode = raw["episode_id"].astype(int)
    frame_step = raw["step"].astype(int)
    episode_names = raw["episode_names"].astype(str)
    starts, _, episodes, _ = effect_indices(frame_episode, frame_step, args.horizon)
    if not np.array_equal(starts, raw[f"starts_h{args.horizon}"].astype(int)):
        raise ValueError("motion cache alignment mismatch")
    prefix = f"h{args.horizon}_{args.pool}"
    x = raw[f"features_{prefix}"].astype(np.float64)
    y = action_targets(actions, starts, frame_episode, args.horizon, 0, "mean")
    folds = episode_folds(len(episode_names), 5, args.split_seed)

    models, fold_names, oof_y, oof_pred = [], [], [], []
    all_ids = np.arange(len(episode_names), dtype=int)
    print(f"[setup] fixed h={args.horizon} pool={args.pool} folds={len(folds)}", flush=True)
    for fi, test_ids in enumerate(folds):
        remaining = np.setdiff1d(all_ids, test_ids)
        inner = np.random.default_rng(args.split_seed + 1009 * (fi + 1)).permutation(remaining)
        n_cal = max(1, int(round(0.25 * len(inner))))
        cal_ids, train_ids = inner[:n_cal], inner[n_cal:]
        train = np.isin(episodes, train_ids)
        cal = np.isin(episodes, cal_ids)
        fit = train | cal
        test = np.isin(episodes, test_ids)
        design_train = RidgeDesign.from_array(x[train], args.device)
        best, _ = select_alpha(design_train, y[train], x[cal], y[cal], ALPHA_GRID)
        design = RidgeDesign.from_array(x[fit], args.device)
        model = design.fit_parameters(y[fit], best["alpha"])
        pred = design.fit_predict(y[fit], x[test], best["alpha"])
        models.append(model)
        fold_names.append(episode_names[test_ids].tolist())
        oof_y.append(y[test])
        oof_pred.append(pred)
        print(
            f"[fold {fi + 1}/5] alpha={best['alpha']:.4g} "
            f"testR2={np.mean(r2_per_dim(y[test], pred)):+.4f} "
            f"episodes={','.join(str(x) for x in test_ids)}",
            flush=True,
        )

    y_all = np.concatenate(oof_y)
    pred_all = np.concatenate(oof_pred)
    r2 = r2_per_dim(y_all, pred_all)
    weights = np.maximum(r2, 0)
    if weights.sum() <= 0:
        raise RuntimeError("no observable action dimension remains after cross-fitting")
    weights /= weights.sum()
    residual_scale = np.maximum(np.sqrt(np.mean((pred_all - y_all) ** 2, axis=0)), 1e-6)
    metadata = {
        "horizon": args.horizon,
        "pool": args.pool,
        "offset": 0,
        "target": "mean",
        "split_seed": args.split_seed,
        "source_cache": os.path.abspath(args.cache),
        "source_crossfit": os.path.abspath(args.crossfit),
        "oof_r2_per_dim": r2.tolist(),
        "weight_rule": "normalize(max(oof_r2,0))",
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_payload(out, models, residual_scale, weights, fold_names, metadata)
    print("OOF R2=" + ",".join(f"{v:+.4f}" for v in r2))
    print("weights=" + ",".join(f"{v:.4f}" for v in weights))
    print(f"saved {out}\nMOTION_ACTION_PAYLOAD_OK", flush=True)


if __name__ == "__main__":
    main()
