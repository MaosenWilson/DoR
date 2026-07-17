"""Low-capacity latent inverse verifier for RCAV Gate A.

The verifier is deliberately linear and episode-disjoint.  Gate A asks whether
frozen FSQ transitions contain recoverable action information; it is not allowed
to hide a large reward model behind the world-model result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dor.constants import MOTION_DIMS

# Mode/base dimensions are constant or binary in the 20-episode processed subset
# and are excluded from Gate A.
ARM_MOTION_DIMS = tuple(MOTION_DIMS)
GRIPPER_DIM = 3
DEFAULT_ALPHA_GRID = tuple(10.0**p for p in range(-4, 3))


def transition_features(z_t, z_tp1, pool_hw=(4, 5)):
    """Return fixed pooled state/delta/absolute-delta features.

    Args:
        z_t, z_tp1: post-quant FSQ codes ``[N,C,H,W]`` or ``[C,H,W]``.
        pool_hw: fixed spatial output size.
    """
    if z_t.ndim == 3:
        z_t = z_t.unsqueeze(0)
    if z_tp1.ndim == 3:
        z_tp1 = z_tp1.unsqueeze(0)
    if z_t.shape != z_tp1.shape or z_t.ndim != 4:
        raise ValueError(f"code tensors must share [N,C,H,W], got {z_t.shape}/{z_tp1.shape}")
    delta = z_tp1.float() - z_t.float()
    parts = (z_t.float(), delta, delta.abs())
    return torch.cat(
        [F.adaptive_avg_pool2d(x, pool_hw).flatten(1) for x in parts], dim=1
    )


def split_episodes(episode_names, seed=2027):
    """Deterministic 60/20/20 episode split with no transition leakage."""
    names = np.asarray(sorted(set(str(x) for x in episode_names)), dtype=str)
    if len(names) < 5:
        raise ValueError("Gate A requires at least five episodes for grouped splitting")
    rng = np.random.default_rng(seed)
    names = names[rng.permutation(len(names))]
    n_train = max(1, int(np.floor(0.60 * len(names))))
    n_cal = max(1, int(np.floor(0.20 * len(names))))
    if n_train + n_cal >= len(names):
        n_cal = 1
        n_train = len(names) - 2
    split = {
        "train": np.sort(names[:n_train]),
        "calibration": np.sort(names[n_train:n_train + n_cal]),
        "test": np.sort(names[n_train + n_cal:]),
    }
    if set(split["train"]) & set(split["calibration"]):
        raise AssertionError("train/calibration episode leakage")
    if (set(split["train"]) | set(split["calibration"])) & set(split["test"]):
        raise AssertionError("test episode leakage")
    return split


def _standardizer(x, eps=1e-6):
    mean = np.asarray(x, np.float64).mean(0)
    scale = np.asarray(x, np.float64).std(0)
    return mean, np.maximum(scale, eps)


def _ridge_coef(x, y, alpha):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def _r2(y, pred, eps=1e-12):
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def _nrmse(y, pred, reference_scale):
    err = (np.asarray(y, np.float64) - np.asarray(pred, np.float64)) / reference_scale
    return np.sqrt(np.mean(err**2, axis=0))


def predict(payload, x):
    """Predict raw action values from cached transition features."""
    x = np.asarray(x, np.float64)
    xn = (x - payload["x_mean"]) / payload["x_scale"]
    yn = xn @ payload["coef"]
    return yn * payload["y_scale"] + payload["y_mean"]


def action_score(payload, x, action):
    """Per-transition frozen inverse score; higher means more action-consistent."""
    pred = predict(payload, x)
    target = np.asarray(action, np.float64)
    if target.ndim == 1:
        target = target[None]
    target = target[:, np.asarray(payload["action_dims"], dtype=int)]
    residual_scale = np.maximum(payload["residual_scale"], 1e-6)
    return -np.mean(((pred - target) / residual_scale) ** 2, axis=1)


def _fit_with_stats(x, y, alpha, x_mean, x_scale, y_mean, y_scale):
    xn = (x - x_mean) / x_scale
    yn = (y - y_mean) / y_scale
    return _ridge_coef(xn, yn, alpha)


def fit_grouped_ridge(x, actions, episode_names, *, action_dims=ARM_MOTION_DIMS,
                      split_seed=2027, alpha_grid=DEFAULT_ALPHA_GRID,
                      permutations=200, permutation_seed=7301):
    """Fit and independently gate a grouped ridge latent inverse verifier."""
    x = np.asarray(x, np.float64)
    actions = np.asarray(actions, np.float64)
    episodes = np.asarray(episode_names, dtype=str)
    dims = np.asarray(action_dims, dtype=int)
    y = actions[:, dims]
    if not (len(x) == len(y) == len(episodes)):
        raise ValueError("features, actions and episodes must have the same row count")

    split = split_episodes(episodes, split_seed)
    masks = {name: np.isin(episodes, values) for name, values in split.items()}
    if any(mask.sum() == 0 for mask in masks.values()):
        raise ValueError("an episode split contains no transitions")

    x_mean, x_scale = _standardizer(x[masks["train"]])
    y_mean, y_scale = _standardizer(y[masks["train"]])
    alpha_rows = []
    for alpha in alpha_grid:
        coef = _fit_with_stats(
            x[masks["train"]], y[masks["train"]], alpha,
            x_mean, x_scale, y_mean, y_scale,
        )
        pred = ((x[masks["calibration"]] - x_mean) / x_scale) @ coef
        pred = pred * y_scale + y_mean
        r2 = _r2(y[masks["calibration"]], pred)
        alpha_rows.append({"alpha": float(alpha), "mean_r2": float(np.mean(r2))})
    best = max(alpha_rows, key=lambda row: (row["mean_r2"], -row["alpha"]))

    fit_mask = masks["train"] | masks["calibration"]
    x_final_mean, x_final_scale = _standardizer(x[fit_mask])
    y_final_mean, y_final_scale = _standardizer(y[fit_mask])
    coef = _fit_with_stats(
        x[fit_mask], y[fit_mask], best["alpha"],
        x_final_mean, x_final_scale, y_final_mean, y_final_scale,
    )
    payload = {
        "x_mean": x_final_mean,
        "x_scale": x_final_scale,
        "y_mean": y_final_mean,
        "y_scale": y_final_scale,
        "coef": coef,
        "action_dims": dims,
        "alpha": float(best["alpha"]),
        "split_seed": int(split_seed),
        "split": split,
    }

    # Calibration residuals come from a model that never fitted calibration rows.
    cal_coef = _fit_with_stats(
        x[masks["train"]], y[masks["train"]], best["alpha"],
        x_mean, x_scale, y_mean, y_scale,
    )
    cal_pred = ((x[masks["calibration"]] - x_mean) / x_scale) @ cal_coef
    cal_pred = cal_pred * y_scale + y_mean
    payload["residual_scale"] = np.maximum(
        np.sqrt(np.mean((cal_pred - y[masks["calibration"]]) ** 2, axis=0)), 1e-6
    )

    test_y = y[masks["test"]]
    test_pred = predict(payload, x[masks["test"]])
    test_r2 = _r2(test_y, test_pred)
    test_nrmse = _nrmse(test_y, test_pred, y_final_scale)
    mean_pred = np.broadcast_to(y_final_mean, test_y.shape)
    mean_nrmse = _nrmse(test_y, mean_pred, y_final_scale)

    # Fixed-design null: circularly misalign labels inside each fit episode.  This
    # preserves per-episode action marginals and temporal autocorrelation while
    # breaking the frame-transition/action correspondence; test is untouched.
    rng = np.random.default_rng(permutation_seed)
    xn_fit = (x[fit_mask] - x_final_mean) / x_final_scale
    xn_test = (x[masks["test"]] - x_final_mean) / x_final_scale
    reg = np.eye(xn_fit.shape[1], dtype=np.float64) * best["alpha"]
    operator = np.linalg.solve(xn_fit.T @ xn_fit + reg, xn_fit.T)
    yn_fit = (y[fit_mask] - y_final_mean) / y_final_scale
    fit_episodes = episodes[fit_mask]
    null = np.empty(int(permutations), dtype=np.float64)
    for pi in range(int(permutations)):
        yn_null = np.empty_like(yn_fit)
        for episode in np.unique(fit_episodes):
            loc = np.flatnonzero(fit_episodes == episode)
            shift = int(rng.integers(1, len(loc))) if len(loc) > 1 else 0
            yn_null[loc] = np.roll(yn_fit[loc], shift, axis=0)
        coef_null = operator @ yn_null
        pred_null = (xn_test @ coef_null) * y_final_scale + y_final_mean
        null[pi] = float(np.mean(_r2(test_y, pred_null)))
    observed = float(np.mean(test_r2))
    p_perm = float((1 + np.sum(null >= observed)) / (len(null) + 1))

    report = {
        "n_transitions": int(len(x)),
        "n_features": int(x.shape[1]),
        "action_dims": dims.tolist(),
        "split": {k: v.tolist() for k, v in split.items()},
        "split_counts": {k: int(mask.sum()) for k, mask in masks.items()},
        "alpha_search": alpha_rows,
        "selected_alpha": float(best["alpha"]),
        "test_r2_per_dim": test_r2.tolist(),
        "test_mean_r2": observed,
        "test_positive_dims": int(np.sum(test_r2 > 0)),
        "test_nrmse_per_dim": test_nrmse.tolist(),
        "test_mean_nrmse": float(np.mean(test_nrmse)),
        "mean_baseline_nrmse": float(np.mean(mean_nrmse)),
        "permutations": int(permutations),
        "permutation_unit": "within_episode_circular_shift",
        "permutation_p": p_perm,
        "permutation_q95": float(np.quantile(null, 0.95)),
    }
    report["green"] = bool(
        observed > 0
        and report["test_positive_dims"] >= 4
        and p_perm < 0.01
        and report["test_mean_nrmse"] < report["mean_baseline_nrmse"]
    )
    return payload, report


def save_payload(path, payload, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "alpha": float(payload["alpha"]),
        "split_seed": int(payload["split_seed"]),
        "split": {k: np.asarray(v, dtype=str).tolist() for k, v in payload["split"].items()},
    }
    if metadata:
        meta.update(metadata)
    np.savez_compressed(
        path,
        x_mean=payload["x_mean"].astype(np.float32),
        x_scale=payload["x_scale"].astype(np.float32),
        y_mean=payload["y_mean"].astype(np.float32),
        y_scale=payload["y_scale"].astype(np.float32),
        coef=payload["coef"].astype(np.float32),
        residual_scale=payload["residual_scale"].astype(np.float32),
        action_dims=np.asarray(payload["action_dims"], dtype=np.int64),
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
    )


def load_payload(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata"].item()))
    return {
        "x_mean": data["x_mean"].astype(np.float64),
        "x_scale": data["x_scale"].astype(np.float64),
        "y_mean": data["y_mean"].astype(np.float64),
        "y_scale": data["y_scale"].astype(np.float64),
        "coef": data["coef"].astype(np.float64),
        "residual_scale": data["residual_scale"].astype(np.float64),
        "action_dims": data["action_dims"].astype(int),
        "alpha": float(meta["alpha"]),
        "split_seed": int(meta["split_seed"]),
        "split": {k: np.asarray(v, dtype=str) for k, v in meta["split"].items()},
        "metadata": meta,
    }
