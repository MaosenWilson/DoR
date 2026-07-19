"""Frozen branch-rollout diagnostics for adaptive temporal credit."""

from __future__ import annotations

import numpy as np


def _validate(immediate, future, groups):
    immediate = np.asarray(immediate, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if immediate.ndim != 1 or future.ndim != 2:
        raise ValueError("immediate must be [prefix] and future [prefix,continuation]")
    if len(immediate) != future.shape[0] or len(groups) != len(immediate):
        raise ValueError("immediate, future, and groups must share the prefix axis")
    if future.shape[1] < 2:
        raise ValueError("at least two continuations per prefix are required")
    if not np.all(np.isfinite(immediate)) or not np.all(np.isfinite(future)):
        raise ValueError("delayed-influence inputs must be finite")
    return immediate, future, groups


def grouped_cross_fitted_linear_prediction(
    immediate,
    target,
    groups,
    *,
    folds=5,
    ridge=1e-3,
    seed=2027,
):
    """Predict prefix targets from immediate reward without group leakage."""
    immediate = np.asarray(immediate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if immediate.ndim != 1 or target.shape != immediate.shape or groups.shape != immediate.shape:
        raise ValueError("immediate, target, and groups must be aligned vectors")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("grouped cross-fitting requires at least two groups")
    folds = min(max(int(folds), 2), len(unique))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    group_folds = np.array_split(shuffled, folds)
    prediction = np.empty_like(target)
    for held_out in group_folds:
        test = np.isin(groups, held_out)
        train = ~test
        x_train = np.column_stack([np.ones(train.sum()), immediate[train]])
        penalty = np.diag([0.0, float(ridge)])
        gram = x_train.T @ x_train + penalty
        coef = np.linalg.solve(gram, x_train.T @ target[train])
        prediction[test] = coef[0] + coef[1] * immediate[test]
    return prediction


def delayed_influence_ratio(
    immediate,
    future,
    groups,
    *,
    folds=5,
    ridge=1e-3,
    seed=2027,
    epsilon=1e-12,
):
    """Estimate reliable residual future variation attributable to a prefix.

    The immediate reward predicts each prefix's mean future utility under
    grouped cross-fitting.  A shrinkage-corrected variance ratio then separates
    stable between-prefix residual variation from continuation noise.
    """
    immediate, future, groups = _validate(immediate, future, groups)
    target = future.mean(axis=1)
    prediction = grouped_cross_fitted_linear_prediction(
        immediate, target, groups, folds=folds, ridge=ridge, seed=seed
    )
    residual = future - prediction[:, None]
    prefix_mean = residual.mean(axis=1)
    within = float(np.mean(np.var(residual, axis=1, ddof=1)))
    between_observed = float(np.var(prefix_mean, ddof=1))
    corrected_between = max(0.0, between_observed - within / future.shape[1])
    coefficient = corrected_between / (corrected_between + within + epsilon)
    baseline_error = float(np.sum((target - target.mean()) ** 2))
    prediction_error = float(np.sum((target - prediction) ** 2))
    oof_r2 = 1.0 - prediction_error / max(baseline_error, epsilon)
    return {
        "coefficient": float(np.clip(coefficient, 0.0, 1.0)),
        "between_variance": corrected_between,
        "within_variance": within,
        "observed_between_variance": between_observed,
        "oof_immediate_r2": float(oof_r2),
        "residual": residual,
    }


def shuffle_continuations_within_group(future, groups, *, seed=2027):
    """Break prefix-continuation identity while preserving each group marginal."""
    future = np.asarray(future, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if future.ndim != 2 or groups.shape != (future.shape[0],):
        raise ValueError("future and groups must be aligned")
    rng = np.random.default_rng(seed)
    shuffled = future.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        for continuation in range(future.shape[1]):
            shuffled[indices, continuation] = future[
                rng.permutation(indices), continuation
            ]
    return shuffled
