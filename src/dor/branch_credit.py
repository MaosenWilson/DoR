"""Candidate-specific branch-value diagnostics for temporal credit assignment."""

from __future__ import annotations

import numpy as np

from dor.rank_stats import rowwise_spearman


def branch_td_credit(immediate, value_after, value_before, gamma=0.95):
    """Return a block credit from immediate utility and a branch-value residual."""
    immediate = np.asarray(immediate, dtype=np.float64)
    value_after = np.asarray(value_after, dtype=np.float64)
    value_before = np.asarray(value_before, dtype=np.float64)
    if immediate.shape != value_after.shape or immediate.shape != value_before.shape:
        raise ValueError("immediate and branch values must have matching shapes")
    if not np.all(np.isfinite(immediate)) or not np.all(np.isfinite(value_after)):
        raise ValueError("branch credit inputs must be finite")
    if not np.all(np.isfinite(value_before)):
        raise ValueError("branch credit inputs must be finite")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0,1]")
    return immediate + float(gamma) * value_after - value_before


def grouped_matrix(values, contexts):
    """Pack aligned candidate rows into a [context, candidate] matrix."""
    values = np.asarray(values, dtype=np.float64)
    contexts = np.asarray(contexts).astype(str)
    if values.ndim != 1 or contexts.shape != values.shape:
        raise ValueError("values and contexts must be aligned vectors")
    ordered = []
    seen = set()
    for context in contexts:
        if context not in seen:
            ordered.append(context)
            seen.add(context)
    counts = [int(np.count_nonzero(contexts == context)) for context in ordered]
    if not counts or min(counts) < 2 or len(set(counts)) != 1:
        raise ValueError("every context must contain the same candidate group of size >=2")
    return np.stack([values[contexts == context] for context in ordered]), np.asarray(ordered)


def heldout_branch_rows(
    immediate,
    future_after,
    future_before,
    contexts,
    *,
    gamma=0.95,
    seed=2027,
):
    """Cross-fit branch credit across continuation draws.

    Each continuation draw is held out in turn. The remaining draws construct
    candidate-specific credit; the held-out draw evaluates ranking and top-choice
    utility. A within-context permutation of the before-prefix value is the
    identity-breaking control.
    """
    immediate = np.asarray(immediate, dtype=np.float64)
    future_after = np.asarray(future_after, dtype=np.float64)
    future_before = np.asarray(future_before, dtype=np.float64)
    contexts = np.asarray(contexts).astype(str)
    if future_after.shape != future_before.shape or future_after.ndim != 2:
        raise ValueError("future branch arrays must share shape [candidate,draw]")
    if immediate.shape != (future_after.shape[0],) or contexts.shape != immediate.shape:
        raise ValueError("immediate, contexts, and branch candidates must align")
    if future_after.shape[1] < 4:
        raise ValueError("branch gate requires at least four continuation draws")

    immediate_group, ordered_contexts = grouped_matrix(immediate, contexts)
    rng = np.random.default_rng(seed)
    rows = {
        "split_rho": [],
        "immediate_rho": [],
        "delta_rho": [],
        "selection_gain": [],
        "aligned_minus_shuffled": [],
        "context": [],
    }
    for held_out in range(future_after.shape[1]):
        train_draws = [index for index in range(future_after.shape[1]) if index != held_out]
        after_fit, _ = grouped_matrix(future_after[:, train_draws].mean(axis=1), contexts)
        before_fit, _ = grouped_matrix(future_before[:, train_draws].mean(axis=1), contexts)
        after_test, _ = grouped_matrix(future_after[:, held_out], contexts)
        before_test, _ = grouped_matrix(future_before[:, held_out], contexts)
        fitted = branch_td_credit(immediate_group, after_fit, before_fit, gamma)
        target = branch_td_credit(immediate_group, after_test, before_test, gamma)

        shuffled_before = before_fit.copy()
        for row in range(len(shuffled_before)):
            shuffled_before[row] = shuffled_before[row, rng.permutation(shuffled_before.shape[1])]
        shuffled = branch_td_credit(immediate_group, after_fit, shuffled_before, gamma)

        split_rho = rowwise_spearman(fitted, target)
        immediate_rho = rowwise_spearman(immediate_group, target)
        fit_top = np.argmax(fitted, axis=1)
        immediate_top = np.argmax(immediate_group, axis=1)
        shuffled_top = np.argmax(shuffled, axis=1)
        row_index = np.arange(len(target))
        selection_gain = target[row_index, fit_top] - target[row_index, immediate_top]
        aligned_control = target[row_index, fit_top] - target[row_index, shuffled_top]

        rows["split_rho"].extend(split_rho.tolist())
        rows["immediate_rho"].extend(immediate_rho.tolist())
        rows["delta_rho"].extend((split_rho - immediate_rho).tolist())
        rows["selection_gain"].extend(selection_gain.tolist())
        rows["aligned_minus_shuffled"].extend(aligned_control.tolist())
        rows["context"].extend(ordered_contexts.tolist())
    return {name: np.asarray(value) for name, value in rows.items()}


def cluster_bootstrap(values, clusters, rounds=2000, seed=2027):
    """Bootstrap cluster means without treating branch draws as independent."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    if values.ndim != 1 or clusters.shape != values.shape:
        raise ValueError("values and clusters must be aligned vectors")
    unique = np.unique(clusters)
    means = np.asarray([
        np.mean(values[clusters == cluster]) for cluster in unique
    ], dtype=np.float64)
    if len(means) < 2 or not np.all(np.isfinite(means)):
        raise ValueError("cluster bootstrap requires at least two finite clusters")
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(int(rounds), len(means)))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
        "clusters": int(len(means)),
    }
