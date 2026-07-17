"""Small, dependency-free rank diagnostics shared by external verifier gates."""
from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks along the last axis, assigning tied values one rank."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    flat = values.reshape(-1, values.shape[-1])
    result = np.empty_like(flat)
    for row_id, row in enumerate(flat):
        order = np.argsort(row, kind="mergesort")
        sorted_row = row[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_row)) + 1]
        ends = np.r_[starts[1:], len(row)]
        ranks = np.empty_like(row)
        for begin, end in zip(starts, ends):
            ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        result[row_id] = ranks
    return result.reshape(values.shape)


def rowwise_pearson(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Pearson correlation for every row, returning zero for a constant row."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("left and right must share a non-scalar shape")
    a = left.reshape(-1, left.shape[-1])
    b = right.reshape(-1, right.shape[-1])
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    value = np.zeros(len(a), dtype=np.float64)
    valid = denominator > eps
    value[valid] = (a[valid] * b[valid]).sum(axis=1) / denominator[valid]
    return value.reshape(left.shape[:-1])


def rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Tie-aware Spearman correlation for every row."""
    return rowwise_pearson(average_ranks(left), average_ranks(right))


def pair_flip_fraction(score: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fraction of non-tied candidate pairs whose preference disagrees with reference."""
    score = np.asarray(score, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if score.shape != reference.shape or score.ndim < 1:
        raise ValueError("score and reference must share a non-scalar shape")
    a = score.reshape(-1, score.shape[-1])
    b = reference.reshape(-1, reference.shape[-1])
    upper = np.triu(np.ones(a.shape[1], dtype=bool), k=1)
    result = np.full(len(a), np.nan, dtype=np.float64)
    for index, (left, right) in enumerate(zip(a, b)):
        delta_left = left[:, None] - left[None, :]
        delta_right = right[:, None] - right[None, :]
        valid = upper & (delta_left != 0.0) & (delta_right != 0.0)
        if valid.any():
            result[index] = float(np.mean((delta_left[valid] * delta_right[valid]) < 0.0))
    return result.reshape(score.shape[:-1])
