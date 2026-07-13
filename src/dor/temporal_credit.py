"""Temporal credit-assignment utilities and falsification controls."""

from __future__ import annotations

import numpy as np


def discounted_returns(rewards, gamma, max_terms=0):
    """Compute reward-to-go, optionally truncated to ``max_terms`` rewards."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [group,horizon], got {rewards.shape}")
    if max_terms < 0:
        raise ValueError("max_terms must be non-negative")
    group, horizon = rewards.shape
    returns = np.zeros((group, horizon), dtype=np.float64)
    for start in range(horizon):
        stop = horizon if max_terms == 0 else min(horizon, start + max_terms)
        discount = 1.0
        for future in range(start, stop):
            returns[:, start] += discount * rewards[:, future]
            discount *= gamma
    return returns


def normalize_by_horizon(values, epsilon=1e-6):
    """Apply GRPO group normalization independently at every horizon."""
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=0, keepdims=True)) / (
        values.std(axis=0, keepdims=True) + epsilon
    )


def shuffle_candidate_correspondence(rewards, seed):
    """Break temporal identity while preserving each horizon's reward multiset."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [group,horizon], got {rewards.shape}")
    rng = np.random.default_rng(seed)
    shuffled = rewards.copy()
    for horizon in range(rewards.shape[1]):
        shuffled[:, horizon] = rewards[rng.permutation(rewards.shape[0]), horizon]
    return shuffled


def temporal_return_advantages(
    rewards,
    gamma,
    *,
    max_terms=0,
    shuffle_seed=None,
):
    """Build horizon-normalized returns for the aligned or shuffled control."""
    source = np.asarray(rewards, dtype=np.float64)
    if shuffle_seed is not None:
        source = shuffle_candidate_correspondence(source, shuffle_seed)
    return normalize_by_horizon(discounted_returns(source, gamma, max_terms=max_terms))
