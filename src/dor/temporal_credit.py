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


def equalize_reward_scale_by_horizon(rewards, epsilon=1e-6):
    """Standardize each frame's candidate-group scale before temporal pooling.

    This leaves each horizon's within-group ordering unchanged.  It only prevents
    a high-variance future frame from dominating every earlier reward-to-go.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [group,horizon], got {rewards.shape}")
    return normalize_by_horizon(rewards, epsilon=epsilon)


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


def scale_equalized_temporal_return_advantages(
    rewards,
    gamma,
    *,
    max_terms=0,
    epsilon=1e-6,
):
    """Build temporal-return advantages after per-horizon scale equalization."""
    source = equalize_reward_scale_by_horizon(rewards, epsilon=epsilon)
    return normalize_by_horizon(
        discounted_returns(source, gamma, max_terms=max_terms),
        epsilon=epsilon,
    )


def reachability_consistent_temporal_scores(raw_rewards, rc_rewards, gamma, epsilon=1e-6):
    """Aggregate only temporal-return preferences supported by raw and RC views.

    Both inputs are ``[group, horizon]``. Conflicting candidate pairs abstain;
    concordant pairs contribute their smaller standardized margin. Returned
    scores are antisymmetric pairwise net support before GRPO normalization.
    """
    raw_rewards = np.asarray(raw_rewards, dtype=np.float64)
    rc_rewards = np.asarray(rc_rewards, dtype=np.float64)
    if raw_rewards.shape != rc_rewards.shape or raw_rewards.ndim != 2:
        raise ValueError(
            f"raw/RC rewards must share [group,horizon], got "
            f"{raw_rewards.shape} and {rc_rewards.shape}"
        )
    group, horizon = raw_rewards.shape
    if group < 2:
        raise ValueError("reachability-consistent ranking requires at least two candidates")
    raw_return = normalize_by_horizon(
        discounted_returns(raw_rewards, gamma), epsilon=epsilon
    )
    rc_return = normalize_by_horizon(
        discounted_returns(rc_rewards, gamma), epsilon=epsilon
    )
    scores = np.zeros((group, horizon), dtype=np.float64)
    coverage = np.zeros(horizon, dtype=np.float64)
    upper = np.triu(np.ones((group, group), dtype=bool), k=1)
    for step in range(horizon):
        raw_gap = raw_return[:, step, None] - raw_return[None, :, step]
        rc_gap = rc_return[:, step, None] - rc_return[None, :, step]
        concordant = raw_gap * rc_gap > 0.0
        edge = (
            np.sign(raw_gap)
            * np.minimum(np.abs(raw_gap), np.abs(rc_gap))
            * concordant
        )
        scores[:, step] = edge.sum(axis=1) / (group - 1)
        coverage[step] = float(concordant[upper].mean())
    return scores, coverage


def reachability_consistent_temporal_advantages(raw_rewards, rc_rewards, gamma, epsilon=1e-6):
    """Return blockwise GRPO advantages from concordant raw/RC temporal pairs."""
    scores, _ = reachability_consistent_temporal_scores(
        raw_rewards, rc_rewards, gamma, epsilon=epsilon
    )
    return normalize_by_horizon(scores, epsilon=epsilon)
