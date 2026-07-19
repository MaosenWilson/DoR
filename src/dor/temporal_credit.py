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
    epsilon=1e-6,
):
    """Build horizon-normalized returns for the aligned or shuffled control."""
    source = np.asarray(rewards, dtype=np.float64)
    if shuffle_seed is not None:
        source = shuffle_candidate_correspondence(source, shuffle_seed)
    return normalize_by_horizon(
        discounted_returns(source, gamma, max_terms=max_terms),
        epsilon=epsilon,
    )


def gae_frame_advantages(rewards, gamma, lam, *, shuffle_seed=None, epsilon=1e-6):
    """Critic-free GAE frame-block advantages with a per-horizon group baseline.

    The per-horizon group mean is the (critic-free) value baseline, giving TD
    residuals ``delta[i,u] = r[i,u] - mean_j r[j,u]``. The block-``t`` advantage is
    the ``(gamma*lam)``-discounted sum of these residuals, then horizon-normalized:

        A[i,t] = normalize_t( sum_{u>=t} (gamma*lam)^{u-t} delta[i,u] ).

    ``lam`` trades variance (long-horizon reward-to-go noise) against bias:
      * ``lam=1`` reduces exactly to ``temporal_return_advantages`` (the baseline
        subtraction is a per-horizon constant absorbed by normalization);
      * ``lam=0`` reduces to per-horizon frame-only advantages
        (``normalize_by_horizon(rewards)``).
    Interior ``lam`` caps the variance that early blocks accumulate over the tail.
    """
    source = np.asarray(rewards, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"rewards must be [group,horizon], got {source.shape}")
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lam must lie in [0,1]")
    if shuffle_seed is not None:
        source = shuffle_candidate_correspondence(source, shuffle_seed)
    residual = source - source.mean(axis=0, keepdims=True)
    gae = discounted_returns(residual, float(gamma) * float(lam))
    return normalize_by_horizon(gae, epsilon=epsilon)


def delayed_temporal_return_advantages(
    rewards,
    gamma,
    *,
    shuffle_seed=None,
    epsilon=1e-6,
):
    """Assign strictly future rewards to each preceding frame block."""
    source = np.asarray(rewards, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"rewards must be [group,horizon], got {source.shape}")
    if shuffle_seed is not None:
        source = shuffle_candidate_correspondence(source, shuffle_seed)
    group, horizon = source.shape
    advantages = np.zeros((group, horizon), dtype=np.float64)
    if horizon > 1:
        advantages[:, :-1] = normalize_by_horizon(
            discounted_returns(source[:, 1:], gamma), epsilon=epsilon
        )
    return advantages


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


def conservative_adaptive_temporal_advantages(
    raw_rewards,
    rc_rewards,
    gamma,
    coefficients,
    *,
    token_counts=None,
    epsilon=1e-6,
):
    """Preserve sequence-raw credit while redistributing reliable RC returns.

    ``raw_rewards`` and ``rc_rewards`` are ``[group, horizon]``.  The returned
    advantages are blockwise, but their token-weighted mean over the horizon is
    exactly the sequence-level raw advantage for every candidate.  Consequently,
    zero coefficients (and the degenerate ``horizon == 1`` case) reduce exactly
    to sequence-raw GRPO.
    """
    raw_rewards = np.asarray(raw_rewards, dtype=np.float64)
    rc_rewards = np.asarray(rc_rewards, dtype=np.float64)
    if raw_rewards.shape != rc_rewards.shape or raw_rewards.ndim != 2:
        raise ValueError(
            f"raw/RC rewards must share [group,horizon], got "
            f"{raw_rewards.shape} and {rc_rewards.shape}"
        )
    group, horizon = raw_rewards.shape
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (horizon,):
        raise ValueError(
            f"coefficients must have shape {(horizon,)}, got {coefficients.shape}"
        )
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0) or np.any(coefficients > 1.0):
        raise ValueError("coefficients must be finite and lie in [0,1]")
    if token_counts is None:
        token_counts = np.ones(horizon, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.float64)
    if token_counts.shape != (horizon,) or np.any(token_counts <= 0.0):
        raise ValueError("token_counts must be positive with one value per horizon")

    raw_scalar = raw_rewards.mean(axis=1)
    raw_advantage = (raw_scalar - raw_scalar.mean()) / (
        raw_scalar.std() + epsilon
    )
    rc_return = temporal_return_advantages(rc_rewards, gamma)
    active_weight = token_counts * coefficients
    if active_weight.sum() <= epsilon:
        correction = np.zeros_like(rc_return)
    else:
        active_mean = (
            rc_return * active_weight[None, :]
        ).sum(axis=1, keepdims=True) / active_weight.sum()
        correction = coefficients[None, :] * (rc_return - active_mean)
    advantages = raw_advantage[:, None] + correction
    if advantages.shape != (group, horizon):
        raise RuntimeError("unexpected conservative temporal advantage shape")
    return advantages


def influence_adaptive_temporal_advantages(
    raw_rewards,
    rc_rewards,
    gamma,
    coefficients,
    *,
    shuffle_seed=None,
    epsilon=1e-6,
):
    """Blend sequence-raw credit with calibrated RC temporal returns.

    ``coefficients`` contains one frozen delayed-influence weight per future
    frame block.  A zero vector reduces exactly to sequence-raw GRPO.  Setting
    every coefficient to one recovers the aligned RC temporal-return advantage.
    ``shuffle_seed`` provides the candidate-identity control while preserving
    each horizon's reward multiset.
    """
    raw_rewards = np.asarray(raw_rewards, dtype=np.float64)
    rc_rewards = np.asarray(rc_rewards, dtype=np.float64)
    if raw_rewards.shape != rc_rewards.shape or raw_rewards.ndim != 2:
        raise ValueError(
            f"raw/RC rewards must share [group,horizon], got "
            f"{raw_rewards.shape} and {rc_rewards.shape}"
        )
    group, horizon = raw_rewards.shape
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (horizon,):
        raise ValueError(
            f"coefficients must have shape {(horizon,)}, got {coefficients.shape}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    if np.any(coefficients < 0.0) or np.any(coefficients > 1.0):
        raise ValueError("coefficients must lie in [0,1]")

    raw_scalar = raw_rewards.mean(axis=1)
    raw_advantage = (raw_scalar - raw_scalar.mean()) / (
        raw_scalar.std() + epsilon
    )
    rc_return = temporal_return_advantages(
        rc_rewards,
        gamma,
        shuffle_seed=shuffle_seed,
        epsilon=epsilon,
    )
    advantages = (
        (1.0 - coefficients[None, :]) * raw_advantage[:, None]
        + coefficients[None, :] * rc_return
    )
    if advantages.shape != (group, horizon):
        raise RuntimeError("unexpected influence-adaptive advantage shape")
    return advantages


def influence_adaptive_delayed_advantages(
    raw_rewards,
    rc_rewards,
    gamma,
    coefficients,
    *,
    shuffle_seed=None,
    epsilon=1e-6,
):
    """Blend sequence-raw credit with calibrated future-only RC credit."""
    raw_rewards = np.asarray(raw_rewards, dtype=np.float64)
    rc_rewards = np.asarray(rc_rewards, dtype=np.float64)
    if raw_rewards.shape != rc_rewards.shape or raw_rewards.ndim != 2:
        raise ValueError(
            f"raw/RC rewards must share [group,horizon], got "
            f"{raw_rewards.shape} and {rc_rewards.shape}"
        )
    group, horizon = raw_rewards.shape
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (horizon,):
        raise ValueError(
            f"coefficients must have shape {(horizon,)}, got {coefficients.shape}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    if np.any(coefficients < 0.0) or np.any(coefficients > 1.0):
        raise ValueError("coefficients must lie in [0,1]")
    if coefficients[-1] != 0.0:
        raise ValueError("terminal delayed-influence coefficient must be zero")

    raw_scalar = raw_rewards.mean(axis=1)
    raw_advantage = (raw_scalar - raw_scalar.mean()) / (
        raw_scalar.std() + epsilon
    )
    delayed_return = delayed_temporal_return_advantages(
        rc_rewards,
        gamma,
        shuffle_seed=shuffle_seed,
        epsilon=epsilon,
    )
    advantages = (
        (1.0 - coefficients[None, :]) * raw_advantage[:, None]
        + coefficients[None, :] * delayed_return
    )
    if advantages.shape != (group, horizon):
        raise RuntimeError("unexpected delayed influence-adaptive advantage shape")
    return advantages
