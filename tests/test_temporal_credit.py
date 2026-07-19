import numpy as np

from dor.temporal_credit import (
    conservative_adaptive_temporal_advantages,
    delayed_temporal_return_advantages,
    discounted_returns,
    equalize_reward_scale_by_horizon,
    gae_frame_advantages,
    influence_adaptive_delayed_advantages,
    influence_adaptive_temporal_advantages,
    normalize_by_horizon,
    reachability_consistent_temporal_advantages,
    reachability_consistent_temporal_scores,
    scale_equalized_temporal_return_advantages,
    shuffle_candidate_correspondence,
    temporal_return_advantages,
)


def test_gae_frame_advantages_reduce_to_endpoints():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(16, 7))
    # lam=1 reduces exactly to temporal-return advantages
    np.testing.assert_allclose(
        gae_frame_advantages(rewards, 0.95, 1.0),
        temporal_return_advantages(rewards, 0.95), atol=1e-12,
    )
    # lam=0 reduces exactly to per-horizon frame-only advantages
    np.testing.assert_allclose(
        gae_frame_advantages(rewards, 0.95, 0.0),
        normalize_by_horizon(rewards), atol=1e-12,
    )


def test_gae_frame_advantages_are_horizon_normalized_and_bounded():
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=(16, 10))
    for lam in (0.0, 0.3, 0.7, 0.9, 1.0):
        a = gae_frame_advantages(rewards, 0.95, lam)
        assert a.shape == rewards.shape
        np.testing.assert_allclose(a.mean(axis=0), 0.0, atol=1e-9)
        assert np.all(np.isfinite(a))
    # interior lam controls early-block variance: variance of the earliest block's
    # advantage input (pre-normalization return-to-go) shrinks as lam decreases.
    import numpy as _np
    def early_var(lam):
        res = rewards - rewards.mean(axis=0, keepdims=True)
        g = discounted_returns(res, 0.95 * lam)
        return float(_np.var(g[:, 0]))
    assert early_var(0.3) < early_var(0.9) < early_var(1.0)


def test_gae_frame_advantages_rejects_bad_lambda():
    import pytest
    with pytest.raises(ValueError):
        gae_frame_advantages(np.zeros((4, 3)), 0.95, 1.5)


def test_discounted_return_full_and_truncated():
    rewards = np.array([[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(
        discounted_returns(rewards, 0.5), [[3.0, 4.0, 4.0]]
    )
    np.testing.assert_allclose(
        discounted_returns(rewards, 0.5, max_terms=2), [[2.0, 4.0, 4.0]]
    )


def test_candidate_shuffle_preserves_each_horizon_multiset():
    rewards = np.arange(20, dtype=np.float64).reshape(5, 4)
    shuffled = shuffle_candidate_correspondence(rewards, seed=7)
    for horizon in range(rewards.shape[1]):
        np.testing.assert_array_equal(
            np.sort(shuffled[:, horizon]), np.sort(rewards[:, horizon])
        )
    assert not np.array_equal(shuffled, rewards)


def test_temporal_advantages_are_group_normalized_per_horizon():
    rewards = np.array([
        [0.0, 1.0, 0.5],
        [0.0, 2.0, 1.5],
        [0.0, 4.0, 3.0],
    ])
    advantages = temporal_return_advantages(rewards, 0.95)
    np.testing.assert_allclose(advantages.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(advantages.std(axis=0)[1:], 1.0, atol=2e-6)


def test_scale_equalization_removes_horizon_magnitude_without_changing_order():
    rewards = np.array([
        [0.0, 0.0],
        [1.0, 100.0],
        [2.0, 200.0],
    ])
    equalized = equalize_reward_scale_by_horizon(rewards)
    np.testing.assert_allclose(equalized.std(axis=0), 1.0, atol=2e-6)
    np.testing.assert_array_equal(np.argsort(equalized, axis=0), np.argsort(rewards, axis=0))
    advantages = scale_equalized_temporal_return_advantages(rewards, 0.95)
    np.testing.assert_allclose(advantages.mean(axis=0), 0.0, atol=1e-12)


def test_reachability_consistent_scores_abstain_on_conflicting_pair():
    raw = np.array([[3.0], [2.0], [1.0]])
    rc = np.array([[3.0], [1.0], [2.0]])
    scores, coverage = reachability_consistent_temporal_scores(raw, rc, gamma=0.95)

    assert scores.shape == raw.shape
    assert np.isclose(coverage[0], 2.0 / 3.0)
    assert scores[0, 0] > 0.0
    assert np.isclose(scores.sum(), 0.0)


def test_reachability_consistent_advantages_are_block_normalized():
    raw = np.array([[3.0, 1.0], [2.0, 3.0], [1.0, 2.0]])
    rc = np.array([[4.0, 1.0], [2.0, 2.5], [1.0, 3.0]])
    advantage = reachability_consistent_temporal_advantages(raw, rc, gamma=0.95)

    assert advantage.shape == raw.shape
    np.testing.assert_allclose(advantage.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(advantage.std(axis=0), 1.0, atol=1e-5)


def test_conservative_adaptive_credit_preserves_token_weighted_anchor():
    raw = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [4.0, 3.0, 1.0],
    ])
    rc = np.array([
        [1.5, 1.0, 4.0],
        [1.0, 3.0, 2.0],
        [4.0, 2.0, 1.5],
    ])
    token_counts = np.array([2.0, 3.0, 5.0])
    advantage = conservative_adaptive_temporal_advantages(
        raw, rc, 0.95, [0.8, 0.3, 0.0], token_counts=token_counts
    )
    scalar = raw.mean(axis=1)
    anchor = (scalar - scalar.mean()) / (scalar.std() + 1e-6)
    weighted = (advantage * token_counts[None, :]).sum(axis=1) / token_counts.sum()

    np.testing.assert_allclose(weighted, anchor, atol=1e-12)
    assert np.sqrt(np.mean((advantage - anchor[:, None]) ** 2)) > 0.0
    np.testing.assert_allclose(advantage[:, 2], anchor, atol=1e-12)


def test_conservative_adaptive_credit_zero_and_single_step_reduce_to_sequence_raw():
    raw = np.array([[1.0, 2.0], [2.0, 1.0], [4.0, 2.0]])
    rc = raw[:, ::-1]
    zero = conservative_adaptive_temporal_advantages(raw, rc, 0.95, [0.0, 0.0])
    scalar = raw.mean(axis=1)
    anchor = (scalar - scalar.mean()) / (scalar.std() + 1e-6)
    np.testing.assert_allclose(zero, np.repeat(anchor[:, None], 2, axis=1))

    single = conservative_adaptive_temporal_advantages(
        raw[:, :1], rc[:, :1], 0.95, [1.0]
    )
    single_scalar = raw[:, 0]
    single_anchor = (single_scalar - single_scalar.mean()) / (
        single_scalar.std() + 1e-6
    )
    np.testing.assert_allclose(single[:, 0], single_anchor, atol=1e-12)


def test_influence_adaptive_credit_has_exact_zero_and_one_limits():
    raw = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [4.0, 3.0, 1.0],
    ])
    rc = np.array([
        [1.5, 1.0, 4.0],
        [1.0, 3.0, 2.0],
        [4.0, 2.0, 1.5],
    ])
    scalar = raw.mean(axis=1)
    raw_anchor = (scalar - scalar.mean()) / (scalar.std() + 1e-6)

    zero = influence_adaptive_temporal_advantages(
        raw, rc, 0.95, [0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        zero, np.repeat(raw_anchor[:, None], 3, axis=1), atol=1e-12
    )

    one = influence_adaptive_temporal_advantages(
        raw, rc, 0.95, [1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        one, temporal_return_advantages(rc, 0.95), atol=1e-12
    )


def test_influence_adaptive_shuffle_changes_identity_not_reward_marginals():
    raw = np.arange(24, dtype=np.float64).reshape(6, 4)
    rc = np.array([
        [0.0, 2.0, 1.0, 8.0],
        [1.0, 5.0, 2.0, 7.0],
        [2.0, 1.0, 7.0, 6.0],
        [3.0, 4.0, 4.0, 5.0],
        [4.0, 0.0, 8.0, 4.0],
        [5.0, 3.0, 5.0, 3.0],
    ])
    aligned = influence_adaptive_temporal_advantages(
        raw, rc, 0.95, [0.7, 0.7, 0.7, 0.0]
    )
    shuffled = influence_adaptive_temporal_advantages(
        raw, rc, 0.95, [0.7, 0.7, 0.7, 0.0], shuffle_seed=17
    )

    assert not np.allclose(aligned[:, :3], shuffled[:, :3])
    np.testing.assert_allclose(aligned[:, 3], shuffled[:, 3], atol=1e-12)


def test_delayed_return_uses_only_strictly_future_rewards():
    rewards = np.array([
        [100.0, 1.0, 4.0],
        [-100.0, 2.0, 3.0],
        [0.0, 4.0, 1.0],
    ])
    delayed = delayed_temporal_return_advantages(rewards, 0.5)
    expected_first = np.array([3.0, 3.5, 4.5])
    expected_first = (expected_first - expected_first.mean()) / (
        expected_first.std() + 1e-6
    )
    np.testing.assert_allclose(delayed[:, 0], expected_first, atol=1e-12)
    np.testing.assert_allclose(delayed[:, -1], 0.0, atol=1e-12)


def test_influence_adaptive_delayed_limits_and_terminal_guard():
    raw = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [4.0, 3.0, 1.0]])
    rc = np.array([[1.5, 1.0, 4.0], [1.0, 3.0, 2.0], [4.0, 2.0, 1.5]])
    scalar = raw.mean(axis=1)
    anchor = (scalar - scalar.mean()) / (scalar.std() + 1e-6)
    zero = influence_adaptive_delayed_advantages(raw, rc, 0.95, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(zero, np.repeat(anchor[:, None], 3, axis=1))
    full = influence_adaptive_delayed_advantages(raw, rc, 0.95, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(
        full[:, :2], delayed_temporal_return_advantages(rc, 0.95)[:, :2]
    )
    np.testing.assert_allclose(full[:, -1], anchor)

    with np.testing.assert_raises(ValueError):
        influence_adaptive_delayed_advantages(raw, rc, 0.95, [0.0, 0.0, 0.1])
