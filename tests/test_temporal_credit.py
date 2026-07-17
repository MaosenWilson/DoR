import numpy as np

from dor.temporal_credit import (
    discounted_returns,
    equalize_reward_scale_by_horizon,
    reachability_consistent_temporal_advantages,
    reachability_consistent_temporal_scores,
    scale_equalized_temporal_return_advantages,
    shuffle_candidate_correspondence,
    temporal_return_advantages,
)


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
