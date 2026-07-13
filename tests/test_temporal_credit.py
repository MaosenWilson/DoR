import numpy as np

from dor.temporal_credit import (
    discounted_returns,
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
