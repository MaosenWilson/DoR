import numpy as np


def _group_advantage(reward, epsilon=1e-6):
    reward = np.asarray(reward, dtype=np.float64)
    return (reward - reward.mean()) / (reward.std() + epsilon)


def test_context_constant_floor_subtraction_cannot_change_grpo_advantage():
    reward = np.array([-0.31, -0.22, -0.27, -0.19, -0.44])
    floor = 0.083
    np.testing.assert_allclose(
        _group_advantage(reward),
        _group_advantage(reward - floor),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.argsort(reward), np.argsort(reward - floor)
    )
