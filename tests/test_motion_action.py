import numpy as np

from dor.motion_action import action_score, episode_fold, predict_model


def payload():
    return {
        "x_mean": np.zeros((2, 3)),
        "x_scale": np.ones((2, 3)),
        "y_mean": np.zeros((2, 6)),
        "y_scale": np.ones((2, 6)),
        "coef": np.stack([np.eye(3, 6), np.eye(3, 6)]),
        "residual_scale": np.ones(6),
        "dimension_weights": np.asarray([1, 1, 1, 0, 0, 0], dtype=float) / 3,
        "action_dims": np.asarray([0, 1, 2, 4, 5, 6]),
        "fold_test_episode_names": np.asarray([["a", "b"], ["c", "d"]]),
        "metadata": {"pool": "8x10"},
    }


def test_crossfit_payload_uses_episode_specific_model():
    p = payload()
    assert episode_fold(p, "a") == 0
    assert episode_fold(p, "d") == 1
    pred = predict_model(p, 0, np.asarray([[1.0, 2.0, 3.0]]))
    assert np.allclose(pred[0, :3], [1, 2, 3])


def test_reliability_weighted_score_ignores_unobservable_dimensions():
    p = payload()
    x = np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    action = np.zeros(13)
    action[[0, 1, 2]] = [1, 2, 3]
    action[[4, 5, 6]] = [999, 999, 999]
    score = action_score(p, 0, x, action)
    assert score[0] > score[1]
