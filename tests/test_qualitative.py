import numpy as np
import pytest

from dor.qualitative import (
    SceneCandidate,
    absolute_residual,
    scene_feature,
    select_diverse_scenes,
    temporal_motion,
)


def _candidate(name, motion, feature):
    return SceneCandidate(name, 0, motion, np.asarray(feature, dtype=np.float32))


def test_temporal_motion_is_zero_for_static_video_and_positive_for_change():
    static = np.zeros((4, 8, 10, 3), dtype=np.uint8)
    moving = static.copy()
    moving[2:] = 255
    assert temporal_motion(static) == 0.0
    assert temporal_motion(moving) > 0.0


def test_scene_feature_accepts_thwc_and_tchw():
    rng = np.random.default_rng(3)
    thwc = rng.integers(0, 256, size=(5, 16, 20, 3), dtype=np.uint8)
    tchw = np.moveaxis(thwc, -1, 1)
    np.testing.assert_allclose(scene_feature(thwc), scene_feature(tchw))


def test_diverse_selection_starts_with_highest_motion_then_farthest():
    candidates = [
        _candidate("episode_a", 0.9, [0.0, 0.0]),
        _candidate("episode_b", 0.2, [0.1, 0.1]),
        _candidate("episode_c", 0.3, [5.0, 5.0]),
    ]
    selected = select_diverse_scenes(candidates, 2, motion_weight=0.0)
    assert [item.episode for item in selected] == ["episode_a", "episode_c"]


def test_diverse_selection_is_deterministic_under_input_permutation():
    candidates = [
        _candidate("episode_a", 0.3, [0.0, 0.0]),
        _candidate("episode_b", 0.5, [1.0, 1.0]),
        _candidate("episode_c", 0.4, [3.0, 3.0]),
    ]
    forward = select_diverse_scenes(candidates, 3)
    reverse = select_diverse_scenes(list(reversed(candidates)), 3)
    assert [item.episode for item in forward] == [item.episode for item in reverse]


def test_absolute_residual_has_fixed_physical_scale():
    gt = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    prediction = np.full_like(gt, 64)
    residual = absolute_residual(gt, prediction)
    np.testing.assert_allclose(residual, 64.0 / 255.0)


def test_selection_rejects_invalid_count():
    with pytest.raises(ValueError):
        select_diverse_scenes([_candidate("episode", 0.1, [0.0])], 2)
