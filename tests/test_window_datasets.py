import numpy as np

from vote2world.data.window_dataset import AdaptationWindowDataset, EpisodeRef, EvaluationWindowDataset


def make_episode(length=6):
    images = np.arange(length * 4 * 5 * 3, dtype=np.uint8).reshape(length, 4, 5, 3)
    actions = np.arange(length * 13, dtype=np.float32).reshape(length, 13)
    return EpisodeRef("ep0", images, actions)


def test_adaptation_sample_whitelist_and_shapes():
    dataset = AdaptationWindowDataset([make_episode()])
    sample = dataset[0]
    assert set(sample) == AdaptationWindowDataset.allowed_keys
    assert sample["context_frames"].shape == (4, 3, 4, 5)
    assert sample["context_actions"].shape == (4, 13)
    assert sample["sample_id"] == "ep0:0"


def test_evaluation_sample_has_target_frame():
    dataset = EvaluationWindowDataset([make_episode()])
    sample = dataset[0]
    assert sample["target_frame"].shape == (3, 4, 5)
    np.testing.assert_array_equal(sample["context_actions"][-1], np.arange(3 * 13, 4 * 13))


def test_short_episode_produces_no_windows():
    dataset = AdaptationWindowDataset([make_episode(length=4)])
    assert len(dataset) == 0

