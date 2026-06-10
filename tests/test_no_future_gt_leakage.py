import numpy as np
import pytest

from vote2world.data.leakage import FutureGTLeakageError, assert_no_future_gt
from vote2world.data.window_dataset import AdaptationWindowDataset, EpisodeRef, EvaluationWindowDataset
from vote2world.processors.adaptation_processor import DummyVisualTokenizer, GenerationOnlyAdaptationProcessor


def make_episode():
    return EpisodeRef(
        "ep0",
        np.zeros((6, 4, 5, 3), dtype=np.uint8),
        np.zeros((6, 13), dtype=np.float32),
    )


def test_adaptation_dataset_has_no_gt():
    sample = AdaptationWindowDataset([make_episode()])[0]
    assert "target_frame" not in sample
    assert "future_frame" not in sample
    assert_no_future_gt(sample)


def test_evaluation_dataset_has_gt_only_there():
    sample = EvaluationWindowDataset([make_episode()])[0]
    assert "target_frame" in sample


def test_adaptation_processor_rejects_gt():
    ranges = np.stack([np.full(13, -1.0), np.full(13, 1.0)], axis=1).astype(np.float32)
    processor = GenerationOnlyAdaptationProcessor(DummyVisualTokenizer(), ranges)
    with pytest.raises(FutureGTLeakageError):
        processor(
            {
                "context_frames": np.zeros((1, 4, 3, 8, 10), dtype=np.float32),
                "context_actions": np.zeros((1, 4, 13), dtype=np.float32),
                "target_frame": np.zeros((1, 3, 8, 10), dtype=np.float32),
            }
        )

