import numpy as np
import pytest
import torch

from dor.adapters.ivideogpt_robonet import load_robonet_window_npz


def _sample(path, *, frames=20, actions=19):
    image = np.zeros((frames, 8, 12, 3), dtype=np.uint8)
    for index in range(frames):
        image[index] = index
    action = np.arange(actions * 5, dtype=np.float32).reshape(actions, 5)
    np.savez(path, image=image, action=action)
    return image, action


def test_robonet_window_preserves_transition_alignment_and_appends_zero(tmp_path):
    path = tmp_path / "episode.npz"
    _, action = _sample(path)

    window = load_robonet_window_npz(path, start=3, horizon=4, resolution=8)

    assert window.frames.shape == (6, 3, 8, 8)
    assert window.actions.shape == (6, 5)
    assert torch.equal(window.actions[:-1], torch.from_numpy(action[3:8]))
    assert torch.count_nonzero(window.actions[-1]).item() == 0
    assert window.horizon == 4


def test_robonet_window_rejects_non_transition_aligned_actions(tmp_path):
    path = tmp_path / "bad.npz"
    _sample(path, frames=20, actions=20)

    with pytest.raises(ValueError, match="T frames and T-1 actions"):
        load_robonet_window_npz(path, horizon=4)


def test_robonet_window_rejects_overrun(tmp_path):
    path = tmp_path / "episode.npz"
    _sample(path)

    with pytest.raises(ValueError, match="exceeds trajectory length"):
        load_robonet_window_npz(path, start=10, horizon=10)
