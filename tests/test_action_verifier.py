import numpy as np
import torch

from dor.action_verifier import (
    ARM_MOTION_DIMS,
    action_score,
    fit_grouped_ridge,
    predict,
    split_episodes,
    transition_features,
)
from dor.constants import MOTION_DIMS


def test_action_verifier_uses_canonical_rt1_motion_dimensions():
    assert ARM_MOTION_DIMS == tuple(MOTION_DIMS) == (0, 1, 2, 4, 5, 6)


def test_transition_features_preserve_state_signed_and_absolute_delta():
    z0 = torch.zeros(2, 2, 4, 5)
    z1 = z0.clone()
    z1[0] += 2.0
    z1[1] -= 3.0
    feat = transition_features(z0, z1, (2, 1))
    assert feat.shape == (2, 12)
    state, delta, absolute = feat.chunk(3, dim=1)
    assert torch.all(state == 0)
    assert torch.all(delta[0] == 2) and torch.all(delta[1] == -3)
    assert torch.all(absolute[0] == 2) and torch.all(absolute[1] == 3)


def test_episode_split_is_deterministic_and_disjoint():
    names = [f"ep{i}" for i in range(20)]
    a = split_episodes(names, seed=7)
    b = split_episodes(reversed(names), seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert len(a["train"]) == 12 and len(a["calibration"]) == 4 and len(a["test"]) == 4
    assert not (set(a["train"]) & set(a["test"]))


def test_grouped_ridge_detects_real_signal_and_scores_matching_action_higher():
    rng = np.random.default_rng(4)
    episodes = np.repeat([f"ep{i:02d}" for i in range(20)], 24)
    x = rng.normal(size=(len(episodes), 18))
    w = rng.normal(size=(18, len(ARM_MOTION_DIMS)))
    motion = x @ w + 0.03 * rng.normal(size=(len(x), len(ARM_MOTION_DIMS)))
    actions = np.zeros((len(x), 13), dtype=np.float64)
    actions[:, ARM_MOTION_DIMS] = motion
    payload, report = fit_grouped_ridge(
        x, actions, episodes, permutations=199, permutation_seed=9
    )
    assert report["green"]
    assert report["test_positive_dims"] == 6
    pred = predict(payload, x[:4])
    assert pred.shape == (4, 6)
    matching = action_score(payload, x[:4], actions[:4])
    wrong = actions[:4].copy()
    wrong[:, ARM_MOTION_DIMS] += 5.0
    assert np.all(matching > action_score(payload, x[:4], wrong))
