import numpy as np

from dor.action_observability import (
    RidgeDesign,
    action_targets,
    effect_indices,
    episode_folds,
    evaluate_prediction,
    motion_oracle_features,
    split_masks,
)


def toy_frames(n_episodes=6, length=12):
    episode = np.repeat(np.arange(n_episodes), length)
    step = np.tile(np.arange(length), n_episodes)
    return episode, step


def test_effect_rows_share_a_common_domain_for_all_offsets():
    episode, step = toy_frames(n_episodes=2, length=10)
    starts, ends, row_episode, row_step = effect_indices(episode, step, horizon=3)
    assert len(starts) == 12
    assert np.all(step[ends] - step[starts] == 3)
    assert np.all(row_step >= 1)
    assert np.all(episode[starts - 1] == row_episode)
    assert np.all(episode[starts + 3] == row_episode)


def test_episode_folds_cover_each_episode_once():
    folds = episode_folds(20, 5, seed=11)
    assert [len(fold) for fold in folds] == [4] * 5
    assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(20))
    assert all(not (set(a) & set(b)) for i, a in enumerate(folds) for b in folds[i + 1:])


def test_action_targets_obey_offset_and_interval_mean():
    episode, step = toy_frames(n_episodes=2, length=10)
    actions = np.repeat(step[:, None], 13, axis=1).astype(float)
    starts, _, _, _ = effect_indices(episode, step, horizon=3)
    first = action_targets(actions, starts, episode, 3, -1, "first")
    mean = action_targets(actions, starts, episode, 3, 1, "mean")
    assert np.all(first[:, 0] == step[starts] - 1)
    assert np.all(mean[:, 0] == step[starts] + 2)


def test_split_protocols_are_disjoint_and_blocked_has_temporal_order():
    episode, step = toy_frames(n_episodes=10, length=30)
    names = np.asarray([f"ep{i}" for i in range(10)])
    for protocol in ("random", "blocked", "episode"):
        masks = split_masks(protocol, episode, step, 2, names, seed=7)
        assert not np.any(masks["train"] & masks["calibration"])
        assert not np.any(masks["train"] & masks["test"])
        assert all(mask.any() for mask in masks.values())
    blocked = split_masks("blocked", episode, step, 2, names, seed=7)
    for ep in np.unique(episode):
        assert step[(episode == ep) & blocked["train"]].max() < step[(episode == ep) & blocked["test"]].min()


def test_ridge_and_metrics_recover_synthetic_action_signal():
    rng = np.random.default_rng(9)
    episodes = np.repeat(np.arange(8), 30)
    x = rng.normal(size=(len(episodes), 16))
    weight = rng.normal(size=(16, 6))
    y = x @ weight + 0.02 * rng.normal(size=(len(x), 6))
    train = np.arange(0, 160)
    test = np.arange(160, len(x))
    design = RidgeDesign.from_array(x[train], "cpu")
    pred = design.fit_predict(y[train], x[test], alpha=0.01)
    metrics = evaluate_prediction(y[train], y[test], pred, episodes[test], n_boot=200, seed=3)
    assert metrics["mean_r2"] > 0.99
    assert metrics["positive_dims"] == 6
    assert metrics["direction_balanced_accuracy"] > 0.95
    assert metrics["retrieval"]["q05"] > 0.9


def test_motion_oracle_features_keep_state_and_signed_flow():
    import torch

    current = torch.ones(2, 3, 8, 10)
    flow = torch.zeros(2, 2, 4, 5)
    flow[0, 0] = 2
    flow[1, 1] = -3
    features, state_dim = motion_oracle_features(current, flow, (2, 2))
    assert state_dim == 12
    assert features.shape == (2, 32)  # 3 RGB + 5 flow channels, each over 2x2.
    assert torch.all(features[:, :state_dim] == 1)
    assert not torch.allclose(features[0, state_dim:], features[1, state_dim:])
