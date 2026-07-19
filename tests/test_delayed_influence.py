import numpy as np

from dor.delayed_influence import (
    delayed_influence_ratio,
    shuffle_continuations_within_group,
)


def _synthetic(delayed, seed=7):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"ep{value}" for value in range(10)], 8)
    immediate = rng.normal(size=len(groups))
    prefix_effect = rng.normal(scale=1.0 if delayed else 0.0, size=len(groups))
    future = (
        2.0 * immediate[:, None]
        + prefix_effect[:, None]
        + rng.normal(scale=0.15, size=(len(groups), 6))
    )
    contexts = np.repeat([f"ctx{value}" for value in range(20)], 4)
    return immediate, future, groups, contexts


def test_delayed_influence_is_small_when_immediate_reward_is_sufficient():
    immediate, future, groups, _ = _synthetic(delayed=False)
    report = delayed_influence_ratio(immediate, future, groups, folds=5, seed=11)

    assert report["oof_immediate_r2"] > 0.95
    assert report["coefficient"] < 0.1


def test_delayed_influence_detects_stable_prefix_effect_and_shuffle_removes_it():
    immediate, future, groups, contexts = _synthetic(delayed=True)
    report = delayed_influence_ratio(immediate, future, groups, folds=5, seed=11)
    shuffled = shuffle_continuations_within_group(future, contexts, seed=13)
    null = delayed_influence_ratio(immediate, shuffled, groups, folds=5, seed=11)

    assert report["coefficient"] > 0.8
    assert null["coefficient"] < report["coefficient"] - 0.4
