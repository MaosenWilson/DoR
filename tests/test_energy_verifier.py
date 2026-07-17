import json

import numpy as np
import pytest

from dor.energy_verifier import (
    combine_block_distances,
    cross_group_utility,
    energy_influence,
    energy_objective,
    load_energy_config,
    make_energy_config,
    pair_uncertainty_threshold,
    project_certified_order,
    project_reliable_order,
    radial_residual_reward,
    rowwise_correlation,
    top_safe_energy_reward,
)


def test_combine_block_distances_is_equal_block_hilbert_norm():
    blocks = np.array([[3.0, 4.0], [6.0, 8.0]])
    scales = np.array([3.0, 4.0])
    got = combine_block_distances(blocks, scales)
    np.testing.assert_allclose(got, [1.0, 2.0])


def test_energy_influence_has_full_pairwise_coefficient():
    target = np.array([1.0, 2.0, 4.0])
    pairwise = np.array([[0.0, 3.0, 6.0], [3.0, 0.0, 9.0], [6.0, 9.0, 0.0]])
    got = energy_influence(target, pairwise)
    np.testing.assert_allclose(got, [-1.0 + 4.5, -2.0 + 6.0, -4.0 + 7.5])
    assert energy_objective(target, pairwise) == pytest.approx(-target.mean() + 3.0)


def test_energy_influence_matches_population_score_gradient():
    outcomes = np.array([0.0, 2.0])
    target = 0.0

    def objective(logit):
        p1 = 1.0 / (1.0 + np.exp(-logit))
        probs = np.array([1.0 - p1, p1])
        target_distance = np.abs(outcomes - target)
        pairwise = np.abs(outcomes[:, None] - outcomes[None, :])
        return -probs @ target_distance + 0.5 * np.sum(
            probs[:, None] * probs[None, :] * pairwise
        )

    logit = 0.37
    p1 = 1.0 / (1.0 + np.exp(-logit))
    probs = np.array([1.0 - p1, p1])
    target_distance = np.abs(outcomes - target)
    pairwise = np.abs(outcomes[:, None] - outcomes[None, :])
    population_influence = -target_distance + pairwise @ probs
    score = np.array([-p1, 1.0 - p1])
    analytic = np.sum(probs * population_influence * score)
    eps = 1e-6
    numeric = (objective(logit + eps) - objective(logit - eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, abs=1e-7)


def test_cross_group_utility_uses_independent_group_mean():
    target = np.array([1.0, 3.0])
    cross = np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(cross_group_utility(target, cross), [3.0, -1.0])


def test_rowwise_correlation_supports_spearman_and_constant_rows():
    a = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    b = np.array([[2.0, 4.0, 8.0], [1.0, 2.0, 3.0]])
    assert rowwise_correlation(a, b, "spearman")[0] == pytest.approx(1.0)
    assert np.isnan(rowwise_correlation(a, b, "pearson")[1])


def test_energy_config_round_trip_and_validation(tmp_path):
    path = tmp_path / "energy.json"
    payload = make_energy_config(["rgb", "vgg0"], [0.1, 0.2], metadata={"K": 16})
    path.write_text(json.dumps(payload))
    loaded = load_energy_config(str(path))
    assert loaded["energy_beta"] == 1.0
    assert loaded["metadata"]["K"] == 16
    with pytest.raises(ValueError):
        combine_block_distances(np.ones((2, 2)), [1.0, 1.0], beta=2.0)
    with pytest.raises(ValueError):
        energy_influence(np.ones(1), np.zeros((1, 1)))


def test_pair_uncertainty_threshold_uses_pairwise_target_perturbation():
    reachable = np.zeros((2, 3))
    raw = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    expected = np.quantile([1.0, 2.0, 1.0, 2.0, 4.0, 2.0], 0.75)
    assert pair_uncertainty_threshold(raw, reachable, 0.75) == pytest.approx(expected)


def test_reliable_order_projection_preserves_only_confident_pairs():
    rc = np.array([[3.0, 2.0, 1.9]])
    energy = np.array([[0.0, 4.0, 5.0]])
    projected = project_reliable_order(energy, rc, threshold=0.5)
    assert projected[0, 0] >= projected[0, 1]
    assert projected[0, 0] >= projected[0, 2]
    assert np.isfinite(projected).all()


def test_certified_projection_uses_pair_specific_decoder_interaction():
    rc = np.array([[3.0, 2.0, 1.0]])
    reachable = -rc
    raw = reachable + np.array([[0.0, 0.2, 2.0]])
    energy = np.array([[0.0, 4.0, 5.0]])
    projected = project_certified_order(energy, rc, raw, reachable)
    # 0 > 1 is certified (margin 1 > interaction 0.2).
    assert projected[0, 0] >= projected[0, 1]
    # 0 > 2 is not certified (margin 2 is not greater than interaction 2).
    assert np.isfinite(projected).all()


def test_radial_residual_is_orthogonal_to_target_radius():
    target = np.array([[1.0, 2.0, 3.0, 4.0]])
    rc = -target
    pair = 2.0 * target + np.array([[1.0, -1.0, -1.0, 1.0]])
    reward = radial_residual_reward(rc, pair, target)
    residual = reward - rc
    centered_target = target - target.mean(axis=1, keepdims=True)
    assert float(np.sum(residual * centered_target)) == pytest.approx(0.0, abs=1e-10)


def test_top_safe_energy_preserves_rc_top_without_fixed_coefficient():
    rc = np.array([[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]])
    pair = np.array([[0.0, 4.0, 1.0], [0.0, 0.1, 0.2]])
    reward, coefficient = top_safe_energy_reward(rc, pair)
    np.testing.assert_array_equal(np.argmax(reward, axis=1), np.argmax(rc, axis=1))
    assert coefficient[0] < 1.0
    assert coefficient[1] == pytest.approx(1.0)


def test_top_safe_energy_breaks_crossing_tie_toward_rc_top():
    rc = np.array([[1.0, 0.0]])
    pair = np.array([[0.0, 1.0]])
    reward, coefficient = top_safe_energy_reward(rc, pair)
    assert coefficient[0] == pytest.approx(1.0)
    assert int(np.argmax(reward[0])) == 0
    assert reward[0, 0] > reward[0, 1]
