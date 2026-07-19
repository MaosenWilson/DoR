import numpy as np

from dor.branch_credit import branch_td_credit, heldout_branch_rows
from scripts.external.gate_vp2_branch_credit import analyze


def test_branch_td_credit_has_immediate_limit():
    immediate = np.array([[1.0, 2.0]])
    value = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(
        branch_td_credit(immediate, value, value, gamma=1.0), immediate
    )


def test_heldout_branch_rows_require_independent_draws():
    immediate = np.arange(6, dtype=float)
    future = np.ones((6, 3))
    contexts = np.repeat(["a", "b"], 3)
    try:
        heldout_branch_rows(immediate, future, future, contexts)
    except ValueError as error:
        assert "at least four" in str(error)
    else:
        raise AssertionError("three draws must not pass the branch gate")


def test_gate_detects_candidate_specific_delayed_credit(tmp_path):
    rng = np.random.default_rng(19)
    contexts, group, draws = 48, 4, 6
    names = np.repeat([f"ep{index}" for index in range(contexts)], group)
    payload = {}
    cumulative = rng.normal(scale=0.2, size=(contexts, group))
    for horizon in range(1, 5):
        contribution = rng.normal(scale=0.8, size=(contexts, group))
        before = cumulative.copy()
        cumulative = cumulative + contribution
        immediate = rng.normal(scale=0.1, size=(contexts, group))
        payload[f"immediate_p{horizon}"] = immediate.reshape(-1).astype(np.float32)
        payload[f"future_p{horizon}"] = (
            cumulative[..., None] + rng.normal(scale=0.08, size=(contexts, group, draws))
        ).reshape(-1, draws).astype(np.float32)
        payload[f"context_p{horizon}"] = names
        payload[f"episode_p{horizon}"] = names
    cache = tmp_path / "branch.npz"
    np.savez_compressed(cache, **payload)
    report = analyze(cache, gamma=1.0, bootstrap=500, seed=3, min_horizons=2)
    assert report["verdict"] == "GREEN"
    assert len(report["green_horizons"]) >= 2
