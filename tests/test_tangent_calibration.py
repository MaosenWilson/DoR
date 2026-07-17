import torch

from dor.tangent_calibration import empirical_tangent_target


def test_tangent_target_recovers_residual_inside_candidate_span():
    reachable = torch.zeros(1, 1, 2)
    raw = torch.tensor([[[0.5, 0.0]]])
    candidates = torch.tensor([
        [[[0.0, 0.0]]],
        [[[1.0, 0.0]]],
        [[[0.5, 0.0]]],
    ])
    target, projection, diagnostics = empirical_tangent_target(
        candidates, reachable, raw, ridge=1e-8
    )
    assert target[0, 0, 0] > 0.49
    assert abs(float(projection[0, 0, 1])) < 1e-6
    assert diagnostics["projection_ratio"] > 0.99


def test_tangent_target_rejects_residual_orthogonal_to_candidates():
    reachable = torch.zeros(1, 1, 2)
    raw = torch.tensor([[[0.0, 0.5]]])
    candidates = torch.tensor([
        [[[0.0, 0.0]]],
        [[[1.0, 0.0]]],
        [[[0.5, 0.0]]],
    ])
    target, projection, diagnostics = empirical_tangent_target(
        candidates, reachable, raw
    )
    torch.testing.assert_close(target, reachable)
    torch.testing.assert_close(projection, torch.zeros_like(projection))
    assert diagnostics["projection_ratio"] == 0.0
