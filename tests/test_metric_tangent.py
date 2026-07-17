import torch

from dor.metric_tangent import metric_tangent_scores


def test_metric_tangent_recovers_actionable_residual():
    candidates = (torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),)
    reachable = (torch.tensor([[0.0, 0.0]]),)
    raw = (torch.tensor([[0.5, 0.0]]),)
    scores, diagnostics = metric_tangent_scores(
        candidates, reachable, raw, ridge=1e-8
    )
    torch.testing.assert_close(scores["tangent"], scores["raw"], atol=1e-6, rtol=1e-6)
    assert diagnostics["projection_ratio"] > 0.99


def test_metric_tangent_drops_orthogonal_residual():
    candidates = (torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),)
    reachable = (torch.tensor([[0.0, 0.0]]),)
    raw = (torch.tensor([[0.0, 0.5]]),)
    scores, diagnostics = metric_tangent_scores(candidates, reachable, raw)
    torch.testing.assert_close(scores["tangent"], scores["rc"])
    assert diagnostics["projection_ratio"] == 0.0
