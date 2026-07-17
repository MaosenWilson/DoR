"""Exact MSE+LPIPS feature geometry and empirical tangent projection."""

from __future__ import annotations

import math

import torch


class MSELPIPSGeometry:
    """Expose the frozen RLVR MSE+LPIPS reward as squared feature distance."""

    def __init__(self, lpips_model):
        self.lpips_model = lpips_model

    @staticmethod
    def _normalize(value, eps=1e-10):
        return value / torch.sqrt(value.square().sum(dim=1, keepdim=True) + eps)

    @torch.no_grad()
    def extract(self, images):
        images = images.clamp(0.0, 1.0)
        outputs = self.lpips_model.net.forward(
            self.lpips_model.scaling_layer(images * 2.0 - 1.0)
        )
        rgb = images.float().flatten(1) / math.sqrt(images[0].numel())
        blocks = [rgb]
        for feature, layer in zip(outputs, self.lpips_model.lins):
            normalized = self._normalize(feature.float())
            weight = layer.model[1].weight.detach().float().flatten().clamp_min(0.0)
            spatial = feature.shape[-2] * feature.shape[-1]
            weighted = normalized * torch.sqrt(weight)[None, :, None, None]
            blocks.append(weighted.flatten(1) / math.sqrt(spatial))
        return tuple(blocks)

    @staticmethod
    def squared_distance(left, right):
        if len(left) != len(right):
            raise ValueError("feature block tuples must have equal length")
        distance = None
        for left_block, right_block in zip(left, right):
            if left_block.shape[1] != right_block.shape[1]:
                raise ValueError("feature block dimensions must match")
            if right_block.shape[0] == 1:
                right_block = right_block.expand(left_block.shape[0], -1)
            value = (left_block - right_block).square().sum(dim=1)
            distance = value if distance is None else distance + value
        return distance


def metric_tangent_scores(candidate_blocks, reachable_blocks, raw_blocks, ridge=1e-3):
    """Return raw/RC/tangent/reversed squared distances in exact reward geometry."""
    if not (len(candidate_blocks) == len(reachable_blocks) == len(raw_blocks)):
        raise ValueError("candidate and target block tuples must align")
    group = candidate_blocks[0].shape[0]
    if group < 2:
        raise ValueError("at least two candidates are required")
    gram = torch.zeros(group, group, device=candidate_blocks[0].device, dtype=torch.float64)
    rhs = torch.zeros(group, device=gram.device, dtype=torch.float64)
    residual_norm_sq = 0.0
    bases = []
    residuals = []
    for candidates, reachable, raw in zip(candidate_blocks, reachable_blocks, raw_blocks):
        basis = candidates.float() - candidates.float().mean(dim=0, keepdim=True)
        residual = (raw.float() - reachable.float()).reshape(-1)
        bases.append(basis)
        residuals.append(residual)
        gram += (basis @ basis.t()).double()
        rhs += (basis @ residual).double()
        residual_norm_sq += float(residual.double().square().sum().item())
    scale = torch.trace(gram) / group
    regularizer = float(ridge) * torch.clamp(scale, min=1e-12)
    regularizer = torch.clamp(regularizer, min=torch.finfo(torch.float64).eps)
    coefficients = torch.linalg.solve(
        gram + regularizer * torch.eye(group, device=gram.device, dtype=torch.float64),
        rhs,
    ).float()

    raw_distance = torch.zeros(group, device=gram.device)
    rc_distance = torch.zeros_like(raw_distance)
    tangent_distance = torch.zeros_like(raw_distance)
    reversed_distance = torch.zeros_like(raw_distance)
    projection_norm_sq = 0.0
    projection_residual_dot = 0.0
    for candidates, reachable, raw, basis, residual in zip(
        candidate_blocks, reachable_blocks, raw_blocks, bases, residuals
    ):
        projection = coefficients @ basis
        tangent = reachable.reshape(-1) + projection
        reversed_target = reachable.reshape(-1) - projection
        raw_distance += (candidates - raw).square().sum(dim=1)
        rc_distance += (candidates - reachable).square().sum(dim=1)
        tangent_distance += (candidates - tangent[None, :]).square().sum(dim=1)
        reversed_distance += (candidates - reversed_target[None, :]).square().sum(dim=1)
        projection_norm_sq += float(projection.double().square().sum().item())
        projection_residual_dot += float((projection.double() * residual.double()).sum().item())
    projection_norm = math.sqrt(max(projection_norm_sq, 0.0))
    residual_norm = math.sqrt(max(residual_norm_sq, 0.0))
    diagnostics = {
        "projection_ratio": projection_norm / max(residual_norm, 1e-12),
        "residual_cosine": projection_residual_dot
        / max(projection_norm * residual_norm, 1e-12),
    }
    return {
        "raw": raw_distance,
        "rc": rc_distance,
        "tangent": tangent_distance,
        "reversed": reversed_distance,
    }, diagnostics
