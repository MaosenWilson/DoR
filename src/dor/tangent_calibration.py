"""Empirical reachable-tangent calibration for decoded candidate groups."""

from __future__ import annotations

import torch


def empirical_tangent_target(candidates, reachable, raw, ridge=1e-3):
    """Add back only the GT residual spanned by candidate variations.

    ``candidates`` is ``[K,C,H,W]`` while ``reachable`` and ``raw`` are
    ``[C,H,W]``.  The projection is computed through a ``K x K`` Gram system,
    so the high-dimensional image basis is never explicitly factorized.
    Returns ``(target, projection, diagnostics)``.
    """
    if candidates.ndim != 4 or reachable.ndim != 3 or raw.ndim != 3:
        raise ValueError("expected candidates [K,C,H,W] and targets [C,H,W]")
    if candidates.shape[1:] != reachable.shape or raw.shape != reachable.shape:
        raise ValueError("candidate and target image shapes must match")
    if candidates.shape[0] < 2:
        raise ValueError("at least two candidates are required")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    dtype = torch.float32
    flat = candidates.detach().to(dtype).flatten(1)
    basis = flat - flat.mean(dim=0, keepdim=True)
    residual = (raw.detach().to(dtype) - reachable.detach().to(dtype)).flatten()
    dimension = float(basis.shape[1])
    gram = basis @ basis.t() / dimension
    rhs = basis @ residual / dimension
    gram_solve = gram.double()
    rhs_solve = rhs.double()
    scale = torch.trace(gram_solve) / max(int(gram.shape[0]), 1)
    regularizer = float(ridge) * torch.clamp(scale, min=1e-12)
    regularizer = torch.clamp(regularizer, min=torch.finfo(torch.float64).eps)
    coefficients = torch.linalg.solve(
        gram_solve
        + regularizer * torch.eye(
            gram.shape[0], device=gram.device, dtype=torch.float64
        ),
        rhs_solve,
    ).to(dtype)
    projection = (coefficients @ basis).reshape_as(reachable)
    target = (reachable.detach().to(dtype) + projection).clamp(0.0, 1.0)
    residual_norm = torch.linalg.vector_norm(residual)
    projection_norm = torch.linalg.vector_norm(projection)
    diagnostics = {
        "projection_ratio": float(projection_norm / (residual_norm + 1e-12)),
        "residual_cosine": float(
            torch.dot(projection.flatten(), residual)
            / (projection_norm * residual_norm + 1e-12)
        ),
        "gram_rank": int(torch.linalg.matrix_rank(gram).item()),
    }
    return target, projection, diagnostics
