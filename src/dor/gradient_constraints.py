"""One-sided gradient constraints for fidelity-anchored policy updates."""

from __future__ import annotations

import math


def projection_statistics(dot, primary_norm_sq, auxiliary_norm_sq, eps=1e-20):
    """Closed-form statistics for projecting auxiliary onto ``g_p^T v >= 0``."""
    dot = float(dot)
    primary_norm_sq = float(primary_norm_sq)
    auxiliary_norm_sq = float(auxiliary_norm_sq)
    if primary_norm_sq <= eps or auxiliary_norm_sq <= eps:
        raise ValueError("primary and auxiliary gradients must both be non-zero")
    cosine = dot / math.sqrt(primary_norm_sq * auxiliary_norm_sq)
    conflict = dot < 0.0
    coefficient = dot / primary_norm_sq if conflict else 0.0
    projected_norm_sq = (
        max(auxiliary_norm_sq - dot * dot / primary_norm_sq, 0.0)
        if conflict
        else auxiliary_norm_sq
    )
    cap = min(1.0, math.sqrt(primary_norm_sq / max(projected_norm_sq, eps)))
    retained_ratio = cap * math.sqrt(projected_norm_sq / auxiliary_norm_sq)
    projected_anchor_dot = max(dot, 0.0)
    combined_anchor_ratio = (
        primary_norm_sq + cap * projected_anchor_dot
    ) / primary_norm_sq
    return {
        "dot": dot,
        "primary_norm": math.sqrt(primary_norm_sq),
        "auxiliary_norm": math.sqrt(auxiliary_norm_sq),
        "cosine": cosine,
        "conflict": conflict,
        "projection_coefficient": coefficient,
        "projected_auxiliary_norm": math.sqrt(projected_norm_sq),
        "auxiliary_cap": cap,
        "retained_auxiliary_ratio": retained_ratio,
        "combined_anchor_ratio": combined_anchor_ratio,
    }


def gradient_inner_products(primary_gradients, auxiliary_gradients):
    """Accumulate global dot products without flattening model gradients."""
    if len(primary_gradients) != len(auxiliary_gradients):
        raise ValueError("gradient tuples must be aligned")
    dot = primary_norm_sq = auxiliary_norm_sq = 0.0
    for primary, auxiliary in zip(primary_gradients, auxiliary_gradients):
        if primary is not None:
            primary_norm_sq += float(primary.detach().double().square().sum().item())
        if auxiliary is not None:
            auxiliary_norm_sq += float(auxiliary.detach().double().square().sum().item())
        if primary is not None and auxiliary is not None:
            dot += float((primary.detach().double() * auxiliary.detach().double()).sum().item())
    return dot, primary_norm_sq, auxiliary_norm_sq


def correction_projection_statistics(
    primary_norm_sq,
    corrected_norm_sq,
    primary_corrected_dot,
    eps=1e-20,
):
    """Audit a corrected objective as ``g_primary + g_correction``.

    The correction is recovered from scalar products without materializing a
    third model-sized gradient tuple.  It is projected onto the half-space
    ``g_primary^T g_correction >= 0`` and norm-capped exactly as in
    :func:`combine_anchored_gradients`.
    """
    primary_norm_sq = float(primary_norm_sq)
    corrected_norm_sq = float(corrected_norm_sq)
    primary_corrected_dot = float(primary_corrected_dot)
    correction_dot = primary_corrected_dot - primary_norm_sq
    correction_norm_sq = max(
        corrected_norm_sq + primary_norm_sq - 2.0 * primary_corrected_dot,
        0.0,
    )
    if primary_norm_sq <= eps:
        raise ValueError("primary gradient must be non-zero")
    if correction_norm_sq <= eps:
        corrected_cosine = primary_corrected_dot / math.sqrt(
            max(primary_norm_sq * corrected_norm_sq, eps)
        )
        return {
            "dot": correction_dot,
            "primary_norm": math.sqrt(primary_norm_sq),
            "auxiliary_norm": 0.0,
            "cosine": 0.0,
            "conflict": False,
            "projection_coefficient": 0.0,
            "projected_auxiliary_norm": 0.0,
            "auxiliary_cap": 1.0,
            "retained_auxiliary_ratio": 0.0,
            "combined_anchor_ratio": 1.0,
            "primary_corrected_dot": primary_corrected_dot,
            "corrected_norm": math.sqrt(max(corrected_norm_sq, 0.0)),
            "corrected_primary_cosine": corrected_cosine,
            "corrected_anchor_ratio": primary_corrected_dot / primary_norm_sq,
            "safe_combined_norm": math.sqrt(primary_norm_sq),
            "safe_primary_cosine": 1.0,
        }
    stats = projection_statistics(
        correction_dot,
        primary_norm_sq,
        correction_norm_sq,
        eps=eps,
    )
    cap = stats["auxiliary_cap"]
    projected_dot = max(correction_dot, 0.0)
    projected_norm_sq = stats["projected_auxiliary_norm"] ** 2
    combined_norm_sq = (
        primary_norm_sq
        + cap * cap * projected_norm_sq
        + 2.0 * cap * projected_dot
    )
    stats.update({
        "primary_corrected_dot": primary_corrected_dot,
        "corrected_norm": math.sqrt(max(corrected_norm_sq, 0.0)),
        "corrected_primary_cosine": primary_corrected_dot / math.sqrt(
            max(primary_norm_sq * corrected_norm_sq, eps)
        ),
        "corrected_anchor_ratio": primary_corrected_dot / max(primary_norm_sq, eps),
        "safe_combined_norm": math.sqrt(max(combined_norm_sq, 0.0)),
        "safe_primary_cosine": (
            primary_norm_sq + cap * projected_dot
        ) / math.sqrt(max(primary_norm_sq * combined_norm_sq, eps)),
    })
    return stats


def combine_anchored_gradients(primary_gradients, auxiliary_gradients):
    """Return ``g_primary + capped(project(g_auxiliary))`` and audit statistics."""
    dot, primary_norm_sq, auxiliary_norm_sq = gradient_inner_products(
        primary_gradients, auxiliary_gradients
    )
    stats = projection_statistics(dot, primary_norm_sq, auxiliary_norm_sq)
    coefficient = stats["projection_coefficient"]
    cap = stats["auxiliary_cap"]
    combined = []
    for primary, auxiliary in zip(primary_gradients, auxiliary_gradients):
        if primary is None and auxiliary is None:
            combined.append(None)
            continue
        if primary is None:
            projected_auxiliary = auxiliary
            combined.append(cap * projected_auxiliary)
            continue
        if auxiliary is None:
            combined.append(primary)
            continue
        projected_auxiliary = auxiliary - coefficient * primary
        combined.append(primary + cap * projected_auxiliary)
    return tuple(combined), stats


def project_to_primary_progress(primary_gradients, preferred_gradients, eps=1e-20):
    """Project ``g_preferred`` onto ``<g,g_primary> >= ||g_primary||^2``."""
    dot, primary_norm_sq, preferred_norm_sq = gradient_inner_products(
        primary_gradients, preferred_gradients
    )
    if primary_norm_sq <= eps:
        combined = tuple(
            None if gradient is None else gradient.detach().clone()
            for gradient in preferred_gradients
        )
        return combined, {
            "constraint_active": False,
            "primary_degenerate": True,
            "coefficient": 0.0,
            "primary_norm": math.sqrt(max(primary_norm_sq, 0.0)),
            "preferred_norm": math.sqrt(max(preferred_norm_sq, 0.0)),
            "gradient_cosine": 0.0,
            "preferred_progress_ratio": float("nan"),
            "projected_progress_ratio": float("nan"),
            "projected_norm": math.sqrt(max(preferred_norm_sq, 0.0)),
        }

    coefficient = max((primary_norm_sq - dot) / primary_norm_sq, 0.0)
    combined = []
    for primary, preferred in zip(primary_gradients, preferred_gradients):
        if primary is None and preferred is None:
            combined.append(None)
        elif primary is None:
            combined.append(preferred.detach().clone())
        elif preferred is None:
            combined.append(coefficient * primary.detach())
        else:
            combined.append(preferred.detach() + coefficient * primary.detach())

    projected_dot = dot + coefficient * primary_norm_sq
    projected_norm_sq = (
        preferred_norm_sq
        + 2.0 * coefficient * dot
        + coefficient * coefficient * primary_norm_sq
    )
    denominator = math.sqrt(max(primary_norm_sq * preferred_norm_sq, eps))
    return tuple(combined), {
        "constraint_active": bool(coefficient > 0.0),
        "primary_degenerate": False,
        "coefficient": float(coefficient),
        "primary_norm": math.sqrt(primary_norm_sq),
        "preferred_norm": math.sqrt(max(preferred_norm_sq, 0.0)),
        "gradient_cosine": float(dot / denominator),
        "preferred_progress_ratio": float(dot / primary_norm_sq),
        "projected_progress_ratio": float(projected_dot / primary_norm_sq),
        "projected_norm": math.sqrt(max(projected_norm_sq, 0.0)),
    }


def accumulate_parameter_gradients(parameters, gradients, scale=1.0):
    """Add a detached gradient tuple to ``parameter.grad`` in place."""
    parameters = tuple(parameters)
    gradients = tuple(gradients)
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must be aligned")
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("gradient accumulation scale must be finite")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        value = gradient.detach().to(device=parameter.device, dtype=parameter.dtype)
        if scale != 1.0:
            value = value * scale
        if parameter.grad is None:
            parameter.grad = value.clone()
        else:
            parameter.grad.add_(value)
