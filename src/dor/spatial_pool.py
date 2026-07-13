"""Floor-aware spatial pooling for reconstruction-calibrated rewards (plan appendix S).

Pooling weights are the inverse (variance | magnitude) of the LOCAL reconstruction
floor, computed per window from the GT / reachable-target error map alone -- no
fitted parameters, frozen before training.  Uniform weights (constant floor map)
degenerate exactly to the existing mean-pooled RC metrics.
"""
import torch
import torch.nn.functional as F


POOL = 8          # block size; 256x320 -> 32x40 blocks (divides exactly: preserves the mean)
RATIO_CAP = 10.0  # max weight ratio inside one window (IQA pooling practice)


def block_pool(m, k=POOL):
    """Average-pool [H,W] or [K,H,W] into blocks; exact-mean preserving when k divides H,W."""
    if m.ndim == 2:
        return F.avg_pool2d(m[None, None], k)[0, 0]
    if m.ndim == 3:
        return F.avg_pool2d(m[:, None], k)[:, 0]
    raise ValueError(f"expected [H,W] or [K,H,W], got {tuple(m.shape)}")


def floor_weights(phi, eps_quantile, scheme):
    """Per-window pooling weights (sum 1) from a pooled floor map [h,w].

    scheme "iv": 1/(phi^2 + eps^2)  -- inverse-variance (Kendall & Gal form), primary;
    scheme "im": 1/(phi + eps)      -- inverse-magnitude, sensitivity variant.
    eps is the given quantile of phi, so a flat floor map degenerates to uniform weights.
    """
    phi = phi.clamp_min(0.0)
    eps = torch.quantile(phi.flatten(), float(eps_quantile)).clamp_min(1e-12)
    if scheme == "iv":
        w = 1.0 / (phi.square() + eps.square())
    elif scheme == "im":
        w = 1.0 / (phi + eps)
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    w = torch.maximum(w, w.max() / RATIO_CAP)
    return w / w.sum()


def weighted_pool(maps, w):
    """Candidate error maps [K,h,w] x normalized weights [h,w] -> weighted means [K]."""
    if maps.ndim != 3 or w.shape != maps.shape[1:]:
        raise ValueError(f"shape mismatch: maps {tuple(maps.shape)} vs weights {tuple(w.shape)}")
    return (maps * w).flatten(1).sum(1)
