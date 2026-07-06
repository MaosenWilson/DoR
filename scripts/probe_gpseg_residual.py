"""Offline sanity checks (no GPU) for GP-SegGRPO's segment residual (method.md Sec.3.4 v4).

Checks, on synthetic per-token errors:
  1. zero-mean: mean_k(resid[i,k]) == 0 for every candidate i (so the residual cannot
     shift the rollout-level learning signal -- it only redistributes within a rollout).
  2. K=1 degeneration: with a single segment the residual is identically 0.
  3. informativeness: with K=4 the residual is NOT identically 0 (it carries signal).

This is G2 in experiments.md -- must pass before GPU time. lambda=0 degeneration needs
no runtime check: the loss is implemented as L_global + lambda * L_residual, so lambda=0
reproduces the standard-path expression by construction.
"""
import torch

from dor.grpo import _zscore_cols, pool_seg_rms

K = 16
torch.manual_seed(0)

def residual(k_seg, tpf_per_seg=80):
    tpf = k_seg * tpf_per_seg
    seg_ids = torch.arange(tpf) // tpf_per_seg
    sq = torch.rand(K, tpf) * 0.01                    # synthetic per-token squared error
    r_seg = _zscore_cols(pool_seg_rms(sq, seg_ids, k_seg))
    a_seg = _zscore_cols(r_seg)
    resid = a_seg - a_seg.mean(dim=1, keepdim=True)
    return resid

r4 = residual(4)
r1 = residual(1, tpf_per_seg=320)

zm = r4.mean(dim=1).abs().max()
print(f"K=4: per-candidate residual mean, max|.| = {zm:.2e}  (want ~ 0)")
print(f"K=4: residual std = {r4.std():.4f}  (want > 0, carries signal)")
print(f"K=1: residual max|.| = {r1.abs().max():.2e}  (want exactly 0)")

ok = bool(zm < 1e-5 and r4.std() > 0.1 and r1.abs().max() == 0.0 and torch.isfinite(r4).all())
print("PROBE_GPSEG_RESIDUAL_OK" if ok else "PROBE_GPSEG_RESIDUAL_FAIL")
