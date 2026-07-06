"""Offline sanity check (no GPU) for codec_fused_reward v3's reliability weight
q = relu(corr(r_dec, r_tok))^2 (method.md Sec.3.2 v3). Checks the required extremes:

  seg A: decoded reward = scaled token signal + small noise  -> high corr -> q close to 1
  seg B: decoded reward = pure noise (floor-dominated)        -> corr ~ 0 -> q close to 0
  seg C: decoded reward anti-correlated (pathological)        -> relu clips -> q = 0

Also numerically compares q against the ground-truth Wiener weight
sigma_star^2/(sigma_star^2+sigma_eta^2) for seg A (they should approximately match --
that's the whole point of the corr^2 estimator).

Pure synthetic; CPU; <1s. This is A9e in experiments.md -- must pass before GPU time.
"""
import torch

from dor.grpo import _colcorr

K, k_seg = 16, 3
torch.manual_seed(0)

S = torch.randn(K, k_seg)                       # shared clean signal per segment
r_tok = S + 0.02 * torch.randn(K, k_seg)        # token reward ~ clean reference

sig_a, noise_a = 1.0, 0.3                       # seg A: strong signal, small floor jitter
sig_b, noise_b = 0.05, 1.0                      # seg B: floor jitter dominates

r_dec = torch.empty(K, k_seg)
r_dec[:, 0] = sig_a * S[:, 0] + noise_a * torch.randn(K)
r_dec[:, 1] = sig_b * S[:, 1] + noise_b * torch.randn(K)
r_dec[:, 2] = -S[:, 2] + 0.1 * torch.randn(K)   # seg C: anti-correlated

rho = _colcorr(r_dec, r_tok)
q = rho.clamp_min(0.0) ** 2

wiener_a = sig_a**2 / (sig_a**2 + noise_a**2)   # ground-truth Wiener weight for seg A

print(f"seg A (signal-dominated):  rho={rho[0]:+.3f}  q={q[0]:.3f}  (want ~ Wiener={wiener_a:.3f})")
print(f"seg B (noise-dominated):   rho={rho[1]:+.3f}  q={q[1]:.3f}  (want -> close to 0)")
print(f"seg C (anti-correlated):   rho={rho[2]:+.3f}  q={q[2]:.3f}  (want = 0 exactly)")

ok = (abs(q[0] - wiener_a) < 0.25 and q[1] < 0.3 and q[2] == 0.0
      and torch.isfinite(q).all() and (q >= 0).all() and (q <= 1).all())
print("PROBE_CODEC_FUSED_Q_OK" if ok else "PROBE_CODEC_FUSED_Q_FAIL")
