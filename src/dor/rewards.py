"""Reward and advantage shaping for GRPO.

The reward stays a pure RLVR-World verifiable GT signal (negative LPIPS vs the
ground-truth next frame). Intra-group consensus only *shapes the advantage*:

  gt_only      A = z(r_gt)                                  RLVR-World baseline
  hybrid_add   A = z( z(r_gt) + lam * z(c) )                additive reward shaping  (ablation arm A)
  hybrid_mult  A = z( z(r_gt) * max(delta, 1 + beta*z(c)) ) advantage modulation     (main method B)

Why (B) multiplies the consensus weight onto the GT-signed advantage:
  * Collective-hallucination guard: with w = max(delta, 1 + beta*z(c)) > 0 the
    sign of z(r_gt) is preserved, so a high-consensus-but-GT-bad candidate cannot
    be flipped into a positive update.
  * Graceful degradation: beta = 0  =>  w = 1  =>  exactly gt_only.
The reward r_gt is never modified, so the verifiable objective is preserved.
"""
import numpy as np

MODES = ("gt_only", "hybrid_add", "hybrid_mult", "drgrpo", "rankrel")


def _zscore(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def rank_reliability_weights(r, sigma_eta, eps=1e-12):
    """Candidate-level confidence that local reward ranks survive floor noise.

    For two candidates with observed reward gap |r_i-r_j| and independent Gaussian
    reward noise of std sigma_eta per candidate, the confidence that their ordering is
    not a coin flip is

        2*Phi(|gap|/(sqrt(2)*sigma_eta)) - 1 = erf(|gap|/(2*sigma_eta)).

    We assign each candidate the confidence of its nearest sorted neighbour, i.e. the
    weakest adjacent ordering that can flip its local rank. K is tiny (16), but this
    adjacent version avoids over-penalising top/bottom samples for far-away pairs.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.size <= 1 or sigma_eta is None or sigma_eta <= eps:
        return np.ones_like(r, dtype=np.float64)

    order = np.argsort(r)
    rs = r[order]
    gaps = np.full(r.size, np.inf, dtype=np.float64)
    if r.size > 1:
        left = np.r_[np.inf, np.abs(np.diff(rs))]
        right = np.r_[np.abs(np.diff(rs)), np.inf]
        gaps_sorted = np.minimum(left, right)
        gaps[order] = gaps_sorted

    # Vectorised erf via numpy if available; fallback keeps the code portable.
    try:
        conf = np.erf(gaps / (2.0 * float(sigma_eta)))
    except AttributeError:
        import math
        conf = np.array([math.erf(float(g) / (2.0 * float(sigma_eta))) for g in gaps])
    return np.clip(conf, 0.0, 1.0)


def static_copy_gate(motion, copy_sim, tau_motion=0.5, tau_copy=0.10):
    """1.0 keep / 0.0 gate. Under non-trivial action, gate near-static-copy candidates.

    copy_sim [K] = LPIPS(candidate, current frame); small => barely changed.
    """
    copy_sim = np.asarray(copy_sim, float)
    if motion <= tau_motion:
        return np.ones_like(copy_sim)
    return np.where(copy_sim < tau_copy, 0.0, 1.0)


def shape_advantage(r_gt, consensus=None, mode="gt_only", *, lam=0.5, beta=0.5,
                    delta=0.1, motion=None, copy_sim=None, gate_penalty=2.0,
                    tau_motion=0.5, tau_copy=0.10):
    """Compute the group-normalised GRPO advantage. r_gt is GT-based and never modified.

    Args:
      r_gt:      [K] GT reward (e.g. -LPIPS vs GT next frame).
      consensus: [K] transition-feature consensus support (required for hybrid_*).
      mode:      one of MODES.
      motion, copy_sim: enable the static-copy gate when both are provided.
    Returns:
      (advantage [K] np.float64, info dict).
    """
    r_gt = np.asarray(r_gt, float)
    if mode == "drgrpo":
        # Dr.GRPO: subtract group mean only, NO /std. Floor-dominated groups (whose std is
        # noise) are then not amplified to unit variance. (advisor_meeting_20260627.md §6-补3)
        adv = r_gt - r_gt.mean()
        return adv, {"adv": adv}
    zr = _zscore(r_gt)
    info = {"zr": zr}

    if mode in ("gt_only", "rankrel"):
        shaped = zr
    elif mode in ("hybrid_add", "hybrid_mult"):
        if consensus is None:
            raise ValueError(f"mode {mode!r} requires `consensus`")
        zc = _zscore(consensus)
        info["zc"] = zc
        if mode == "hybrid_add":
            shaped = zr + lam * zc
        else:  # hybrid_mult (B)
            w = np.maximum(delta, 1.0 + beta * zc)
            info["w"] = w
            shaped = zr * w
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")

    if motion is not None and copy_sim is not None:
        gate = static_copy_gate(motion, copy_sim, tau_motion, tau_copy)
        shaped = shaped - gate_penalty * (1.0 - gate)
        info["gate"] = gate

    adv = _zscore(shaped)  # group-normalise so step scale is comparable across modes
    info["adv"] = adv
    return adv, info
