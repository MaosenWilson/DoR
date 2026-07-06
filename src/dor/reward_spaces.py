"""Reward-space sweep for the DoR AAAI-2027 experiments (story_spine §6, matrix §1).

This is the *space-sweep superset* of `dor.grpo.gt_reward`: the verifiable GT
distance D is computed in a selectable metric space, moving from "post-decode"
(pixel) to "pre-decode" (phi continuous features, code post-quant). Every arm
stays a pure GT-anchored reward (higher == closer to GT); intra-group consensus
shaping is unchanged and still lives in `dor.rewards`.

Arms (matrix Table §1):
  A0 pixel  -LPIPS(decode(cand), gt)                 RLVR-World baseline (post-decode)
  A1 mse    -MSE                                      pixel
  A2 ssim    SSIM                                     pixel, structural
  A3 floor  -(LPIPS - phi_tok)  (clamped >=0)         floor-calibrated pixel  *** key baseline ***
  A4 multi   mean z-score(-LPIPS, PSNR, SSIM)         ToolRL-style multi-component pixel
  A5 phi    -RMS(phi(cand) - phi(gt))                 pre-decode continuous encoder features
  A6 code   -RMS(codes(cand) - codes(gt))             pre-decode FSQ code space  == DoR
     hybrid  alpha*z(pixel) + (1-alpha)*z(code)       z-score fusion (kept for completeness)

Drop-in use in training: in `scripts/train_grpo.py` / `dor.grpo.train`, replace
    from dor.grpo import gt_reward
with
    from dor.reward_spaces import gt_reward
and pass `kind` from the extended REWARDS tuple. The pixel/code/hybrid arms are
bit-for-bit identical to the originals, so this never breaks the existing runs.

DINOv2 is deliberately NOT a reward arm here -- it is reserved as the held-out
"true-quality" reference for the rank-preservation analysis (story_spine §7).
"""
import json
import os

import numpy as np
import torch

from dor.constants import ROOT
from dor.grpo import code_rms, code_vec
from dor.rewards import _zscore
from dor.tokenization import decode_tokens, encode_feature_map, encode_indices

ARMS = ("pixel", "mse", "ssim", "floor", "floorpc", "multi", "phi", "code", "hybrid",
        "a0faithful", "dorw", "pixel_tok", "ssim_tok", "mse_tok", "hybrid_tok",
        "code_dyn", "pixel_tok_dyn")

_WEIGHTS = None


def load_weights(path=None):
    """Lazy-load the floor-aware reward weights (Phase 0 output, configs/aaai2027/
    reward_weights.json). Cached; only touched when the 'dorw' arm runs."""
    global _WEIGHTS
    if _WEIGHTS is None:
        cands = [path, f"{ROOT}/configs/aaai2027/reward_weights.json",
                 os.path.join(os.path.dirname(__file__), "../../configs/aaai2027/reward_weights.json")]
        for p in cands:
            if p and os.path.exists(p):
                with open(p) as f:
                    _WEIGHTS = json.load(f)
                break
        if _WEIGHTS is None:
            raise FileNotFoundError("reward_weights.json not found; run scripts/compute_reward_weights.py first")
    return _WEIGHTS


def phi_rms(tok, imgs, gt):
    """Pre-decode continuous-feature RMS, per candidate [K] np (no decode path).

    imgs [K,3,256,320], gt [3,256,320] in [0,1]. Uses the FULL spatial feature map
    (same granularity as code_rms) so A5(phi) vs A6(code) isolates quantization.
    """
    pc = encode_feature_map(tok, imgs)                   # [K, C'*h*w]
    gc = encode_feature_map(tok, gt.unsqueeze(0))        # [1, C'*h*w]
    return (pc - gc).pow(2).mean(dim=1).sqrt().detach().cpu().numpy()


def code_delta_reward(tok, cand, gt_idx, cur_idx, *, gamma=0.25, tau=0.0):
    """Pre-decode motion-residual alignment in FSQ code space.

    We use direction + magnitude ratio rather than ||(z_i-z_t)-(z'-z_t)||, because the
    latter algebraically collapses to ||z_i-z'|| and adds no new motion signal.
    """
    if cur_idx is None:
        raise ValueError("dynamic reward needs cur_idx (encoded current/context-last frame)")
    if cur_idx.ndim == 2:
        cur_idx = cur_idx.unsqueeze(0)
    zc = code_vec(tok, cand.reshape(cand.shape[0], *gt_idx.shape[-2:]))
    zg = code_vec(tok, gt_idx)
    zt = code_vec(tok, cur_idx)
    dc = zc - zt
    dg = zg - zt
    eps = 1e-6
    nc = dc.norm(dim=1)
    ng = dg.norm(dim=1).clamp_min(eps)  # [1]
    if float(ng.item()) <= float(tau):
        return np.zeros(cand.shape[0], dtype=float)
    cos = (dc * dg).sum(dim=1) / (nc.clamp_min(eps) * ng)
    mag = (torch.log((nc + eps) / (ng + eps))).abs()
    return (cos - float(gamma) * mag).detach().cpu().numpy()


def gt_reward(kind, metrics, tok, cand, imgs, gt, gt_idx, *, alpha=0.5, phi_tok=0.0,
              weight_temp=1.0, cur_idx=None, dyn_lambda=0.25, dyn_gamma=0.25,
              dyn_tau=0.0):
    """Verifiable GT reward r_gt [K] (higher == closer to GT), never consensus-shaped.

    Args mirror `dor.grpo.gt_reward` plus:
      phi_tok: scalar reward-noise-floor for the constant 'floor' arm.
      weight_temp: temperature on the floor-aware weights w_m^tau for the 'dorw' arm
                   (tau=0 -> equal weight; tau=1 -> designed; large -> hard gating).
    """
    if kind in ("code", "code_dyn"):
        r_code = -code_rms(tok, cand, gt_idx)
        if kind == "code":
            return r_code
        r_dyn = code_delta_reward(tok, cand, gt_idx, cur_idx, gamma=dyn_gamma, tau=dyn_tau)
        return _zscore(r_code) + float(dyn_lambda) * _zscore(r_dyn)
    if kind == "phi":
        return -phi_rms(tok, imgs, gt)
    if kind in ("pixel_tok", "pixel_tok_dyn", "ssim_tok", "mse_tok"):
        # FLOOR-CANCELLED reward: compare to the ACHIEVABLE target decode(encode(gt)),
        # not raw gt. The decoder's systematic floor is shared by decode(cand) and
        # decode(encode(gt)) -> cancels, leaving the content/dynamics difference. The
        # token-optimal candidate gets ~0 (floor removed), unlike vs-raw-gt.
        # mse_tok probes whether the cancellation gain scales with the metric's floor
        # (MSE has a small floor -> expect a small gain; LPIPS large -> large gain).
        rec_gt = decode_tokens(tok, encode_indices(tok, gt.unsqueeze(0)).reshape(1, -1))[0]  # [3,H,W]
        qt = metrics.eval_batch(imgs, rec_gt)
        if kind in ("pixel_tok", "pixel_tok_dyn"):
            r_pix = -np.asarray(qt["lpips"], float)
            if kind == "pixel_tok":
                return r_pix
            r_dyn = code_delta_reward(tok, cand, gt_idx, cur_idx, gamma=dyn_gamma, tau=dyn_tau)
            return _zscore(r_pix) + float(dyn_lambda) * _zscore(r_dyn)
        if kind == "mse_tok":
            return -qt["mse"]
        if "ssim" not in qt:
            raise RuntimeError("ssim_tok needs piqa installed")
        return np.asarray(qt["ssim"], float)
    if kind == "hybrid_tok":
        # z-fusion of two LOW-floor signals: pre-decode code (L2 dynamics) + floor-cancelled
        # perceptual (decode(cand) vs decode(encode(gt))). Both clean -> fusion can help.
        rec_gt = decode_tokens(tok, encode_indices(tok, gt.unsqueeze(0)).reshape(1, -1))[0]
        r_perc = -np.asarray(metrics.eval_batch(imgs, rec_gt)["lpips"], float)
        r_cod = -code_rms(tok, cand, gt_idx)
        return alpha * _zscore(r_perc) + (1.0 - alpha) * _zscore(r_cod)

    # arms that need pixel-space metrics (compute once)
    q = metrics.eval_batch(imgs, gt)
    if kind == "a0faithful":
        # faithful RLVR-World reward: -(MSE + LPIPS), equal weight, post-decode
        return -(np.asarray(q["mse"], float) + np.asarray(q["lpips"], float))
    if kind == "dorw":
        # floor-aware multi-space reward: R_i = -sum_m w_m^tau * d_m/s_m  (renormalised)
        W = load_weights()["components"]
        comps = {"code": np.asarray(code_rms(tok, cand, gt_idx), float),
                 "recon": np.asarray(q["mse"], float),
                 "perc": np.asarray(q["lpips"], float)}
        wt = {c: W[c]["w"] ** weight_temp for c in comps}
        tot = sum(wt.values()) + 1e-12
        R = sum((wt[c] / tot) * (comps[c] / W[c]["s"]) for c in comps)
        return -np.asarray(R, float)
    if kind == "pixel":
        return -q["lpips"]
    if kind == "mse":
        return -q["mse"]
    if kind == "ssim":
        if "ssim" not in q:
            raise RuntimeError("ssim arm needs piqa installed (pip install piqa)")
        return np.asarray(q["ssim"], float)
    if kind == "floor":
        # floor-calibrated pixel: subtract a CONSTANT reward-noise floor, clamp, negate.
        # NB: a within-group-constant offset is rank-invariant (Spearman unchanged), so
        # in GRPO this arm collapses to A0 by construction -- kept only as the theory demo.
        return -np.maximum(np.asarray(q["lpips"], float) - float(phi_tok), 0.0)
    if kind == "floorpc":
        # honest floor-calibrated pixel (A3): subtract a PER-CANDIDATE floor = each
        # candidate's own decode round-trip error LPIPS(decode(encode(x_i)), x_i). This
        # varies within the group, so it can actually re-rank -- not a strawman.
        rec_idx = encode_indices(tok, imgs).reshape(imgs.shape[0], -1)  # [K,320]
        rec = decode_tokens(tok, rec_idx)                               # [K,3,256,320]
        phi_i = np.asarray(metrics.eval_batch(rec, imgs)["lpips"], float)  # [K] per-cand floor
        return -np.maximum(np.asarray(q["lpips"], float) - phi_i, 0.0)
    if kind == "multi":
        comps = [_zscore(-q["lpips"]), _zscore(q["psnr"])]
        if "ssim" in q:
            comps.append(_zscore(q["ssim"]))
        return np.mean(comps, axis=0)
    if kind == "hybrid":
        r_pix = -q["lpips"]
        r_cod = -code_rms(tok, cand, gt_idx)
        return alpha * _zscore(r_pix) + (1.0 - alpha) * _zscore(r_cod)
    raise ValueError(f"unknown reward arm: {kind!r} (expected one of {ARMS})")
