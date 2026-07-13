"""Lean GRPO training/eval loop (bypasses verl/vllm; runs on Blackwell sm_120).

The verifiable GT distance D is selectable via `reward`:
  pixel   -LPIPS(decode(cand), gt)                RLVR-World baseline
  code    -RMS(codes(cand) - codes(gt))           FSQ code space, no decode
  hybrid  alpha*z(pixel) + (1-alpha)*z(code)       z-score fusion
Intra-group consensus only reshapes the advantage (see dor.rewards.MODES); the
GT reward r_gt itself is never modified.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dor.consensus import consensus_support, motion_magnitude
from dor.constants import CTX, GRID, TPF
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model


def set_determinism(seed):
    """Make one training run bitwise-reproducible for a given seed (opt-in).

    All explicit seeds (data windows, per-step generation, eval) are already fixed,
    yet two runs with the *same* seed drift apart over many steps because the GPU's
    floating-point reductions are nondeterministic by default; the error compounds
    into materially different policies (observed: same-seed flow varies by ~0.03-0.08).
    This pins down cuBLAS / cuDNN / SDPA so same seed -> same result. Slower (forces the
    math SDP backend) -- enable only for final paper runs, not exploration.
    """
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # NB: we deliberately do NOT force the math SDP backend here -- it materialises the
    # full attention matrix and OOMs at K=16 on 32 GB. The dominant same-seed drift is
    # cuBLAS GEMM + cuDNN (pinned above), not attention. warn_only keeps flash/mem-eff
    # SDPA (which lacks a deterministic kernel) running instead of crashing. If a smoke
    # test still shows drift, fall back to math SDPA only around the backpropped forward.
    torch.use_deterministic_algorithms(True, warn_only=True)


from dor.rewards import _zscore, rank_reliability_weights, shape_advantage
from dor.tokenization import build_prompt, decode_tokens, encode_features, encode_indices


def _hms(sec):
    """seconds -> H:MM:SS for progress / ETA display."""
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _bar(frac, width=20):
    """ASCII progress bar, log/tee-friendly (no carriage return)."""
    frac = min(max(frac, 0.0), 1.0)
    n = int(frac * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {frac * 100:3.0f}%"


_RAFT = None


def _get_raft(device):
    """Lazy frozen RAFT-small for optical-flow dynamics fidelity (weights downloaded once)."""
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        _RAFT = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
        for p in _RAFT.parameters():
            p.requires_grad_(False)
    return _RAFT


@torch.no_grad()
def _flow(raft, a, b, size=(128, 160)):
    """Optical flow a->b. a,b [N,3,H,W] in [0,1] -> flow [N,2,h,w] (downsampled for speed)."""
    a = F.interpolate(a, size, mode="bilinear", align_corners=False) * 2 - 1
    b = F.interpolate(b, size, mode="bilinear", align_corners=False) * 2 - 1
    return raft(a.contiguous(), b.contiguous())[-1]


@torch.no_grad()
def flow_fidelity(raft, cur, preds, gt):
    """Motion-weighted cosine between flow(cur->pred_i) and flow(cur->gt). Higher == predicted
    motion matches true motion. preds [K,3,H,W], cur/gt [3,H,W] -> [K] np."""
    K = preds.shape[0]
    fg = _flow(raft, cur.unsqueeze(0), gt.unsqueeze(0))               # [1,2,h,w]
    fp = _flow(raft, cur.unsqueeze(0).expand(K, -1, -1, -1), preds)   # [K,2,h,w]
    w = fg.norm(dim=1)                                                # [1,h,w] gt motion magnitude
    cos = F.cosine_similarity(fp, fg, dim=1)                          # [K,h,w]
    return ((cos * w).sum((-1, -2)) / (w.sum((-1, -2)) + 1e-6)).detach().cpu().numpy()


_EVAL = None


def _get_evaluator(device):
    """RLVR-World's exact frame-metric Evaluator (MAE/MSE/PSNR/SSIM/LPIPS-vgg) for
    one-to-one comparability with the published numbers. Lazy; i3d_path=None -> no FVD/FID."""
    global _EVAL
    if _EVAL is None:
        from ivideogpt.utils.video_metric import Evaluator
        _EVAL = Evaluator(i3d_path=None).to(device).eval()
    return _EVAL


def code_vec(tok, idx):
    """idx [B,16,20] long -> flat post-quant FSQ code [B, 16*20*5] float (no decode)."""
    return tok.indices_to_codes(idx).float().reshape(idx.shape[0], -1)


def code_rms(tok, cand, gt_idx):
    """cand [K,TPF] long, gt_idx [1,16,20] -> per-candidate code-space RMS [K] np."""
    cc = code_vec(tok, cand.reshape(cand.shape[0], GRID[0], GRID[1]))
    gc = code_vec(tok, gt_idx)
    return (cc - gc).pow(2).mean(1).sqrt().detach().cpu().numpy()


def build_seg_ids(gh, gw, device):
    """Map each of the TPF tokens to one of gh*gw spatial segments over the 16x20 grid.

    Returns LongTensor [TPF]; value in [0, gh*gw). Token t sits at (t//20, t%20)."""
    if GRID[0] % gh or GRID[1] % gw:
        raise ValueError(f"grid {GRID} not divisible by {gh}x{gw}")
    rows = torch.arange(GRID[0], device=device) // (GRID[0] // gh)   # [16]
    cols = torch.arange(GRID[1], device=device) // (GRID[1] // gw)   # [20]
    return (rows[:, None] * gw + cols[None, :]).reshape(-1).long()   # [TPF]


def pertoken_code_sq(tok, cand, gt_idx):
    """Pre-decode per-token FSQ code squared error [K, TPF] (mean over 5 code channels)."""
    K = cand.shape[0]
    cc = tok.indices_to_codes(cand.reshape(K, GRID[0], GRID[1])).float()  # [K,5,16,20]
    gc = tok.indices_to_codes(gt_idx).float()                            # [1,5,16,20]
    return ((cc - gc) ** 2).mean(1).reshape(K, -1)                       # [K, TPF]


def pertoken_pix_sq(pred, target):
    """Post-decode per-token patch MSE [B, TPF]: each token = a 16x16 image patch."""
    ph, pw = 256 // GRID[0], 320 // GRID[1]  # 16, 16
    def _patch(x):
        return x.reshape(x.shape[0], 3, GRID[0], ph, GRID[1], pw)
    d = (_patch(pred) - _patch(target)) ** 2                            # [B,3,16,ph,20,pw]
    return d.mean(dim=(1, 3, 5)).reshape(pred.shape[0], -1)             # [B, TPF]


def seg_floor_norm(floor_sq, seg_ids, k_seg):
    """Per-segment reconstruction floor (normalised ~1) from per-token floor MSE [TPF].
    Used for the codec-aware advantage denominator gamma*b_tilde in seg_advantage_from_reward.
    (codec_fused_reward's own reliability weight q uses the RAW, un-normalised floor instead --
    see seg_floor_raw -- for unit consistency with the observed decoded-reward variance.)"""
    tot = torch.zeros(k_seg, device=floor_sq.device)
    cnt = torch.zeros(k_seg, device=floor_sq.device)
    tot.index_add_(0, seg_ids, floor_sq)
    cnt.index_add_(0, seg_ids, torch.ones(seg_ids.shape[0], device=floor_sq.device))
    b = (tot / cnt.clamp_min(1.0)).sqrt()
    return b / (b.mean() + 1e-9)


def seg_floor_raw(floor_sq, seg_ids, k_seg):
    """Per-segment reconstruction floor in RAW (un-normalised) units -- needed for the
    residual-clamp inside `codec_fused_reward`'s r_dec, which subtracts the floor from a
    same-unit raw distance (unlike seg_floor_norm's ~1-mean version used for q/gamma)."""
    tot = torch.zeros(k_seg, device=floor_sq.device)
    cnt = torch.zeros(k_seg, device=floor_sq.device)
    tot.index_add_(0, seg_ids, floor_sq)
    cnt.index_add_(0, seg_ids, torch.ones(seg_ids.shape[0], device=floor_sq.device))
    return (tot / cnt.clamp_min(1.0)).sqrt()


def _colcorr(a, b, eps=1e-6):
    """Per-column Pearson correlation across dim 0. a,b [K, k_seg] -> [k_seg]."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + eps)


def codec_fused_reward(pix_sq, code_sq, floor_sq_raw, seg_ids, k_seg,
                       dyn_seg=None, dyn_lambda=0.0, eps=1e-6):
    """CAST-GRPO's canonical segment reward (method.md Sec.3.2, v3, 2026-07-02):
    reliability-weighted fusion of a decoder-free token-space reward and a decoded reward,
    where reliability is the within-group AGREEMENT with the floor-free reference:

      r_tok[i,k] = -RMS(code_sq) per segment       (pre-decode, floor ~ 0, clean reference)
      r_dec[i,k] = -RMS(pix_sq) per segment        (post-decode; NO explicit floor subtract /
                                                    clamp -- the constant floor offset is
                                                    absorbed by the per-segment z-score, and
                                                    the old clamp created zero-variance
                                                    segments whose z-score amplified noise)
      rho[k]     = corr_i(r_dec[:,k], r_tok[:,k])  (within-group, per segment)
      q[k]       = max(0, rho[k])**2               (NO free hyperparameter)
      r_seg[i,k] = (1-q[k]) z(r_tok) + q[k] z(r_dec) + dyn_lambda * z(dyn_seg)

    Why q = rho^2 is the Wiener weight: with r_dec = a*S + eta and r_tok ~ S (clean),
    corr(r_dec, S)^2 = sigma_star^2/(sigma_star^2 + sigma_eta^2) -- the inverse-variance
    weight with the noise level estimated IMPLICITLY from candidate-level fluctuation.
    This replaces two failed conventions: v1's exp(-alpha*b) heuristic and v2's variance
    decomposition using the floor MEAN b as noise SD. The mean floor is a within-group
    constant that group normalisation cancels (C2's own point); the rank-corrupting noise
    is the candidate-level dispersion sigma_eta << b, so v2 overestimated the noise by
    orders of magnitude and measured q was identically 0 on real data at every granularity
    (the same over-conservatism that once collapsed the dorw arm to code). rho here is the
    per-segment online version of the SAME rho in C2's flip-rate bound arccos(rho)/pi:
    the paper's diagnostic quantity becomes the method's weight.

    dyn_seg [K,k_seg] (optional): per-segment code-space motion residual (seg_dyn_reward),
    added with the Phase-C proven weight dyn_lambda=0.10 -- v2's pilot showed dropping the
    motion term costs most of the dynamics edge (dmot 0.1514 vs legacy 0.1828).

    Returns (r_seg [K,k_seg] higher=better, floor_seg_norm [k_seg] for the advantage's
    separate gamma*b_tilde damping in seg_advantage_from_reward)."""
    r_tok = pool_seg_rms(code_sq, seg_ids, k_seg)                      # [K,k_seg]
    r_dec = pool_seg_rms(pix_sq, seg_ids, k_seg)                        # [K,k_seg] (-RMS, higher=better)
    rho = _colcorr(r_dec, r_tok, eps)                                   # [k_seg]
    q = rho.clamp_min(0.0) ** 2                                         # [k_seg], in [0,1], no free params
    r_seg = (1.0 - q[None, :]) * _zscore_cols(r_tok, eps) + q[None, :] * _zscore_cols(r_dec, eps)
    if dyn_seg is not None and dyn_lambda:
        r_seg = r_seg + float(dyn_lambda) * _zscore_cols(dyn_seg, eps)
    b_raw = seg_floor_raw(floor_sq_raw, seg_ids, k_seg)                 # [k_seg]
    b_tilde = b_raw / (b_raw.mean() + eps)                              # gamma damping only
    return r_seg, b_tilde


def pool_seg_rms(sq, seg_ids, k_seg):
    """Per-token squared error [K,TPF] -> per-segment reward -RMS(err), higher=better [K,k_seg]."""
    K = sq.shape[0]
    tot = torch.zeros(K, k_seg, device=sq.device)
    cnt = torch.zeros(k_seg, device=sq.device)
    tot.index_add_(1, seg_ids, sq)
    cnt.index_add_(0, seg_ids, torch.ones(seg_ids.shape[0], device=sq.device))
    return -(tot / cnt.clamp_min(1.0)).sqrt()


def _zscore_cols(x, eps=1e-6):
    """Column-wise (per segment, across the K candidates) z-score. x: [K, k_seg]."""
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, unbiased=False, keepdim=True)
    return (x - mu) / (sd + eps)


def pertoken_code_vecs(tok, cand, gt_idx, cur_idx):
    """Per-token FSQ code vectors (not squared error) for motion-residual decomposition.

    Returns (z_cand [K,TPF,5], z_gt [1,TPF,5], z_cur [1,TPF,5])."""
    def codes(idx):  # idx [B,16,20] long -> [B,TPF,5] float (chan-last)
        c = tok.indices_to_codes(idx).float()          # [B,5,16,20] (chan-first)
        return c.permute(0, 2, 3, 1).reshape(c.shape[0], -1, c.shape[1])
    K = cand.shape[0]
    return (codes(cand.reshape(K, GRID[0], GRID[1])), codes(gt_idx), codes(cur_idx))


def seg_dyn_reward(z_cand, z_gt, z_cur, seg_ids, k_seg, gamma, eps=1e-6):
    """Per-segment code-space motion-residual reward (direction + magnitude), higher=better.

    Mirrors `reward_spaces.code_delta_reward` (global cos(Δz_i,Δz') - gamma*|log(|Δz_i|/|Δz'|)|)
    but restricted to each segment's local motion sub-vector, so it captures WHERE the
    predicted motion agrees with the true motion, not just an aggregate whole-frame score.
    z_cand [K,TPF,C], z_gt/z_cur [1,TPF,C] -> [K,k_seg]."""
    dz_i = z_cand - z_cur                                                # [K,TPF,C]
    dz_g = (z_gt - z_cur).expand_as(dz_i[:1])                            # [1,TPF,C]
    order = torch.argsort(seg_ids, stable=True)
    K, TPF, C = dz_i.shape
    per_seg = TPF // k_seg
    dz_i_seg = dz_i[:, order, :].reshape(K, k_seg, per_seg * C)
    dz_g_seg = dz_g[:, order, :].reshape(1, k_seg, per_seg * C).expand(K, -1, -1)
    cos = F.cosine_similarity(dz_i_seg, dz_g_seg, dim=-1, eps=eps)
    mag_i = dz_i_seg.norm(dim=-1) + eps
    mag_g = dz_g_seg.norm(dim=-1).clamp_min(eps)
    return cos - float(gamma) * (mag_i / mag_g).log().abs()             # [K,k_seg]


@torch.no_grad()
def seg_advantage_from_reward(r_seg, seg_ids, lam, gamma, floor_seg=None, eps=1e-6):
    """Segment-level GRPO advantage from an already-pooled per-segment reward (higher=better).

      A_seg[i,k] = (r[i,k]-mean_i) / (std_i + gamma*b_k + eps)
      A_glob[i]  = z(mean_k r[i,k])
      A[i,k]     = lam*A_seg[i,k] + (1-lam)*A_glob[i]

    lam=0 (or k_seg=1) reduces to the rollout-level baseline (no spatial credit).
    Returns (adv_tok [K,TPF], reward_mean float)."""
    mu = r_seg.mean(0, keepdim=True)
    sd = r_seg.std(0, unbiased=False, keepdim=True)
    damp = gamma if floor_seg is None else gamma * floor_seg[None, :]
    a_seg = (r_seg - mu) / (sd + damp + eps)
    Ri = r_seg.mean(1)
    a_glob = (Ri - Ri.mean()) / (Ri.std(unbiased=False) + eps)
    A = lam * a_seg + (1.0 - lam) * a_glob[:, None]                     # [K,k_seg]
    return A[:, seg_ids], float(r_seg.mean())                          # [K,TPF]


@torch.no_grad()
def seg_advantage(sq, seg_ids, k_seg, lam, gamma, floor_seg=None, eps=1e-6):
    """Spatial segment-level GRPO advantage from a per-token error map (thin wrapper:
    pool_seg_rms + seg_advantage_from_reward). See seg_advantage_from_reward for the math."""
    r_seg = pool_seg_rms(sq, seg_ids, k_seg)
    return seg_advantage_from_reward(r_seg, seg_ids, lam, gamma, floor_seg, eps)


def gt_reward(kind, metrics, tok, cand, imgs, gt, gt_idx, alpha=0.5):
    """Verifiable GT reward r_gt [K] (higher=closer to GT). Never consensus-shaped."""
    if kind == "pixel":
        return -metrics.eval_batch(imgs, gt)["lpips"]
    if kind == "code":
        return -code_rms(tok, cand, gt_idx)
    if kind == "hybrid":
        r_pix = -metrics.eval_batch(imgs, gt)["lpips"]
        r_cod = -code_rms(tok, cand, gt_idx)
        return alpha * _zscore(r_pix) + (1.0 - alpha) * _zscore(r_cod)
    raise ValueError(f"unknown reward kind: {kind!r}")


def seq_logp(model, prompt, gen, autocast=True):
    """prompt [P], gen [K, TPF] -> (per-seq summed logprob [K], per-token logprob [K, TPF])."""
    K = gen.shape[0]
    full = torch.cat([prompt.unsqueeze(0).expand(K, -1), gen], dim=1)  # [K, P+TPF]
    P = prompt.shape[0]
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else torch.enable_grad()
    with ctx:
        logits = model(full).logits
    pred = logits[:, P - 1:P - 1 + TPF, :].float()
    logp = F.log_softmax(pred, dim=-1)
    tok_logp = logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1)  # [K, TPF]
    return tok_logp.sum(dim=1), tok_logp


def rank_label_loss(tok_logp, rewards, *, pos_frac=0.25, neg_frac=0.25,
                    tau=0.5, min_gap=0.0):
    """REAL-style rank-label objective for one rollout group.

    `rewards` are continuous verifiable scores. We only use them to define
    high-confidence top/bottom labels, then optimize a classification loss on
    the length-normalized relative log-probability. The old-policy logprob is
    the stopped current logprob because this lean harness samples and updates
    on-policy in one pass, matching the zero-valued-but-nonzero-gradient REAL
    construction.
    """
    r = np.asarray(rewards, dtype=np.float64)
    K = int(r.shape[0])
    if K < 2 or not np.isfinite(r).all():
        return None, {"valid": 0.0, "gap": 0.0, "pos_frac": 0.0}
    gap = float(np.max(r) - np.min(r))
    if gap < float(min_gap):
        return None, {"valid": 0.0, "gap": gap, "pos_frac": 0.0}

    n_pos = max(1, int(round(K * float(pos_frac))))
    n_neg = max(1, int(round(K * float(neg_frac))))
    if n_pos + n_neg > K:
        n_pos = max(1, K // 2)
        n_neg = max(1, K - n_pos)

    order = np.argsort(r)
    neg_idx = order[:n_neg]
    pos_idx = order[-n_pos:]
    pos = torch.zeros(K, device=tok_logp.device, dtype=torch.bool)
    neg = torch.zeros(K, device=tok_logp.device, dtype=torch.bool)
    pos[torch.as_tensor(pos_idx, device=tok_logp.device)] = True
    neg[torch.as_tensor(neg_idx, device=tok_logp.device)] = True

    tau = max(float(tau), 1e-6)
    rel = (tok_logp - tok_logp.detach()).mean(dim=1) / tau
    zero = torch.zeros(1, device=tok_logp.device, dtype=tok_logp.dtype)
    neg_loss = torch.logsumexp(torch.cat([rel[neg], zero], dim=0), dim=0)
    pos_loss = torch.logsumexp(torch.cat([-rel[pos], zero], dim=0), dim=0)
    loss = neg_loss + pos_loss
    return loss, {"valid": 1.0, "gap": gap, "pos_frac": float(n_pos) / K}


def gspo_loss(tok_logp, adv, old_tok_logp=None, *, clip_low=3e-4, clip_high=4e-4):
    """GSPO sequence-level clipped surrogate (Zheng et al., arXiv:2507.18071; mirrors
    REAL's compute_policy_loss_gspo at sequence granularity).

    CRITICAL scale/degeneracy notes for THIS harness (fixed length TPF=320):
    - `old_tok_logp=None` (pure on-policy, first pass) makes the forward ratio exactly 1:
      the clip NEVER fires and the gradient is -A * (1/L) * sum_t grad(logp_t) -- i.e.
      vanilla GRPO scaled by the CONSTANT 1/320 (GSPO's length normalisation only changes
      the geometry for variable-length sequences). We therefore multiply the loss by L so
      the on-policy pass is gradient-identical to the vanilla baseline at the same lr,
      and GSPO-specific content (ratio != 1, live clipping) exists ONLY on extra
      off-policy epochs where `old_tok_logp` is the cached rollout-time log-prob.
    """
    adv_t = torch.as_tensor(adv, device=tok_logp.device, dtype=tok_logp.dtype)
    L = tok_logp.shape[1]
    old = tok_logp.detach() if old_tok_logp is None else old_tok_logp
    # sequence-level ratio value with per-token gradient path (REAL's trick, seq-pooled)
    log_seq_ratio = tok_logp.mean(dim=1) - tok_logp.mean(dim=1).detach() \
        + (tok_logp.detach() - old).mean(dim=1)
    log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
    ratio = torch.exp(log_seq_ratio)
    clipped = torch.clamp(ratio, 1.0 - float(clip_low), 1.0 + float(clip_high))
    loss1 = -adv_t * ratio
    loss2 = -adv_t * clipped
    loss = torch.maximum(loss1, loss2).mean() * L      # L: fixed-length scale alignment
    clipfrac = torch.gt(loss2, loss1).float().mean().detach().item()
    return loss, {"ratio_mean": float(ratio.detach().mean().item()), "clipfrac": float(clipfrac)}


@torch.no_grad()
def eval_model(model, tok, metrics, wins, device, K=8, seed=999, flow=True):
    """Held-out metrics via RLVR-World's Evaluator (MAE/MSE/PSNR/SSIM/LPIPS-vgg, one-to-one
    comparable) + code-RMS/repeat (ours) + DYNAMICS (RAFT flow / frame-delta cosine).
    Per window the Evaluator returns the mean over the K candidates."""
    keys = ("mae", "mse", "psnr", "ssim", "lpips", "code_rms", "repeat", "dmotion", "flow")
    acc = {k: [] for k in keys}
    ar = load_action_ranges(device)
    ev = _get_evaluator(device)
    raft = _get_raft(device) if flow else None
    for wi, (p, s) in enumerate(wins):
        frames, actions = get_window_tensors(p, s, device)
        gt, cur = frames[CTX], frames[CTX - 1]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, K, seed=seed + wi)
        preds = decode_tokens(tok, cand)
        g = gt.unsqueeze(0).unsqueeze(0).expand(preds.shape[0], 1, *gt.shape).contiguous()  # [K,1,3,H,W]
        mae, mse, psnr, ssim, lp = ev(g, preds.unsqueeze(1).contiguous())                   # means over K
        acc["mae"].append(mae.item()); acc["mse"].append(mse.item()); acc["psnr"].append(psnr.item())
        acc["ssim"].append(ssim.item()); acc["lpips"].append(lp.item())
        gt_idx = encode_indices(tok, gt.unsqueeze(0))
        acc["code_rms"].append(float(code_rms(tok, cand, gt_idx).mean()))
        cur_tok = encode_indices(tok, frames[:CTX])[CTX - 1].reshape(-1)
        acc["repeat"].append((cand == cur_tok.unsqueeze(0)).all(dim=1).float().mean().item())
        # dynamics: frame-delta cosine (free, robust) + RAFT flow fidelity (best-effort)
        dp = (preds - cur.unsqueeze(0)).flatten(1)
        dg = (gt - cur).flatten().unsqueeze(0)
        acc["dmotion"].append(F.cosine_similarity(dp, dg, dim=1).mean().item())
        if raft is not None:
            try:
                acc["flow"].append(float(np.mean(flow_fidelity(raft, cur, preds, gt))))
            except Exception:
                acc["flow"].append(float("nan"))
    return {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in acc.items()}


def train(mode, *, reward="pixel", alpha=0.5, phi_tok=0.0, weight_temp=1.0,
          rank_weight=False, rank_sigma=-1.0, rank_sigma_scale=1.0,
          rank_min_weight=0.05,
          dyn_lambda=0.25, dyn_gamma=0.25, dyn_tau=0.0, rcmg_pre_weight=0.5,
          rankcal_weights_path=None, reachable_target_cache=None,
          floor_filter=False, tau=1.0, steps=40, K=8, batch_windows=2,
          train_windows=24, eval_windows=12, lr=1e-5, lam=0.5, beta=0.5, kl=0.0,
          eval_every=10, seed=0, ckpt_dir=None, device="cuda", deterministic=False,
          adv_estimator="grpo", seg_grid=(2, 2), seg_lambda=0.7, seg_gamma=0.0,
          seg_reward="code", real_tau=0.5, real_pos_frac=0.25,
          real_neg_frac=0.25, real_min_gap=0.0,
          gspo_clip_low=3e-4, gspo_clip_high=4e-4, ppo_epochs=1):
    """Run GRPO for one (reward, mode) design; returns a log dict of curves.

    `reward` may be any arm in dor.reward_spaces.ARMS (superset of pixel/code/hybrid,
    which stay bit-identical). `phi_tok` feeds the constant 'floor' arm; `weight_temp`
    the 'dorw' arm. `mode='drgrpo'` = debiased advantage (no /std);
    `mode='rankrel'` or `rank_weight=True` = rank-reliability soft advantage weighting;
    `floor_filter`+`tau` skips floor-dominated groups. All opt-in: defaults reproduce
    the original behaviour bit-for-bit.
    """
    from dor.reward_spaces import gt_reward as gt_reward_space  # local: avoid import cycle
    mrrt_targets = None
    if reward in ("mrrt", "mrrt_random"):
        from dor.reachable_projection import load_mrrt_cache
        mrrt_targets = load_mrrt_cache(reachable_target_cache)
    if deterministic:
        set_determinism(seed)
    s_code = sigma_eta_norm = None
    W_cfg = None
    if floor_filter or rank_weight or mode == "rankrel":
        from dor.reward_spaces import load_weights
        W_cfg = load_weights()
        s_code, sigma_eta_norm = W_cfg["s_code"], W_cfg["sigma_eta_norm"]
    tok = load_tokenizer(device)
    model = load_world_model(device, "base", dtype=torch.float32)
    model.config.use_cache = False
    model.train()
    ref = load_world_model(device, "base", dtype=torch.float32).eval() if kl > 0 else None
    ar = load_action_ranges(device)
    metrics = Metrics(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    allw = sample_windows(list_episodes(), train_windows + eval_windows, seed=1)
    train_w, eval_w = allw[:train_windows], allw[train_windows:]

    seg_ids = k_seg = None
    if adv_estimator in ("seg_grpo", "gpseg"):
        seg_ids = build_seg_ids(seg_grid[0], seg_grid[1], device)
        k_seg = seg_grid[0] * seg_grid[1]

    log = {"step": [], "reward_mean": [], "adv_abs_mean": [], "frac_pos": [],
           "rank_w_mean": [],
           "eval_mae": [], "eval_mse": [], "eval_psnr": [], "eval_ssim": [], "eval_lpips": [],
           "eval_code_rms": [], "eval_repeat": [], "eval_flow": [], "eval_dmotion": []}

    def _rank_sigma_for_reward():
        if rank_sigma and rank_sigma > 0:
            return float(rank_sigma) * float(rank_sigma_scale)
        if W_cfg is None:
            return 0.0
        comps = W_cfg["components"]
        if reward in ("pixel", "pixel_tok", "pixel_tok_dyn", "hybrid_tok", "hybrid"):
            sig = comps["perc"]["phi"]
        elif reward in ("mse", "mse_tok"):
            sig = comps["recon"]["phi"]
        elif reward in ("dorw",):
            sig = W_cfg["sigma_eta_norm"]
        else:
            sig = comps.get("code", {}).get("phi", 0.0)
        return float(sig) * float(rank_sigma_scale)

    def _log_eval(step, r_mean=0.0, adv_abs=0.0, frac_pos=0.0, rank_w=1.0):
        e = eval_model(model, tok, metrics, eval_w, device, K=K)
        log["step"].append(step)
        log["reward_mean"].append(r_mean)
        log["adv_abs_mean"].append(adv_abs)
        log["frac_pos"].append(frac_pos)
        log["rank_w_mean"].append(rank_w)
        for k in ("mae", "mse", "psnr", "ssim", "lpips", "code_rms", "repeat", "flow", "dmotion"):
            log[f"eval_{k}"].append(e[k])
        print(f"[{reward}/{mode}] step {step} LPIPSvgg={e['lpips']:.4f} PSNR={e['psnr']:.2f} "
              f"SSIM={e['ssim']:.4f} flow={e['flow']:.3f} codeRMS={e['code_rms']:.4f} "
              f"rankW={rank_w:.3f}", flush=True)

    _log_eval(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train_w), size=batch_windows)
        opt.zero_grad()
        rwd_acc, adv_acc, pos_acc, rankw_acc, nseq = 0.0, 0.0, 0.0, 0.0, 0
        gspo_cache = []
        for bi in idx:
            p, s = train_w[bi]
            frames, actions = get_window_tensors(p, s, device)
            gt, cur = frames[CTX], frames[CTX - 1]
            prompt = build_prompt(tok, frames, actions, ar)
            with torch.no_grad():
                cand = generate_candidates(model, prompt, K, seed=step * 1000 + int(bi))
                gt_idx = encode_indices(tok, gt.unsqueeze(0))
                reachable_target_idx = None
                if mrrt_targets is not None:
                    target_key = (os.path.basename(p), int(s))
                    if target_key not in mrrt_targets:
                        raise KeyError(f"training window missing from MRRT cache: {target_key}")
                    target_name = "mrrt" if reward == "mrrt" else "mrrt_random"
                    reachable_target_idx = torch.as_tensor(
                        mrrt_targets[target_key][target_name],
                        device=device,
                        dtype=torch.long,
                    )
                if adv_estimator == "seg_grpo":
                    # spatial segment-level credit assignment
                    if seg_reward == "codec_fused":
                        # CAST-GRPO canonical reward (method.md Sec.3.2 v3): SD-GRPO-style
                        # segment decomposition + reliability weight q = relu(corr)^2 between
                        # decoded and token rewards (the per-segment online version of C2's
                        # rho) + Phase-C proven motion residual. Floor enters only via the
                        # advantage's gamma damping (b_tilde), not the reward itself.
                        imgs_s = decode_tokens(tok, cand)
                        target = decode_tokens(tok, gt_idx.reshape(1, -1))
                        # v3.1: r_dec compares to the ACHIEVABLE target decode(encode(gt)),
                        # not raw gt -- the proven C3b floor-cancellation-by-target-alignment
                        # (+14% flow). v3 compared to raw gt assuming z-score absorbs the
                        # offset, but the per-candidate triangle-inequality slack varies, so
                        # target alignment carries real signal beyond a constant shift.
                        pix_sq = pertoken_pix_sq(imgs_s, target)
                        floor_sq_raw = pertoken_pix_sq(target, gt.unsqueeze(0))[0]  # gamma damping only
                        code_sq = pertoken_code_sq(tok, cand, gt_idx)
                        cur_idx = encode_indices(tok, cur.unsqueeze(0))
                        z_cand, z_gt, z_cur = pertoken_code_vecs(tok, cand, gt_idx, cur_idx)
                        dyn_seg = seg_dyn_reward(z_cand, z_gt, z_cur, seg_ids, k_seg, dyn_gamma)
                        r_seg, floor_seg = codec_fused_reward(
                            pix_sq, code_sq, floor_sq_raw, seg_ids, k_seg,
                            dyn_seg=dyn_seg, dyn_lambda=dyn_lambda)
                        adv_tok_t, r_seg_mean = seg_advantage_from_reward(
                            r_seg, seg_ids, seg_lambda, seg_gamma, floor_seg)
                    elif seg_reward in ("pixtok", "pixtok_dyn"):
                        # post-decode floor-cancelled target + per-segment codec floor (gamma active)
                        imgs_s = decode_tokens(tok, cand)
                        target = decode_tokens(tok, gt_idx.reshape(1, -1))
                        pix_sq = pertoken_pix_sq(imgs_s, target)
                        floor_seg = seg_floor_norm(
                            pertoken_pix_sq(target, gt.unsqueeze(0))[0], seg_ids, k_seg)
                        if seg_reward == "pixtok_dyn":
                            # our METHOD: segment-level analogue of the proven pixel_tok_dyn
                            # reward (floor-cancelled pix + code-space motion residual, both
                            # z-scored per segment then combined), not a bare distance.
                            cur_idx = encode_indices(tok, cur.unsqueeze(0))
                            pix_seg = pool_seg_rms(pix_sq, seg_ids, k_seg)          # higher=better
                            z_cand, z_gt, z_cur = pertoken_code_vecs(tok, cand, gt_idx, cur_idx)
                            dyn_seg = seg_dyn_reward(z_cand, z_gt, z_cur, seg_ids, k_seg, dyn_gamma)
                            r_seg = _zscore_cols(pix_seg) + float(dyn_lambda) * _zscore_cols(dyn_seg)
                            adv_tok_t, r_seg_mean = seg_advantage_from_reward(
                                r_seg, seg_ids, seg_lambda, seg_gamma, floor_seg)
                        else:
                            adv_tok_t, r_seg_mean = seg_advantage(
                                pix_sq, seg_ids, k_seg, seg_lambda, seg_gamma, floor_seg)
                    else:  # code (decode-free, floor ~ 0)
                        sq = pertoken_code_sq(tok, cand, gt_idx)
                        adv_tok_t, r_seg_mean = seg_advantage(
                            sq, seg_ids, k_seg, seg_lambda, seg_gamma, None)
                    seg_adv_np = adv_tok_t.detach().float().cpu().numpy()
                else:
                    imgs = decode_tokens(tok, cand)
                    cur_idx = encode_indices(tok, cur.unsqueeze(0))
                    if floor_filter:
                        # skip floor-dominated groups: clean (code) signal below the floor scale
                        cstd = float(np.std(code_rms(tok, cand, gt_idx)))
                        if cstd / s_code <= tau * sigma_eta_norm:
                            continue
                    r_gt = gt_reward_space(reward, metrics, tok, cand, imgs, gt, gt_idx,
                                           alpha=alpha, phi_tok=phi_tok, weight_temp=weight_temp,
                                           cur_idx=cur_idx, dyn_lambda=dyn_lambda,
                                           dyn_gamma=dyn_gamma, dyn_tau=dyn_tau,
                                           rcmg_pre_weight=rcmg_pre_weight,
                                           rankcal_weights_path=rankcal_weights_path,
                                           reachable_target_idx=reachable_target_idx)
                    if mode in ("gt_only", "drgrpo", "rankrel"):
                        adv, _ = shape_advantage(r_gt, mode=mode)
                    else:
                        copy_sim = metrics.eval_batch(imgs, cur)["lpips"]
                        delta = encode_features(tok, imgs) - encode_features(tok, cur.unsqueeze(0))
                        v, _, _ = consensus_support(delta)
                        mm = motion_magnitude(actions[CTX - 1], ar)
                        adv, _ = shape_advantage(r_gt, v, mode=mode, lam=lam, beta=beta,
                                                 motion=mm, copy_sim=copy_sim)
                    rank_w = np.ones_like(adv, dtype=np.float64)
                    if rank_weight or mode == "rankrel":
                        rank_w = rank_reliability_weights(r_gt, _rank_sigma_for_reward())
                        rank_w = float(rank_min_weight) + (1.0 - float(rank_min_weight)) * rank_w
                        adv = rank_w * adv
                        adv = adv - np.mean(adv)
                    adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                    if adv_estimator == "gpseg":
                        # GP-SegGRPO residual (method.md Sec.3.4 v4): the global signal above
                        # is untouched (same code path as standard GRPO); segments only supply
                        # a ZERO-MEAN per-candidate redistribution. lambda=0 or K=1 -> exactly
                        # the global loss. Residual signal = segment-level pixel_tok_dyn proxy
                        # (patch-MSE vs achievable target + code-space motion residual).
                        target = decode_tokens(tok, gt_idx.reshape(1, -1))
                        pix_sq = pertoken_pix_sq(imgs, target)
                        pix_seg = pool_seg_rms(pix_sq, seg_ids, k_seg)
                        zc, zg, zu = pertoken_code_vecs(tok, cand, gt_idx, cur_idx)
                        dyn_seg = seg_dyn_reward(zc, zg, zu, seg_ids, k_seg, dyn_gamma)
                        r_seg = _zscore_cols(pix_seg) + float(dyn_lambda) * _zscore_cols(dyn_seg)
                        if seg_gamma:  # phase-2 optional floor damping (sigma_eta-style, off by default)
                            fl = seg_floor_norm(pertoken_pix_sq(target, gt.unsqueeze(0))[0],
                                                seg_ids, k_seg)
                            mu_s = r_seg.mean(0, keepdim=True)
                            sd_s = r_seg.std(0, unbiased=False, keepdim=True)
                            a_seg = (r_seg - mu_s) / (sd_s + float(seg_gamma) * fl[None, :] + 1e-6)
                        else:
                            a_seg = _zscore_cols(r_seg)
                        resid = a_seg - a_seg.mean(dim=1, keepdim=True)   # zero-mean per candidate
                        resid_tok_t = resid[:, seg_ids]                   # [K, TPF]
            logp_sum, tok_logp = seq_logp(model, prompt, cand)
            if adv_estimator == "seg_grpo":
                pg = -(adv_tok_t * tok_logp).sum(1).mean()
            elif adv_estimator == "gpseg":
                # decomposed loss: L_global (expression-identical to the standard path)
                # + lambda * L_residual (zero-mean redistribution within each rollout)
                pg = (-(adv_t * logp_sum).mean()
                      - float(seg_lambda) * (resid_tok_t * tok_logp).sum(1).mean())
            elif adv_estimator == "real":
                pg, real_info = rank_label_loss(
                    tok_logp, r_gt,
                    pos_frac=real_pos_frac, neg_frac=real_neg_frac,
                    tau=real_tau, min_gap=real_min_gap)
                if pg is None:
                    continue
            elif adv_estimator == "gspo":
                pg, gspo_info = gspo_loss(
                    tok_logp, adv,
                    clip_low=gspo_clip_low, clip_high=gspo_clip_high)
                # cache rollout-time (prompt, cand, adv, old logp) for the extra
                # off-policy epochs -- the ONLY place GSPO differs from vanilla here
                gspo_cache.append((prompt, cand, adv, tok_logp.detach()))
            else:
                pg = -(adv_t * logp_sum).mean()
            if ref is not None:
                with torch.no_grad():
                    _, ref_tok = seq_logp(ref, prompt, cand)
                pg = pg + kl * (tok_logp - ref_tok).mean()
            (pg / len(idx)).backward()
            if adv_estimator == "seg_grpo":
                rwd_acc += r_seg_mean
                adv_acc += float(np.abs(seg_adv_np).mean())
                pos_acc += float((seg_adv_np > 0).mean())
                rankw_acc += 1.0
            elif adv_estimator == "real":
                rwd_acc += float(np.mean(r_gt))
                adv_acc += float(real_info["gap"])
                pos_acc += float(real_info["pos_frac"])
                rankw_acc += float(real_info["valid"])
            elif adv_estimator == "gspo":
                rwd_acc += float(np.mean(r_gt))
                adv_acc += float(np.mean(np.abs(adv)))
                pos_acc += float(np.mean(adv > 0))
                rankw_acc += float(gspo_info["clipfrac"])
            else:
                rwd_acc += float(np.mean(r_gt))
                adv_acc += float(np.mean(np.abs(adv)))
                pos_acc += float(np.mean(adv > 0))
                rankw_acc += float(np.mean(rank_w))
            nseq += 1
        if nseq > 0:  # all groups may be filtered out under floor_filter
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if adv_estimator == "gspo" and ppo_epochs > 1 and gspo_cache:
            # GSPO's actual content: off-policy re-updates on the SAME rollouts against
            # the cached rollout-time logp. From epoch 2 the weights have moved, so the
            # sequence-level ratio deviates from 1 and the clip genuinely fires.
            for _ep in range(ppo_epochs - 1):
                opt.zero_grad()
                ep_ratio, ep_clip = 0.0, 0.0
                for (c_prompt, c_cand, c_adv, c_old) in gspo_cache:
                    _, tok_logp2 = seq_logp(model, c_prompt, c_cand)
                    pg2, info2 = gspo_loss(tok_logp2, c_adv, c_old,
                                           clip_low=gspo_clip_low, clip_high=gspo_clip_high)
                    (pg2 / len(gspo_cache)).backward()
                    ep_ratio += info2["ratio_mean"]; ep_clip += info2["clipfrac"]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                gspo_ep_stats = (ep_ratio / len(gspo_cache), ep_clip / len(gspo_cache))
            if step % eval_every == 0 or step == steps:
                print(f"[gspo] step {step} offpolicy ep ratio_mean={gspo_ep_stats[0]:.6f} "
                      f"clipfrac={gspo_ep_stats[1]:.3f}", flush=True)
        if step % eval_every == 0 or step == steps:
            n = max(nseq, 1)
            _log_eval(step, rwd_acc / n, adv_acc / n, pos_acc / n, rankw_acc / n)
            el = time.time() - t0
            eta = el / step * (steps - step)
            print(f"[{reward}/{mode}] {_bar(step / steps)} {step}/{steps} "
                  f"elapsed={_hms(el)} eta={_hms(eta)} r_gt={rwd_acc / n:.4f} "
                  f"rankW={rankw_acc / n:.3f}", flush=True)

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        print(f"[ckpt] saved {ckpt_dir}", flush=True)
    del model, ref, tok
    torch.cuda.empty_cache()
    return log
