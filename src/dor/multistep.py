"""Multi-step CAST-GRPO scaffolding for the dor harness.

Two parts:
  (1) TEMPORAL-SPATIAL SEGMENT MACHINERY  -- weight-independent, unit-tested here.
      Generalises the single-step spatial seg-ids to H frames x K per-frame segments,
      so `dor.grpo.seg_advantage` (already generic in seg_ids) computes the spatio-
      temporal advantage with no change.
  (2) MULTI-STEP ROLLOUT  -- must faithfully reuse RLVR-World's ctx_msp processor +
      CompressiveVQModelFSQ + per-frame BOS/EOS sequence format; finalised & tested
      against the downloaded weights (see ROLLOUT PLAN below). Left as a documented
      stub on purpose: dumping it untested would silently corrupt generation/decoding.

Downloaded assets (checkpoints/):
  rt1-world-model-multi-step-base   policy base (Llama-style, vocab 9008)
  rt1-world-model-multi-step-rlvr   RLVR's own multi-step RL result (baseline)
  rt1-compressive-tokenizer         CompressiveVQModelFSQ (context full + future 80 dyn tokens)
"""
import torch


def frame_seg_ids(per_frame_tokens, K, grid=None):
    """Per-frame token -> segment id [per_frame_tokens] in [0, Kf); returns (ids, Kf).

    grid=(gh0, gw0): the per-frame tokens are a clean gh0 x gw0 spatial grid. K is then a
        (sgh, sgw) spatial split (each must divide gh0/gw0)  -> true spatial segments.
    grid=None: contiguous token-chunk split into K parts (fallback when the tokens are not
        a clean grid, e.g. the compressive tokenizer's 80 dynamics tokens).
    """
    if grid is not None:
        gh0, gw0 = grid
        if gh0 * gw0 != per_frame_tokens:
            raise ValueError(f"grid {grid} != per_frame_tokens {per_frame_tokens}")
        sgh, sgw = K if isinstance(K, (tuple, list)) else (1, int(K))
        if gh0 % sgh or gw0 % sgw:
            raise ValueError(f"spatial split {(sgh, sgw)} does not divide grid {grid}")
        rows = torch.arange(gh0) // (gh0 // sgh)
        cols = torch.arange(gw0) // (gw0 // sgw)
        return (rows[:, None] * sgw + cols[None, :]).reshape(-1).long(), sgh * sgw
    Kc = K if isinstance(K, int) else K[0] * K[1]
    per = -(-per_frame_tokens // Kc)  # ceil, contiguous chunks
    ids = (torch.arange(per_frame_tokens) // per).clamp(max=Kc - 1)
    return ids.long(), Kc


def build_seg_ids_st(H, per_frame_tokens, K, grid=None, device="cpu"):
    """Temporal x spatial segment ids for H frames x per_frame_tokens tokens.

    Returns (seg_ids [H*per_frame_tokens] in [0, H*Kf), n_seg=H*Kf). Token order is
    frame-major: [frame0 tokens..., frame1 tokens..., ...]. seg = h*Kf + frame_seg.
    H=1 reduces to the single-step spatial seg-ids; K=1 -> temporal-only credit.
    """
    fseg, Kf = frame_seg_ids(per_frame_tokens, K, grid)
    fseg = fseg.to(device)
    h = torch.arange(H, device=device).repeat_interleave(per_frame_tokens)
    seg = h * Kf + fseg.repeat(H)
    return seg.long(), H * Kf


# ---------------------------------------------------------------------------
# MULTI-STEP (ctx_msp) ROLLOUT + REWARDS -- implemented & smoke-tested 2026-07-04
# (probe_msp.py: MSP_PROBE_OK; vocab layout verified V=4375 | ctx +V | act +2V,
#  multi-step floor LPIPS ~0.077, rollout sane with smooth horizon accumulation)
# ---------------------------------------------------------------------------
import numpy as np
import torch.nn.functional as F_

from dor.constants import ROOT
from dor.episodes import load_episode

MSP_TOK_DIR = f"{ROOT}/checkpoints/rt1-compressive-tokenizer"
MSP_BASE_DIR = f"{ROOT}/checkpoints/rt1-world-model-multi-step-base"
MSP_RLVR_DIR = f"{ROOT}/checkpoints/rt1-world-model-multi-step-rlvr"
V_MSP = 4375                 # dyn vocab; ctx offset +V, action offset +2V (verified P0)
DYN_PER_FRAME = 80           # 8x10 dyn grid
ACT_DIM, ACT_BINS = 13, 256
MSP_EOS = 9007


def load_msp(dev, which="base"):
    """(compressive tokenizer, multi-step world model). which in {base, rlvr}."""
    import dor.compat  # noqa: F401
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
    from transformers import AutoModelForCausalLM
    tok = CompressiveVQModelFSQ.from_pretrained(MSP_TOK_DIR).to(dev).eval()
    path = {"base": MSP_BASE_DIR, "rlvr": MSP_RLVR_DIR}[which]
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(dev)
    return tok, model


def msp_window(path, start, T, dev):
    img, act = load_episode(path)
    frames = torch.from_numpy(img[start:start + T]).float().div(255.0).permute(0, 3, 1, 2).to(dev)
    actions = torch.from_numpy(act[start:start + T]).to(dev)
    return frames, actions


def msp_sample_windows(paths, n_windows, T, seed=0, stride=8):
    rng = np.random.default_rng(seed)
    wins = []
    for p in paths:
        n = np.load(p, allow_pickle=True)["image"].shape[0]
        wins += [(p, s) for s in range(0, n - T, stride)]
    rng.shuffle(wins)
    return wins[:n_windows]


def discretize_actions(actions, ar):
    """actions [T,13] float, ar [13,2] (min,max) -> long [T,13] in [0,256)."""
    lo, hi = ar[:, 0], ar[:, 1]
    a = ((actions - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return (a * ACT_BINS).floor().long().clamp(0, ACT_BINS - 1)


@torch.no_grad()
def detok_chunked(tok, idx_c, idx_d, k_chunk=8):
    """Memory-safe detokenize: per future frame, candidates sub-chunked (the conditional
    decoder OOMs on 32GB if all frames decode at once). idx_c [B,1,1280], idx_d [B,F,80]
    -> [B, F, 3, 256, 320] float in [0,1]."""
    B, F = idx_d.shape[0], idx_d.shape[1]
    frames_out = []
    for i in range(F):
        rows = []
        for b0 in range(0, B, k_chunk):
            b1 = min(b0 + k_chunk, B)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                r = tok.detokenize(idx_c[b0:b1], idx_d[b0:b1, i:i + 1])
            r = r.float().clamp(0, 1)
            rows.append(r.reshape(b1 - b0, -1, *r.shape[-3:])[:, -1])
        frames_out.append(torch.cat(rows, 0))
        torch.cuda.empty_cache()
    return torch.stack(frames_out, dim=1)


@torch.no_grad()
def msp_rollout(model, ctx_off, act_off, n_future, K, seed,
                temperature=1.0, top_k=100):
    """Faithful ctx_msp interact loop (HF generate): per future frame, sample 80 dyn
    tokens then FORCE the frame's action tokens. ctx_off [1,1280] (+V applied),
    act_off [n_future,13] (+2V applied). Returns dyn [K, n_future, 80] raw-range."""
    torch.manual_seed(seed)
    seq = ctx_off.expand(K, -1).contiguous()
    outs = []
    for t in range(n_future):
        gen = model.generate(
            input_ids=seq, do_sample=True, temperature=temperature, top_k=top_k,
            num_return_sequences=1, max_new_tokens=DYN_PER_FRAME,
            min_new_tokens=DYN_PER_FRAME, eos_token_id=None, pad_token_id=MSP_EOS)
        outs.append(gen[:, seq.shape[1]:seq.shape[1] + DYN_PER_FRAME])
        seq = torch.cat([gen, act_off[t].unsqueeze(0).expand(K, -1)], dim=1)
    return torch.stack(outs, dim=1).clamp(0, V_MSP - 1)


def msp_token_logp(model, ctx_off, act_off, dyn, autocast=True):
    """Per-candidate, per-frame, per-token log-prob of the sampled DYN tokens.

    The full sequence is [ctx | dyn_1 act_1 | ... | dyn_F act_F]. Action positions are
    excluded from the objective because they are forced conditions, not sampled outputs.
    dyn [K,F,80] raw-range. Returns tok_logp [K,F,80].
    """
    K, F = dyn.shape[0], dyn.shape[1]
    per_frame = torch.cat([dyn, act_off.unsqueeze(0).expand(K, -1, -1)], dim=2)  # [K,F,93]
    full = torch.cat([ctx_off.expand(K, -1), per_frame.reshape(K, -1)], dim=1)   # [K, 1280+F*93]
    ctxlen = ctx_off.shape[1]
    ctx_mgr = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast \
        else torch.enable_grad()
    with ctx_mgr:
        logits = model(full).logits
    toks = []
    for t in range(F):
        s = ctxlen + t * (DYN_PER_FRAME + ACT_DIM)
        pred = logits[:, s - 1:s - 1 + DYN_PER_FRAME, :].float()
        lp = F_.log_softmax(pred, dim=-1)
        tok_lp = lp.gather(-1, dyn[:, t, :].unsqueeze(-1)).squeeze(-1)           # [K,80]
        toks.append(tok_lp)
    return torch.stack(toks, dim=1)                                               # [K,F,80]


def msp_seq_logp(model, ctx_off, act_off, dyn, autocast=True):
    """Per-candidate summed log-prob of the DYN tokens. Returns logp_sum [K]."""
    return msp_token_logp(model, ctx_off, act_off, dyn, autocast=autocast).sum(dim=(1, 2))


def msp_rewards(pred, real_f, target_f, metrics, kind="rc"):
    """Per-candidate scalar reward, following msp_reward_fn's convention: SKIP the first
    future frame (no action conditions it), aggregate mean over the rest.
      raw: -(MSE + LPIPS)(pred, REAL)            (RLVR-World faithful)
      rc:  -(MSE + LPIPS)(pred, REACHABLE target = detokenize(ctx, GT dyn))  (ours)
    pred [K,F,3,H,W]; real_f/target_f [F,3,H,W]. Returns np [K]."""
    frame = msp_rewards_frame(pred, real_f, target_f, metrics, kind=kind)
    return frame[:, 1:].mean(axis=1)


def msp_rewards_frame(pred, real_f, target_f, metrics, kind="rc"):
    """Per-candidate, per-frame reward [K,F].

    Frame 0 is kept at zero to preserve RLVR-World's multi-step convention: the first
    future frame has no direct action-conditioned reward. Temporal-return GRPO can still
    assign it credit through later rewards.
    """
    ref = target_f if kind == "rc" else real_f
    K, F = pred.shape[0], pred.shape[1]
    out = np.zeros((K, F), dtype=np.float64)
    for i in range(1, F):                                        # skip first future frame
        q = metrics.eval_batch(pred[:, i], ref[i])
        out[:, i] = -(np.asarray(q["mse"], float) + np.asarray(q["lpips"], float))
    return out


def _selftest():
    # spatial: 16x20=320 grid, 2x2 spatial, H=3 -> 3*4=12 segments
    ids, n = build_seg_ids_st(3, 320, (2, 2), grid=(16, 20))
    assert ids.shape == (3 * 320,) and n == 12 and int(ids.max()) == 11
    # H=1 spatial reduces to single-step (4 segments)
    ids1, n1 = build_seg_ids_st(1, 320, (2, 2), grid=(16, 20))
    assert n1 == 4 and int(ids1.max()) == 3
    # token_chunk fallback: 80 tokens into 4 chunks, H=2 -> 8 segments
    ids2, n2 = build_seg_ids_st(2, 80, 4, grid=None)
    assert ids2.shape == (160,) and n2 == 8 and int(ids2.max()) == 7
    # K=1 -> temporal-only: H segments
    ids3, n3 = build_seg_ids_st(4, 80, 1, grid=None)
    assert n3 == 4 and int(ids3.max()) == 3
    print("MULTISTEP_SEGIDS_OK")


if __name__ == "__main__":
    _selftest()
