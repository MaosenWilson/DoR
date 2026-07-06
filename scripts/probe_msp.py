"""Multi-step (ctx_msp) engineering smoke probe -- M2/M3 in experiments.md.

Three phases, each a go/no-go gate before any multi-step training:

  P0  tokenize a real window with the compressive tokenizer; print shapes and index
      ranges; empirically confirm the vocab layout the SFT model was trained with
      (dyn raw [0,V) | ctx +V | action +2V | BOS/EOS 9006/9007, V inferred).
  P1  FLOOR: detokenize(GT ctx, GT dyn) -> recon frames; per-frame MSE/LPIPS vs real.
      This is the MULTI-STEP reconstruction floor (does floor-cancellation transfer?).
      Gate: recon must look like the scene (LPIPS well below garbage level ~0.5).
  P2  ROLLOUT: interleaved autoregressive generation with the multi-step BASE model,
      faithfully reproducing RLVR-World's interact loop (generate 80 dyn tokens per
      frame, then FORCE the next action's 13 discretized tokens; first future frame
      is conditioned on ctx only and skipped by the reward, per their convention).
      Gate: decoded rollout frames are sane (finite, LPIPS < garbage), error grows
      smoothly with horizon rather than exploding at frame 1.

Sequence layout (verified against ivideogpt/processor.py ContextMultiStepPrediction-
Processor and verl vllm_rollout interact loop):
  [ctx(1280)+V | dyn_1(80) | act_1(13)+2V | dyn_2(80) | act_2(13)+2V | ...]
Reward convention (verl ray_trainer.msp_reward_fn): pred[:,1:] vs real[:,2:].
"""
import argparse
import time

import numpy as np
import torch

from dor.constants import ROOT
from dor.episodes import list_episodes, load_episode
from dor.metrics import Metrics
from dor.models import load_action_ranges

TOK_DIR = f"{ROOT}/checkpoints/rt1-compressive-tokenizer"
BASE_DIR = f"{ROOT}/checkpoints/rt1-world-model-multi-step-base"
DYN_PER_FRAME = 80          # 8x10 grid (compressive_vq_model_fsq.detokenize dyn_res)
ACT_DIM = 13
ACT_BINS = 256


def load_msp_models(dev):
    import dor.compat  # noqa: F401  puts ivideogpt on sys.path
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
    from transformers import AutoModelForCausalLM
    tok = CompressiveVQModelFSQ.from_pretrained(TOK_DIR).to(dev).eval()
    model = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float32).to(dev).eval()
    return tok, model


def discretize_actions(actions, ar):
    """actions [T,13] float, ar [13,2] (min,max) -> [T,13] long in [0,256)."""
    lo, hi = ar[:, 0], ar[:, 1]
    a = ((actions - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return (a * ACT_BINS).floor().long().clamp(0, ACT_BINS - 1)


def msp_window(path, start, T, dev):
    img, act = load_episode(path)
    seg = img[start:start + T]
    frames = torch.from_numpy(seg).float().div(255.0).permute(0, 3, 1, 2).to(dev)
    actions = torch.from_numpy(act[start:start + T]).to(dev)
    return frames, actions


@torch.no_grad()
def detok_chunked(tok, idx_c, idx_d):
    """Memory-safe detokenize: one future frame per call (the conditional decoder OOMs
    on a 5090 if all frames are decoded in one batch). idx_c [B,1,1280], idx_d [B,F,80]
    -> frames [B, F, 3, 256, 320] float in [0,1]."""
    outs = []
    for i in range(idx_d.shape[1]):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r = tok.detokenize(idx_c, idx_d[:, i:i + 1])
        r = r.float().clamp(0, 1)
        outs.append(r.reshape(idx_c.shape[0], -1, *r.shape[-3:])[:, -1])  # last = the future frame
        torch.cuda.empty_cache()
    return torch.stack(outs, dim=1)


@torch.no_grad()
def interleaved_rollout(model, ctx_ids_offset, act_tok_offset, n_future, K, seed, dev,
                        temperature=1.0, top_k=100, eos_id=9007):
    """Faithful reproduction of the verl interact loop with HF generate.

    ctx_ids_offset [1, 1280] long (already +V); act_tok_offset [T-1, 13] (already +2V,
    act_tok_offset[t-1] is the action appended AFTER generated frame t).
    Returns dyn tokens [K, n_future, 80] (raw, un-offset -- generation happens in raw
    dyn range [0,V) because that's all the SFT model ever emits at dyn positions;
    we clamp to [0,V) exactly like msp_reward_fn does)."""
    torch.manual_seed(seed)
    seq = ctx_ids_offset.expand(K, -1).contiguous()             # [K, 1280]
    outs = []
    for t in range(n_future):
        gen = model.generate(
            input_ids=seq, do_sample=True, temperature=temperature, top_k=top_k,
            num_return_sequences=1, max_new_tokens=DYN_PER_FRAME,
            min_new_tokens=DYN_PER_FRAME, eos_token_id=None, pad_token_id=eos_id,
        )                                                        # [K, len+80]
        dyn = gen[:, seq.shape[1]:seq.shape[1] + DYN_PER_FRAME]  # [K, 80]
        outs.append(dyn)
        act = act_tok_offset[t].unsqueeze(0).expand(K, -1)       # [K, 13] forced
        seq = torch.cat([gen, act], dim=1)
    return torch.stack(outs, dim=1)                              # [K, n_future, 80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8, help="segment length (1 ctx + T-1 future)")
    ap.add_argument("--K", type=int, default=2, help="rollout candidates (smoke)")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"

    tok, model = load_msp_models(dev)
    ar = load_action_ranges(dev)
    metrics = Metrics(dev)
    V = None
    print(f"[setup] compressive tokenizer + multi-step base loaded; model vocab="
          f"{model.config.vocab_size}", flush=True)

    eps = list_episodes()
    rng = np.random.default_rng(args.seed)
    floors, rollouts = [], []
    for wi in range(args.windows):
        path = eps[int(rng.integers(0, len(eps)))]
        Tlen = np.load(path, allow_pickle=True)["image"].shape[0]
        start = int(rng.integers(0, Tlen - args.T))
        frames, actions = msp_window(path, start, args.T, dev)   # [T,3,256,320], [T,13]

        # ---- P0: tokenize, infer vocab layout ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            idx_c, idx_d = tok.tokenize(frames.unsqueeze(0))     # [1,1,1280], [1,T-1,80]
        if V is None:
            V = model.config.vocab_size // 2 - (ACT_BINS + 2) // 2  # candidate; verify below
            v_fit = 2 * 4375 + ACT_BINS + 2  # = 9008 layout hypothesis
            print(f"[P0] idx_c shape={tuple(idx_c.shape)} range=[{idx_c.min()},{idx_c.max()}]  "
                  f"idx_d shape={tuple(idx_d.shape)} range=[{idx_d.min()},{idx_d.max()}]  "
                  f"vocab_fit(2*4375+258={v_fit})={'OK' if v_fit == model.config.vocab_size else 'MISMATCH'}",
                  flush=True)
            V = 4375
            assert idx_c.max() < V and idx_d.max() < V, "index exceeds assumed V=4375!"

        # ---- P1: GT round-trip = multi-step floor ----
        rec_f = detok_chunked(tok, idx_c, idx_d)[0]               # [T-1, 3,256,320]
        real_f = frames[1:]
        fl = [float(metrics.eval_batch(rec_f[i:i + 1], real_f[i])["lpips"][0])
              for i in range(args.T - 1)]
        floors.append(fl)
        print(f"[P1] window {wi}: per-frame roundtrip LPIPS (floor) = "
              f"{[round(x, 3) for x in fl]}", flush=True)

        # ---- P2: interleaved rollout with base model ----
        act_disc = discretize_actions(actions, ar)               # [T,13]
        act_off = act_disc[1:args.T - 1 + 1] + 2 * V             # actions 1..T-1, offset
        ctx_off = (idx_c.reshape(1, -1) + V).long()
        t0 = time.time()
        dyn = interleaved_rollout(model, ctx_off, act_off, args.T - 1, args.K,
                                  seed=args.seed * 1000 + wi, dev=dev)
        dyn = dyn.clamp(0, V - 1)
        el = time.time() - t0
        pf = detok_chunked(tok, idx_c.expand(args.K, -1, -1), dyn)  # [K, T-1, 3,256,320]
        rl = [[float(metrics.eval_batch(pf[k, i:i + 1], real_f[i])["lpips"][0])
               for i in range(args.T - 1)] for k in range(args.K)]
        rollouts.append(rl)
        print(f"[P2] window {wi}: rollout LPIPS per frame (K={args.K}, {el:.0f}s) =",
              [[round(x, 3) for x in row] for row in rl], flush=True)

    fl_mean = np.mean([f for w in floors for f in w])
    rl_first = np.mean([row[1] for w in rollouts for row in w])   # frame 2 (reward starts here)
    rl_last = np.mean([row[-1] for w in rollouts for row in w])
    print(f"\n[SUMMARY] floor LPIPS mean={fl_mean:.3f} | rollout LPIPS frame2={rl_first:.3f} "
          f"last={rl_last:.3f}", flush=True)
    ok = (fl_mean < 0.35 and rl_first < 0.55 and np.isfinite(rl_last)
          and rl_first >= fl_mean * 0.8)
    print("MSP_PROBE_OK" if ok else "MSP_PROBE_FAIL", flush=True)


if __name__ == "__main__":
    main()
