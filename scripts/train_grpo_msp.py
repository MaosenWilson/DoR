"""Multi-step (ctx_msp) GRPO with RC reward -- lean loop mirroring train_grpo.py.

Arms (--rewards):
  raw   -(MSE+LPIPS)(pred, REAL frames)                RLVR-World-faithful multi-step reward
  rc    -(MSE+LPIPS)(pred, REACHABLE target)           ours: reachable-target alignment
                                                       (target = detokenize(ctx, GT dyn),
                                                        floor ~0.077 LPIPS measured)
Reward convention follows verl msp_reward_fn: skip the first future frame, mean-aggregate.
Eval: held-out windows; per-horizon LPIPS/MSE vs REAL (frames 2..F), mean + last.
Baselines for the paper: base ckpt (step-0 eval) and official multi-step-rlvr
(eval with --eval_only --which rlvr).

Example (smoke):
  python scripts/train_grpo_msp.py --rewards rc --seeds 0 --steps 2 --K 8 --T 8
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from dor.constants import ROOT
from dor.episodes import list_episodes
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges
from dor.multistep import (detok_chunked, discretize_actions, load_msp, msp_rewards_frame,
                           msp_rollout, msp_sample_windows, msp_token_logp, msp_window,
                           V_MSP)


@torch.no_grad()
def eval_msp(model, tok, ar, metrics, wins, T, K, dev, seed=999):
    lp_mean, lp_last, mse_mean = [], [], []
    for wi, (p, s) in enumerate(wins):
        frames, actions = msp_window(p, s, T, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            idx_c, _ = tok.tokenize(frames.unsqueeze(0))
        ctx_off = (idx_c.reshape(1, -1) + V_MSP).long()
        act_off = discretize_actions(actions, ar)[1:T] + 2 * V_MSP
        dyn = msp_rollout(model, ctx_off, act_off, T - 1, K, seed=seed + wi)
        pred = detok_chunked(tok, idx_c.expand(K, -1, -1), dyn)
        real_f = frames[1:]
        lps, mses = [], []
        for i in range(1, T - 1):
            q = metrics.eval_batch(pred[:, i], real_f[i])
            lps.append(float(np.mean(q["lpips"])))
            mses.append(float(np.mean(q["mse"])))
        lp_mean.append(float(np.mean(lps)))
        lp_last.append(lps[-1])
        mse_mean.append(float(np.mean(mses)))
    return {"lpips": float(np.mean(lp_mean)), "lpips_last": float(np.mean(lp_last)),
            "mse": float(np.mean(mse_mean))}


def _discounted_frame_advantage(r_frame, beta):
    ret = np.zeros_like(r_frame)
    running = np.zeros(r_frame.shape[0], dtype=np.float64)
    for t in range(r_frame.shape[1] - 1, -1, -1):
        running = r_frame[:, t] + beta * running
        ret[:, t] = running
    adv = np.zeros_like(ret)
    for t in range(ret.shape[1]):
        x = ret[:, t]
        adv[:, t] = (x - x.mean()) / (x.std() + 1e-6)
    return adv


def _gain_shaped_rewards(r_frame, alpha):
    """Video-specific gain shaping: reward frames that improve over the previous frame.

    r_frame[:,0] is the artificial no-direct-reward future frame in the MSP convention,
    so gain starts at t=2 (0-indexed) to avoid treating that zero as a real score.
    """
    shaped = np.array(r_frame, dtype=np.float64, copy=True)
    if alpha == 0.0 or shaped.shape[1] <= 2:
        return shaped
    gain = np.zeros_like(shaped)
    gain[:, 2:] = shaped[:, 2:] - shaped[:, 1:-1]
    shaped[:, 1:] = shaped[:, 1:] + alpha * gain[:, 1:]
    return shaped


def _temporal_advantages(r_frame, mode, beta, gain_alpha=0.0):
    """Build sequence or frame-block advantages from per-frame rewards.

    r_frame [K,F] follows the MSP convention: frame 0 has no direct reward and is zero.
    Returns (adv_seq [K] or None, adv_frame [K,F] or None, scalar_reward_mean).
    """
    r = np.asarray(r_frame, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"r_frame must be [K,F], got {r.shape}")
    valid = r[:, 1:] if r.shape[1] > 1 else r
    scalar = valid.mean(axis=1)
    if mode == "seq":
        adv = (scalar - scalar.mean()) / (scalar.std() + 1e-6)
        return adv, None, float(scalar.mean())

    if mode == "frame":
        adv = np.zeros_like(r)
        for t in range(1, r.shape[1]):
            x = r[:, t]
            adv[:, t] = (x - x.mean()) / (x.std() + 1e-6)
        return None, adv, float(scalar.mean())

    if mode == "return":
        return None, _discounted_frame_advantage(r, beta), float(scalar.mean())

    if mode == "gain_return":
        shaped = _gain_shaped_rewards(r, gain_alpha)
        return None, _discounted_frame_advantage(shaped, beta), float(scalar.mean())

    raise ValueError(f"unknown adv_temporal={mode!r}")


def _horizon_weights(F, alpha, dev):
    if alpha == 0.0 or F <= 1:
        return torch.ones(F, device=dev, dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, F, device=dev, dtype=torch.float32)
    return 1.0 + alpha * t


def train_one(reward, seed, args, dev="cuda"):
    if args.deterministic:
        set_determinism(seed)
    tok, model = load_msp(dev, "base")
    model.config.use_cache = True
    ar = load_action_ranges(dev)
    metrics = Metrics(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ref = None
    if args.kl > 0:
        # KL anchor to the base policy -- the official RLVR-World multi-step recipe uses
        # kl_loss_coef=0.001. Without it, both v1 (bw=1) and v2 (bw=2) pilots learned well
        # until ~step 30 then diverged (policy drift compounds over the 7-frame rollout);
        # bw=1->2 did NOT fix it, so the missing KL anchor is the prime suspect.
        _, ref = load_msp(dev, "base")
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    allw = msp_sample_windows(list_episodes(), args.train_windows + args.eval_windows,
                              args.T, seed=1)
    train_w, eval_w = allw[:args.train_windows], allw[args.train_windows:]

    log = {"step": [], "reward_mean": [], "eval_lpips": [], "eval_lpips_last": [],
           "eval_mse": []}

    def _log_eval(step, r_mean=0.0):
        model.eval()
        e = eval_msp(model, tok, ar, metrics, eval_w, args.T, args.K, dev)
        model.train()
        log["step"].append(step)
        log["reward_mean"].append(r_mean)
        for k, v in e.items():
            log[f"eval_{k}"].append(v)
        print(f"[{reward}/msp] step {step} LPIPS={e['lpips']:.4f} "
              f"LPIPSlast={e['lpips_last']:.4f} MSE={e['mse']:.5f}", flush=True)

    _log_eval(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train_w), size=args.batch_windows)
        opt.zero_grad()
        r_acc, nseq = 0.0, 0
        for bi in idx:
            p, s = train_w[bi]
            frames, actions = msp_window(p, s, args.T, dev)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    idx_c, idx_d_gt = tok.tokenize(frames.unsqueeze(0))
                ctx_off = (idx_c.reshape(1, -1) + V_MSP).long()
                act_off = discretize_actions(actions, ar)[1:args.T] + 2 * V_MSP
                dyn = msp_rollout(model, ctx_off, act_off, args.T - 1, args.K,
                                  seed=step * 1000 + int(bi))
                pred = detok_chunked(tok, idx_c.expand(args.K, -1, -1), dyn)
                real_f = frames[1:]
                target_f = detok_chunked(tok, idx_c, idx_d_gt)[0] if reward == "rc" else real_f
                r_frame = msp_rewards_frame(pred, real_f, target_f, metrics, kind=reward)
                adv, adv_frame, r_mean = _temporal_advantages(
                    r_frame, args.adv_temporal, args.temporal_gamma, args.gain_alpha)
            model.config.use_cache = False
            tok_logp = msp_token_logp(model, ctx_off, act_off, dyn)
            model.config.use_cache = True
            if args.adv_temporal == "seq":
                adv_t = torch.tensor(adv, device=dev, dtype=torch.float32)
                logp_sum = tok_logp.sum(dim=(1, 2))
                pg = -(adv_t * logp_sum).mean()
            else:
                adv_tok = torch.tensor(adv_frame, device=dev, dtype=torch.float32)
                pg = -((adv_tok[:, :, None] * tok_logp).sum(dim=(1, 2))).mean()
            if ref is not None:
                with torch.no_grad():
                    ref_tok = msp_token_logp(ref, ctx_off, act_off, dyn)
                hw = _horizon_weights(dyn.shape[1], args.horizon_kl_alpha, dev)
                pg = pg + args.kl * ((tok_logp - ref_tok) * hw[None, :, None]).mean()
            (pg / len(idx)).backward()
            r_acc += r_mean; nseq += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            _log_eval(step, r_acc / max(nseq, 1))
            el = time.time() - t0
            print(f"[{reward}/msp] {_bar(step / args.steps)} {step}/{args.steps} "
                  f"elapsed={_hms(el)} eta={_hms(el / step * (args.steps - step))} "
                  f"r={r_acc / max(nseq, 1):.4f}", flush=True)

    ckpt = os.path.join(args.out_dir, "ckpt", f"{reward}_msp_s{seed}")
    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt)
    print(f"[ckpt] saved {ckpt}", flush=True)
    del model, tok
    torch.cuda.empty_cache()
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", default="rc", help="comma list from {raw, rc}")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--T", type=int, default=8, help="segment length (1 ctx + T-1 future)")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_windows", type=int, default=1)
    ap.add_argument("--train_windows", type=int, default=24)
    ap.add_argument("--eval_windows", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=0.001,
                    help="KL coef to the frozen base policy (official multi-step recipe "
                         "uses 0.001; 0 disables and reproduces the diverging v1/v2 setup)")
    ap.add_argument("--adv_temporal", default="seq",
                    choices=["seq", "frame", "return", "gain_return"],
                    help="multi-step advantage granularity: old rollout scalar, per-frame, "
                         "discounted temporal return, or gain-shaped temporal return")
    ap.add_argument("--temporal_gamma", type=float, default=0.95,
                    help="discount beta for --adv_temporal return/gain_return")
    ap.add_argument("--gain_alpha", type=float, default=0.5,
                    help="reward-improvement shaping strength for --adv_temporal gain_return")
    ap.add_argument("--horizon_kl_alpha", type=float, default=0.0,
                    help="0 = uniform KL; >0 linearly strengthens KL on later rollout frames")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--eval_only", action="store_true",
                    help="no training: eval --which ckpt on the held-out windows")
    ap.add_argument("--which", default="rlvr", choices=["base", "rlvr"],
                    help="ckpt for --eval_only (official multi-step baseline)")
    ap.add_argument("--out_dir", default=f"{ROOT}/outputs/msp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.eval_only:
        dev = "cuda"
        tok, model = load_msp(dev, args.which)
        model.eval()
        ar = load_action_ranges(dev)
        metrics = Metrics(dev)
        allw = msp_sample_windows(list_episodes(), args.train_windows + args.eval_windows,
                                  args.T, seed=1)
        e = eval_msp(model, tok, ar, metrics, allw[args.train_windows:], args.T, args.K, dev)
        out = os.path.join(args.out_dir, f"evalonly_{args.which}.json")
        json.dump(e, open(out, "w"), indent=2)
        print(f"[eval_only/{args.which}] {e}\nsaved {out}\nMSP_EVAL_OK", flush=True)
        return

    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    combos = [(r, s) for s in seeds for r in rewards]
    done = 0
    t0 = time.time()
    for reward, seed in combos:
        out = os.path.join(args.out_dir, f"sweep_{reward}_msp_s{seed}.json")
        if os.path.exists(out):
            done += 1
            continue
        print(f"\n===== MSP SWEEP {_bar(done / len(combos))} {done}/{len(combos)} "
              f"next: arm={reward} seed={seed} =====", flush=True)
        log = train_one(reward, seed, args)
        json.dump({"args": vars(args), "run": {f"{reward}-msp": log}}, open(out, "w"))
        print(f"[done] saved {out}", flush=True)
        done += 1
    print(f"\n[sweep done] {done}/{len(combos)} in {_hms(time.time() - t0)}\nGRPO_MSP_OK",
          flush=True)


if __name__ == "__main__":
    main()
