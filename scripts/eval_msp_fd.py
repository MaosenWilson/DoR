"""Reward-independent distributional eval for MULTI-STEP checkpoints:
FD-DINOv2 + KID-DINOv2 between predicted rollout frames and real frames.

Answers the reviewer objection "training reward and evaluation metric coincide
(MSE+LPIPS both sides)": DINOv2 features are never used as reward. Both are SET
statistics -> evaluation only. Lower is better; KID is the primary readout at
our sample sizes (unbiased), FD reported for completeness.

Protocol: held-out windows disjoint from the 24 training windows (same shuffle,
seed=1, indices [24:24+n]); per arm we roll out K candidates per window with the
same per-window seed, decode, keep horizons 2..7 (first future frame skipped,
matching the paper), and pool frames. Real pool = the same windows' real frames
at horizons 2..7.

Arms: base / rlvr / any saved ckpt dir from train_grpo_msp.py.

Example (smoke):
  python scripts/eval_msp_fd.py --n_windows 4 --K 2 --arms base=BASE
Formal:
  python scripts/eval_msp_fd.py --n_windows 64 --K 4 \
    --arms base=BASE,rlvr=RLVR,seq_rc=outputs/msp_step30_seq/ckpt/rc_msp_s0,\
return_rc=outputs/msp_step30_return_hkl00/ckpt/rc_msp_s0
"""
import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy import linalg

from dor.constants import ROOT
from dor.episodes import list_episodes
from dor.models import load_action_ranges
from dor.multistep import (detok_chunked, discretize_actions, load_msp, msp_rollout,
                           msp_sample_windows, msp_window, V_MSP)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Dino:
    def __init__(self, dev, name="facebook/dinov2-small"):
        from transformers import Dinov2Model
        self.m = Dinov2Model.from_pretrained(name).to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(dev), _STD.to(dev)

    @torch.no_grad()
    def __call__(self, imgs, bs=64):
        feats = []
        for i in range(0, imgs.shape[0], bs):
            x = Fnn.interpolate(imgs[i:i + bs], size=224, mode="bilinear",
                                align_corners=False)
            x = (x - self.mean) / self.std
            feats.append(self.m(pixel_values=x).last_hidden_state[:, 0])
        return torch.cat(feats).cpu().numpy()


def frechet(mu1, cov1, mu2, cov2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((cov1 + eps * np.eye(cov1.shape[0])) @
                              (cov2 + eps * np.eye(cov2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))


def kid(X, Y, deg=3, c=1.0):
    d = X.shape[1]
    Kxx = ((X @ X.T) / d + c) ** deg
    Kyy = ((Y @ Y.T) / d + c) ** deg
    Kxy = ((X @ Y.T) / d + c) ** deg
    n, m = X.shape[0], Y.shape[0]
    sx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(sx + sy - 2 * Kxy.mean())


def load_arm_model(spec, dev):
    if spec in ("BASE", "base"):
        return load_msp(dev, "base")
    if spec in ("RLVR", "rlvr"):
        return load_msp(dev, "rlvr")
    tok, _ = load_msp(dev, "base")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=torch.float32).to(dev)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--train_windows", type=int, default=24,
                    help="skip this shuffle prefix so eval stays disjoint from training")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--arms", required=True,
                    help="comma list name=BASE|RLVR|ckpt_dir")
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/msp_fd_dino.json")
    args = ap.parse_args()
    dev = "cuda"

    allw = msp_sample_windows(list_episodes(), args.train_windows + args.n_windows,
                              args.T, seed=1)
    wins = allw[args.train_windows:]
    print(f"[setup] eval windows={len(wins)} K={args.K} horizons 2..{args.T - 1}",
          flush=True)
    dino = Dino(dev)
    ar = load_action_ranges(dev)

    # real pool: window rows are [ctx, future 1..T-1]; horizons 2..T-1 -> rows 2..T-1
    real = torch.cat([msp_window(p, s, args.T, dev)[0][2:args.T] for p, s in wins])
    G = dino(real)
    muG, covG = G.mean(0), np.cov(G, rowvar=False)
    print(f"[real] {real.shape[0]} frames embedded", flush=True)

    results = {}
    for spec in args.arms.split(","):
        name, path = spec.split("=", 1)
        tok, model = load_arm_model(path, dev)
        model.config.use_cache = True
        model.eval()
        feats = []
        t0 = time.time()
        with torch.no_grad():
            for wi, (p, s) in enumerate(wins):
                frames, actions = msp_window(p, s, args.T, dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    idx_c, _ = tok.tokenize(frames.unsqueeze(0))
                ctx_off = (idx_c.reshape(1, -1) + V_MSP).long()
                act_off = discretize_actions(actions, ar)[1:args.T] + 2 * V_MSP
                dyn = msp_rollout(model, ctx_off, act_off, args.T - 1, args.K,
                                  seed=args.seed + wi)
                pred = detok_chunked(tok, idx_c.expand(args.K, -1, -1), dyn)
                # pred: [K, T-1, 3, H, W]; horizons 2..T-1 -> indices 1..T-2
                sel = pred[:, 1:args.T - 1].reshape(-1, *pred.shape[2:])
                feats.append(dino(sel))
                if (wi + 1) % 8 == 0 or wi + 1 == len(wins):
                    el = time.time() - t0
                    print(f"[{name} {wi + 1}/{len(wins)}] elapsed={el:.0f}s "
                          f"eta={el / (wi + 1) * (len(wins) - wi - 1):.0f}s", flush=True)
        P = np.concatenate(feats)
        fd = frechet(P.mean(0), np.cov(P, rowvar=False), muG, covG)
        ki = kid(P, G)
        results[name] = {"fd_dino": fd, "kid_dino": ki, "n_pred": int(P.shape[0]),
                         "n_real": int(G.shape[0])}
        print(f"[{name}] FD-DINO={fd:.3f} KID-DINO={ki:.5f}", flush=True)
        del model, tok
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"saved {args.out}\nMSP_FD_DONE")


if __name__ == "__main__":
    main()
