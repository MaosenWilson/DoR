"""Block 0 (matrix §5): cache per-candidate reward distances in *every* space plus
the held-out DINOv2 reference, WITHOUT any training. This is the de-risk step:
from this npz alone we can compute the reward-noise floor (Fig 1) and the
per-space rank-preservation (Fig 2/3) and decide whether the thesis holds before
spending training compute.

Self-contained like `probe_tokenizer_floor.py` / `probe_phi_dino.py`.

Per window (N) x candidate (K) it stores raw quantities (higher-level rewards are
derived offline in `analyze_rank_preservation.py`, so the floor value can be swept):
  lpips, mse, psnr, ssim   pixel-space metrics vs GT next frame
  code_rms                 pre-decode FSQ code-space RMS  (DoR arm A6)
  phi_rms                  pre-decode continuous encoder-feature RMS (arm A5)
  dino_cos                 1 - cos(DINOv2(cand), DINOv2(gt))  -> REFERENCE quality (held-out)
Per window (N) scalars:
  floor_lpips              LPIPS(decode(encode(gt)), gt)  -> reward-noise floor phi_tok in LPIPS
  motion                   action motion magnitude
"""
import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
import torch.nn.functional as Fnn

from dor.consensus import motion_magnitude
from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import code_rms
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.reward_spaces import phi_rms
from dor.tokenization import build_prompt, decode_tokens, encode_indices

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _hms(sec):
    """seconds -> H:MM:SS for progress / ETA display."""
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Dino:
    """DINOv2 CLS embedder (held-out reference). imgs [B,3,H,W] in [0,1] -> cls [B,D]."""

    def __init__(self, dev, name="facebook/dinov2-small"):
        from transformers import Dinov2Model
        self.m = Dinov2Model.from_pretrained(name).to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(dev), _STD.to(dev)

    @torch.no_grad()
    def __call__(self, imgs):
        x = Fnn.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.m(pixel_values=x).last_hidden_state[:, 0]  # cls [B,D]


def dino_cos(dino, imgs, gt):
    a = Fnn.normalize(dino(imgs), dim=-1)
    b = Fnn.normalize(dino(gt.unsqueeze(0)), dim=-1)
    return (1.0 - (a * b).sum(-1)).detach().cpu().numpy()  # [K]


@torch.no_grad()
def roundtrip_floor(tok, metrics, gt):
    """LPIPS(decode(encode(gt)), gt) -> reward-noise floor phi_tok for this window."""
    idx = encode_indices(tok, gt.unsqueeze(0)).reshape(1, -1)  # [1,320]
    rec = decode_tokens(tok, idx)                              # [1,3,256,320]
    return float(metrics.eval_batch(rec, gt)["lpips"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=100)  # RLVR-World eval default
    ap.add_argument("--which", default="base", choices=["base", "rlvr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dino", default="facebook/dinov2-small")
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/reward_spaces.npz")
    args = ap.parse_args()

    dev = "cuda"
    tok = load_tokenizer(dev)
    model = load_world_model(dev, args.which)
    ar = load_action_ranges(dev)
    M = Metrics(dev)
    dino = Dino(dev, args.dino)
    wins = sample_windows(list_episodes(), args.n_windows, seed=args.seed)
    print(f"[setup] windows={len(wins)} K={args.K} model={args.which} dino={args.dino}", flush=True)

    keys = ("lpips", "mse", "psnr", "ssim", "code_rms", "phi_rms", "dino_cos")
    agg = {k: [] for k in keys}
    floor, motion = [], []
    t0 = time.time()
    for wi, (p, s) in enumerate(wins):
        frames, actions = get_window_tensors(p, s, dev)
        gt = frames[CTX]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, args.K, temperature=args.temperature,
                                   top_k=args.top_k, seed=args.seed * 100000 + wi)
        imgs = decode_tokens(tok, cand)
        q = M.eval_batch(imgs, gt)
        gt_idx = encode_indices(tok, gt.unsqueeze(0))

        agg["lpips"].append(q["lpips"])
        agg["mse"].append(q["mse"])
        agg["psnr"].append(q["psnr"])
        agg["ssim"].append(q["ssim"] if "ssim" in q else np.full(args.K, np.nan, np.float32))
        agg["code_rms"].append(code_rms(tok, cand, gt_idx))
        agg["phi_rms"].append(phi_rms(tok, imgs, gt))
        agg["dino_cos"].append(dino_cos(dino, imgs, gt))
        floor.append(roundtrip_floor(tok, M, gt))
        motion.append(motion_magnitude(actions[CTX - 1], ar))
        if (wi + 1) % 10 == 0 or (wi + 1) == len(wins):
            done = wi + 1
            elapsed = time.time() - t0
            rate = elapsed / done                      # s per window
            eta = rate * (len(wins) - done)            # remaining seconds
            print(f"[{done}/{len(wins)}] elapsed={_hms(elapsed)} "
                  f"eta={_hms(eta)} ({rate:.1f}s/win)", flush=True)

    out = {k: np.stack(v).astype(np.float32) for k, v in agg.items()}  # [N,K]
    out["floor_lpips"] = np.array(floor, np.float32)                   # [N]
    out["motion"] = np.array(motion, np.float32)                       # [N]
    out["meta"] = np.array([args.K, len(wins), args.seed])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"[done] saved {args.out}  N={len(wins)} K={args.K} "
          f"floor_lpips_mean={np.mean(floor):.4f} total={time.time() - t0:.0f}s", flush=True)
    print("CACHE_REWARD_SPACES_OK", flush=True)


if __name__ == "__main__":
    main()
