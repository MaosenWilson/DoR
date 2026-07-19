"""C1 reconstruction-floor audit for the NVIDIA Cosmos discrete (FSQ) tokenizer.

Zero world model, zero training: encode -> FSQ discrete tokens -> decode a real
video clip and measure the reconstruction floor d(decode(encode(s)), s). If a
completely different FSQ tokenizer (Cosmos: wavelet + causal-temporal + FSQ, from
NVIDIA) also has a non-trivial floor, the reconstruction-floor problem is a
property of the FSQ family, not of our specific tokenizers.

Uses the same LPIPS-VGG (dor.metrics.Metrics) as our CNN-FSQ / compressive-FSQ
floor measurements so the magnitudes are directly comparable. Cosmos JIT models
load standalone via torch.jit.load -- no cosmos_tokenizer package required.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dor.metrics import Metrics


def _to_clips(images, clip_len, resolution):
    """images [T,H,W,3] uint8 -> list of [3,clip_len,res,res] float in [-1,1]."""
    frames = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)  # [T,3,H,W]
    if tuple(frames.shape[-2:]) != (resolution, resolution):
        frames = F.interpolate(frames, size=(resolution, resolution),
                               mode="bilinear", align_corners=False, antialias=True)
    clips = []
    for s in range(0, frames.shape[0] - clip_len + 1, clip_len):
        clip = frames[s:s + clip_len]                      # [clip_len,3,res,res] in [0,1]
        clips.append(clip.permute(1, 0, 2, 3) * 2 - 1)     # [3,clip_len,res,res] in [-1,1]
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos_dir", required=True, help="dir with encoder.jit / decoder.jit")
    ap.add_argument("--frames_npz", required=True, help="npz with image [T,H,W,3] uint8")
    ap.add_argument("--image_key", default="image")
    ap.add_argument("--clip_len", type=int, default=9, help="8k+1 for 8x temporal")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    enc = torch.jit.load(f"{args.cosmos_dir}/encoder.jit").to(device).eval()
    dec = torch.jit.load(f"{args.cosmos_dir}/decoder.jit").to(device).eval()
    metrics = Metrics(device)

    images = np.asarray(np.load(args.frames_npz, allow_pickle=False)[args.image_key])
    clips = _to_clips(images, args.clip_len, args.resolution)
    if not clips:
        raise ValueError(f"trajectory too short for clip_len={args.clip_len}")
    print(f"[cosmos] {len(clips)} clips x {args.clip_len} frames @ {args.resolution}px", flush=True)

    lp, ms = [], []
    for ci, clip in enumerate(clips):
        x = clip.unsqueeze(0).to(device)                    # [1,3,T,res,res] in [-1,1]
        with torch.no_grad():
            tokens = enc(x)
            rec = dec(tokens[0] if isinstance(tokens, (tuple, list)) else tokens)
        rec = (rec[0] if isinstance(rec, (tuple, list)) else rec).float().clamp(-1, 1)
        gt01 = (x[0].permute(1, 0, 2, 3) + 1) / 2           # [T,3,res,res] in [0,1]
        rec01 = (rec[0].permute(1, 0, 2, 3) + 1) / 2
        for t in range(gt01.shape[0]):
            q = metrics.eval_batch(rec01[t:t + 1], gt01[t])
            lp.append(float(np.asarray(q["lpips"]).mean()))
            ms.append(float(np.asarray(q["mse"]).mean()))
        print(f"[cosmos clip {ci + 1}/{len(clips)}]", flush=True)

    print("\n=== Cosmos DV-FSQ reconstruction floor (LPIPS-VGG, same metric as ours) ===")
    print("floor LPIPS = %.4f +- %.4f   floor MSE = %.5f  (n=%d frames)" %
          (np.mean(lp), np.std(lp), np.mean(ms), len(lp)))
    print("compare: RLVR-World CNN-FSQ single-step floor ~0.053 LPIPS ; "
          "compressive-FSQ multi-step ~0.077 LPIPS")
    print("COSMOS_FLOOR_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
