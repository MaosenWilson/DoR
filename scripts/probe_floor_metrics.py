"""Cross-metric reward-noise floor (Fig 1 evidence, eval-only, no training).

For each held-out GT next frame s', compare two quantities in EVERY metric:
  floor  = d(decode(encode(s')), s')   # tokenizer round-trip error = phi_tok
  signal = d(prev_frame,        s')    # the real inter-frame change the policy must learn

Claim (universality across metrics): the floor is comparable to / larger than the
dynamic signal in MAE/MSE/SSIM/PSNR/LPIPS alike -> the noise floor is not a quirk of
LPIPS; it drowns the dynamics in every metric. "Drowned" = the round-trip is no closer
to GT than simply copying the previous frame.

Uses dor.metrics.Metrics (LPIPS=vgg, matching the report). Pure encode/decode + metrics,
no candidate generation -> fast.
"""
import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch

from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.metrics import Metrics
from dor.models import load_tokenizer
from dor.tokenization import decode_tokens, encode_indices

KEYS = ("mae", "mse", "psnr", "ssim", "lpips")
LOWER_IS_FARTHER = {"mae", "mse", "lpips"}  # distance metrics; the rest (psnr/ssim) are similarities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/floor_metrics.json")
    args = ap.parse_args()

    dev = "cuda"
    tok = load_tokenizer(dev)
    M = Metrics(dev)  # LPIPS=vgg
    wins = sample_windows(list_episodes(), args.n_windows, seed=args.seed)
    print(f"[setup] windows={len(wins)} (cross-metric floor vs inter-frame signal)", flush=True)

    floor = {k: [] for k in KEYS}
    signal = {k: [] for k in KEYS}
    t0 = time.time()
    for wi, (p, s) in enumerate(wins):
        frames, _ = get_window_tensors(p, s, dev)
        gt, prev = frames[CTX], frames[CTX - 1]
        recon = decode_tokens(tok, encode_indices(tok, gt.unsqueeze(0)).reshape(1, -1))  # [1,3,H,W]
        qf = M.eval_batch(recon, gt)                       # floor = round-trip error
        qm = M.eval_batch(prev.unsqueeze(0), gt)           # signal = inter-frame change
        for k in KEYS:
            floor[k].append(float(qf[k][0]))
            signal[k].append(float(qm[k][0]))
        if (wi + 1) % 50 == 0:
            print(f"[{wi + 1}/{len(wins)}] {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== cross-metric floor vs inter-frame signal (N={len(wins)}) ===")
    print(f"{'metric':8s}{'floor(round-trip)':>20s}{'signal(inter-frame)':>22s}{'verdict':>20s}")
    out = {}
    for k in KEYS:
        f, g = np.array(floor[k]), np.array(signal[k])
        # 'drowned' = round-trip is no closer to GT than copying prev frame
        drowned = (f.mean() >= g.mean()) if k in LOWER_IS_FARTHER else (f.mean() <= g.mean())
        verdict = "FLOOR >= SIGNAL" if drowned else "floor < signal"
        print(f"{k:8s}{f.mean():>12.4f}±{f.std():.4f}{g.mean():>14.4f}±{g.std():.4f}{verdict:>20s}")
        out[k] = {"floor_mean": float(f.mean()), "floor_std": float(f.std()),
                  "signal_mean": float(g.mean()), "signal_std": float(g.std()), "drowned": bool(drowned)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json
    json.dump({"n_windows": len(wins), "metrics": out}, open(args.out, "w"), indent=2)
    print(f"\n[done] saved {args.out}")
    print("FLOOR_METRICS_OK")


if __name__ == "__main__":
    main()
