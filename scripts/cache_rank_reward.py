"""Cache disjoint calibration groups for rank-calibrated reward fitting.

The fixed split follows the single-step training harness: seed=1 windows 0:24
are training and 24:36 evaluation.  Calibration starts at offset 36, so its
contexts cannot leak into either set.
"""
import argparse
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lpips as lpips_lib
import numpy as np
import torch
from piqa import FSIM, GMSD, MS_SSIM, HaarPSI

from dor.constants import CTX, ROOT
from dor.spatial_pool import block_pool, floor_weights, weighted_pool
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import code_rms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.reward_spaces import code_gradient_reward
from dor.tokenization import build_prompt, decode_tokens, encode_indices


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=256)
    ap.add_argument("--exclude_windows", type=int, default=36,
                    help="skip fixed 24 train + 12 eval windows")
    ap.add_argument("--window_seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--generation_seed", type=int, default=7301)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/rank_reward_calibration.npz")
    args = ap.parse_args()

    if args.deterministic:
        set_determinism(args.generation_seed)
    dev = "cuda"
    tok = load_tokenizer(dev)
    model = load_world_model(dev, "base")
    ar = load_action_ranges(dev)
    metrics = Metrics(dev)
    lpips_sp = lpips_lib.LPIPS(net="vgg", spatial=True).to(dev).eval()
    for p in lpips_sp.parameters():
        p.requires_grad_(False)
    panel = {
        "msssim": MS_SSIM(reduction="none").to(dev),
        "gmsd": GMSD(reduction="none").to(dev),
        "haarpsi": HaarPSI(reduction="none").to(dev),
        "fsim": FSIM(reduction="none").to(dev),
    }
    all_windows = sample_windows(
        list_episodes(), args.exclude_windows + args.n_windows, seed=args.window_seed
    )
    windows = all_windows[args.exclude_windows:]
    if len(windows) != args.n_windows:
        raise RuntimeError(f"requested {args.n_windows} calibration windows, got {len(windows)}")

    spatial_configs = tuple(
        (scheme, q) for scheme in ("iv", "im") for q in (0.25, 0.50, 0.75)
    )
    spatial_names = tuple(
        f"{kind}_{scheme}{int(q * 100)}"
        for kind in ("wlp", "wmse") for scheme, q in spatial_configs
    )
    names = ("raw_lpips", "raw_mse", "raw_ssim", "reach_lpips", "reach_mse",
             "reach_ssim", "code", "grad",
             "reach_msssim", "reach_gmsd", "reach_haarpsi", "reach_fsim") + spatial_names
    values = {name: [] for name in names}
    paths, starts = [], []
    t0 = time.time()
    print(f"[setup] calibration windows={len(windows)} K={args.K} "
          f"excluded_prefix={args.exclude_windows}", flush=True)
    for wi, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, dev)
        gt = frames[CTX]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(
            model, prompt, args.K, temperature=args.temperature, top_k=args.top_k,
            seed=args.generation_seed + wi,
        )
        imgs = decode_tokens(tok, cand)
        gt_idx = encode_indices(tok, gt.unsqueeze(0))
        reachable = decode_tokens(tok, gt_idx.reshape(1, -1))[0]
        raw = metrics.eval_batch(imgs, gt)
        reach = metrics.eval_batch(imgs, reachable)
        if "ssim" not in raw or "ssim" not in reach:
            raise RuntimeError("rank calibration requires piqa SSIM (pip install piqa)")

        for prefix, result in (("raw", raw), ("reach", reach)):
            for metric in ("lpips", "mse", "ssim"):
                values[f"{prefix}_{metric}"].append(np.asarray(result[metric], np.float32))
        values["code"].append(-np.asarray(code_rms(tok, cand, gt_idx), np.float32))
        values["grad"].append(
            np.asarray(code_gradient_reward(tok, cand, gt_idx), np.float32)
        )
        with torch.no_grad():
            x = imgs.clamp(0.0, 1.0)
            y = reachable.clamp(0.0, 1.0).unsqueeze(0).expand_as(x).contiguous()
            for pname, module in panel.items():
                values[f"reach_{pname}"].append(
                    module(x, y).detach().cpu().numpy().astype(np.float32)
                )
            # Floor-aware spatial pooling (plan appendix S): local floor maps from the
            # GT/reachable pair, candidate error maps against the reachable target,
            # both block-pooled; weights are frozen functions of the floor map only.
            gtc = gt.clamp(0.0, 1.0)
            rc1 = y[:1]
            phi_lp = block_pool(lpips_sp(rc1 * 2 - 1, gtc.unsqueeze(0) * 2 - 1)[:, 0])[0]
            phi_mse = block_pool((rc1[0] - gtc).square().mean(0))
            lp_maps = block_pool(lpips_sp(x * 2 - 1, y * 2 - 1)[:, 0])
            mse_maps = block_pool((x - y).square().mean(1))
            for scheme, q in spatial_configs:
                tag = f"{scheme}{int(q * 100)}"
                w_lp = floor_weights(phi_lp, q, scheme)
                w_mse = floor_weights(phi_mse, q, scheme)
                values[f"wlp_{tag}"].append(
                    weighted_pool(lp_maps, w_lp).detach().cpu().numpy().astype(np.float32))
                values[f"wmse_{tag}"].append(
                    weighted_pool(mse_maps, w_mse).detach().cpu().numpy().astype(np.float32))
        paths.append(os.path.basename(path))
        starts.append(start)

        done = wi + 1
        if done % 5 == 0 or done == len(windows):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(windows) - done)
            width = 24
            filled = int(done / len(windows) * width)
            bar = "#" * filled + "-" * (width - filled)
            print(f"[cache {bar}] {done}/{len(windows)} elapsed={hms(elapsed)} "
                  f"eta={hms(eta)}", flush=True)

    payload = {name: np.stack(rows).astype(np.float32) for name, rows in values.items()}
    payload.update({
        "episode": np.asarray(paths),
        "start": np.asarray(starts, dtype=np.int32),
        "meta": np.asarray([args.K, len(windows), args.window_seed, args.exclude_windows,
                            args.generation_seed, int(args.deterministic)], dtype=np.int64),
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **payload)
    print(f"[done] {args.out} shape=({len(windows)},{args.K}) "
          f"total={hms(time.time() - t0)}", flush=True)
    print("CACHE_RANK_REWARD_OK", flush=True)


if __name__ == "__main__":
    main()
