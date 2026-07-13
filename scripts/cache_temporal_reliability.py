"""Cache multi-step candidate groups for rank-reliable return replay.

The script uses only the frozen base policy and calibration windows excluded from
the fixed 24-train/8-eval prefix. It stores per-horizon RC reward, raw evaluator
reward, and a continuous pre-decode dynamics-FSQ diagnostic for identical samples.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from dor.episodes import list_episodes
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges
from dor.multistep import (
    V_MSP,
    detok_chunked,
    discretize_actions,
    load_msp,
    msp_code_rms,
    msp_rollout,
    msp_sample_windows,
    msp_window,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=64)
    parser.add_argument("--exclude_windows", type=int, default=32)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--generation_seeds", default="7301,7302,7303")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    generation_seeds = [int(x) for x in args.generation_seeds.split(",") if x.strip()]
    if len(generation_seeds) < 2:
        raise ValueError("at least two generation seeds are required")
    if args.deterministic:
        set_determinism(generation_seeds[0])
    device = "cuda"
    tok, model = load_msp(device, "base")
    model.eval()
    model.config.use_cache = True
    action_ranges = load_action_ranges(device)
    metrics = Metrics(device)
    windows = msp_sample_windows(
        list_episodes(), args.exclude_windows + args.n_windows, args.T,
        seed=args.window_seed,
    )[args.exclude_windows:]
    if len(windows) != args.n_windows:
        raise RuntimeError(f"requested {args.n_windows} windows, found {len(windows)}")

    values = {name: [[] for _ in generation_seeds]
              for name in ("rc_reward", "raw_reward", "code_reward")}
    episodes, starts = [], []
    started = time.time()
    total = len(windows) * len(generation_seeds)
    done = 0
    print(f"[setup] windows={len(windows)} repetitions={len(generation_seeds)} "
          f"K={args.K} T={args.T} excluded={args.exclude_windows}", flush=True)
    for wi, (path, start) in enumerate(windows):
        frames, actions = msp_window(path, start, args.T, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            idx_c, idx_d_gt = tok.tokenize(frames.unsqueeze(0))
        ctx_off = (idx_c.reshape(1, -1) + V_MSP).long()
        act_off = discretize_actions(actions, action_ranges)[1:args.T] + 2 * V_MSP
        reachable = detok_chunked(tok, idx_c, idx_d_gt)[0]
        real = frames[1:args.T]
        for ri, generation_seed in enumerate(generation_seeds):
            dyn = msp_rollout(
                model, ctx_off, act_off, args.T - 1, args.K,
                seed=generation_seed + wi,
                temperature=args.temperature, top_k=args.top_k,
            )
            pred = detok_chunked(tok, idx_c.expand(args.K, -1, -1), dyn)
            code = -msp_code_rms(tok, dyn, idx_d_gt).detach().cpu().numpy()
            rc_rows, raw_rows = [], []
            # Horizon 1 has no direct reward. Cache future horizons 2..T-1.
            for horizon in range(1, args.T - 1):
                q_rc = metrics.eval_batch(pred[:, horizon], reachable[horizon])
                q_raw = metrics.eval_batch(pred[:, horizon], real[horizon])
                rc_rows.append(-(
                    np.asarray(q_rc["mse"], dtype=np.float64)
                    + np.asarray(q_rc["lpips"], dtype=np.float64)
                ))
                raw_rows.append(-(
                    np.asarray(q_raw["mse"], dtype=np.float64)
                    + np.asarray(q_raw["lpips"], dtype=np.float64)
                ))
            values["rc_reward"][ri].append(np.stack(rc_rows))
            values["raw_reward"][ri].append(np.stack(raw_rows))
            values["code_reward"][ri].append(code[:, 1:args.T - 1].T)
            done += 1
            if done % 4 == 0 or done == total:
                elapsed = time.time() - started
                print(f"[cache] {_bar(done / total)} {done}/{total} "
                      f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (total-done))}",
                      flush=True)
        episodes.append(os.path.basename(path))
        starts.append(start)

    payload = {
        name: np.stack([np.stack(rep) for rep in repetitions]).astype(np.float32)
        for name, repetitions in values.items()
    }
    payload.update({
        "episode": np.asarray(episodes),
        "start": np.asarray(starts, dtype=np.int32),
        "horizon": np.arange(2, args.T, dtype=np.int16),
        "meta_json": np.asarray(json.dumps({
            "n_windows": args.n_windows,
            "exclude_windows": args.exclude_windows,
            "window_seed": args.window_seed,
            "generation_seeds": generation_seeds,
            "K": args.K,
            "T": args.T,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "policy": "base",
        }, sort_keys=True)),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"[done] {args.out} shape={payload['rc_reward'].shape} "
          f"elapsed={_hms(time.time()-started)}\nTEMPORAL_RELIABILITY_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
