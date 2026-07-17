"""Cache two independent candidate groups for the RC-Energy admission gate."""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from dor.constants import CTX, ROOT
from dor.energy_verifier import EnergyFeatureGeometry, make_energy_config
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens, encode_indices


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=148)
    parser.add_argument("--exclude_windows", type=int, default=36)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--generation_seeds", default="7301,17301")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--scale_episode_fraction", type=float, default=0.2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default=f"{ROOT}/outputs/rc_energy/two_group_cache.npz")
    parser.add_argument("--config_out", default=f"{ROOT}/configs/aaai2027/rc_energy.json")
    args = parser.parse_args()

    generation_seeds = [int(value) for value in args.generation_seeds.split(",") if value]
    if len(generation_seeds) != 2:
        raise ValueError("the pre-registered gate requires exactly two generation seeds")
    if args.K < 2:
        raise ValueError("RC-Energy requires K >= 2")
    if args.deterministic:
        set_determinism(generation_seeds[0])

    device = "cuda"
    tokenizer = load_tokenizer(device)
    model = load_world_model(device, "base")
    model.eval()
    action_ranges = load_action_ranges(device)
    metrics = Metrics(device)
    geometry = EnergyFeatureGeometry(metrics.lpips)
    windows = sample_windows(
        list_episodes(), args.exclude_windows + args.n_windows, seed=args.window_seed
    )[args.exclude_windows:]
    if len(windows) != args.n_windows:
        raise RuntimeError(f"requested {args.n_windows} windows, found {len(windows)}")

    reach_blocks, raw_blocks, within_blocks, cross_blocks = [], [], [], []
    raw_lpips, raw_mse, raw_ssim = [], [], []
    episodes, starts = [], []
    started = time.time()
    print(
        f"[setup] windows={len(windows)} groups=2 K={args.K} "
        f"feature_blocks={','.join(geometry.block_names)}",
        flush=True,
    )
    for wi, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, device)
        ground_truth = frames[CTX]
        prompt = build_prompt(tokenizer, frames, actions, action_ranges)
        gt_indices = encode_indices(tokenizer, ground_truth.unsqueeze(0))
        reachable = decode_tokens(tokenizer, gt_indices.reshape(1, -1))[0]
        target_blocks = geometry.extract(reachable.unsqueeze(0))
        raw_target_blocks = geometry.extract(ground_truth.unsqueeze(0))

        group_features = []
        context_reach, context_raw, context_within = [], [], []
        context_lpips, context_mse, context_ssim = [], [], []
        for generation_seed in generation_seeds:
            candidates = generate_candidates(
                model,
                prompt,
                args.K,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=generation_seed + wi,
            )
            images = decode_tokens(tokenizer, candidates)
            features = geometry.extract(images)
            group_features.append(features)
            context_reach.append(
                geometry.block_distances(features, target_blocks)[:, 0].cpu().numpy()
            )
            context_raw.append(
                geometry.block_distances(features, raw_target_blocks)[:, 0].cpu().numpy()
            )
            context_within.append(
                geometry.block_distances(features, features).cpu().numpy()
            )
            raw_metrics = metrics.eval_batch(images, ground_truth)
            context_lpips.append(np.asarray(raw_metrics["lpips"], dtype=np.float32))
            context_mse.append(np.asarray(raw_metrics["mse"], dtype=np.float32))
            context_ssim.append(
                np.asarray(raw_metrics.get("ssim", np.full(args.K, np.nan)), dtype=np.float32)
            )

        context_cross = []
        for group_index in range(2):
            partner = 1 - group_index
            context_cross.append(
                geometry.block_distances(
                    group_features[group_index], group_features[partner]
                ).cpu().numpy()
            )
        reach_blocks.append(np.stack(context_reach))
        raw_blocks.append(np.stack(context_raw))
        within_blocks.append(np.stack(context_within))
        cross_blocks.append(np.stack(context_cross))
        raw_lpips.append(np.stack(context_lpips))
        raw_mse.append(np.stack(context_mse))
        raw_ssim.append(np.stack(context_ssim))
        episodes.append(os.path.basename(path))
        starts.append(int(start))

        done = wi + 1
        if done % 4 == 0 or done == len(windows):
            elapsed = time.time() - started
            eta = elapsed / done * (len(windows) - done)
            print(
                f"[cache] {_bar(done / len(windows))} {done}/{len(windows)} "
                f"elapsed={_hms(elapsed)} eta={_hms(eta)}",
                flush=True,
            )

    reach_blocks = np.stack(reach_blocks).transpose(1, 0, 2, 3).astype(np.float32)
    raw_blocks = np.stack(raw_blocks).transpose(1, 0, 2, 3).astype(np.float32)
    within_blocks = np.stack(within_blocks).transpose(1, 0, 2, 3, 4).astype(np.float32)
    cross_blocks = np.stack(cross_blocks).transpose(1, 0, 2, 3, 4).astype(np.float32)
    raw_lpips = np.stack(raw_lpips).transpose(1, 0, 2).astype(np.float32)
    raw_mse = np.stack(raw_mse).transpose(1, 0, 2).astype(np.float32)
    raw_ssim = np.stack(raw_ssim).transpose(1, 0, 2).astype(np.float32)

    unique_episodes = np.unique(np.asarray(episodes))
    if not 0.0 < args.scale_episode_fraction < 0.5:
        raise ValueError("scale_episode_fraction must lie in (0, 0.5)")
    scale_count = max(1, int(np.ceil(len(unique_episodes) * args.scale_episode_fraction)))
    scale_rng = np.random.default_rng(args.window_seed + 991)
    scale_episodes = np.sort(
        scale_rng.choice(unique_episodes, size=scale_count, replace=False)
    )
    is_scale = np.isin(np.asarray(episodes), scale_episodes)
    if is_scale.all() or (~is_scale).sum() < 2:
        raise RuntimeError("episode-disjoint scale/gate split is degenerate")
    scales = np.median(reach_blocks[:, is_scale], axis=(0, 1, 2)).astype(np.float64)
    scales = np.maximum(scales, 1e-6)
    metadata = {
        "n_windows": args.n_windows,
        "exclude_windows": args.exclude_windows,
        "window_seed": args.window_seed,
        "generation_seeds": generation_seeds,
        "K": args.K,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "policy": "base",
        "scale_episode_fraction": args.scale_episode_fraction,
        "scale_episodes": scale_episodes.tolist(),
        "gate_windows": int((~is_scale).sum()),
    }
    config = make_energy_config(geometry.block_names, scales, beta=1.0, metadata=metadata)

    os.makedirs(os.path.dirname(os.path.abspath(args.config_out)), exist_ok=True)
    with open(args.config_out, "w") as handle:
        json.dump(config, handle, indent=2)
    payload = {
        "reach_blocks": reach_blocks,
        "raw_blocks": raw_blocks,
        "within_blocks": within_blocks,
        "cross_blocks": cross_blocks,
        "raw_lpips": raw_lpips,
        "raw_mse": raw_mse,
        "raw_ssim": raw_ssim,
        "episode": np.asarray(episodes),
        "is_scale": is_scale.astype(np.bool_),
        "start": np.asarray(starts, dtype=np.int32),
        "block_names": np.asarray(geometry.block_names),
        "block_scales": scales.astype(np.float32),
        "meta_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(
        f"[done] cache={args.out} config={args.config_out} "
        f"shape={reach_blocks.shape} elapsed={_hms(time.time()-started)}",
        flush=True,
    )
    print("RC_ENERGY_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
