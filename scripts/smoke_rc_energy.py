"""One-window GPU smoke test for the RC-Energy feature and reward wiring."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from dor.constants import CTX, ROOT
from dor.energy_verifier import (
    EnergyFeatureGeometry,
    certified_energy_reward,
    energy_candidate_reward,
    make_energy_config,
)
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.reward_spaces import gt_reward
from dor.tokenization import build_prompt, decode_tokens, encode_indices


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--out", default=f"{ROOT}/outputs/rc_energy/smoke_config.json")
    args = parser.parse_args()
    if args.K < 2:
        raise ValueError("K must be at least two")
    set_determinism(args.seed)
    device = "cuda"
    tokenizer = load_tokenizer(device)
    model = load_world_model(device, "base").eval()
    action_ranges = load_action_ranges(device)
    metrics = Metrics(device)
    geometry = EnergyFeatureGeometry(metrics.lpips)
    path, start = sample_windows(list_episodes(), 1, seed=1)[0]
    frames, actions = get_window_tensors(path, start, device)
    ground_truth = frames[CTX]
    prompt = build_prompt(tokenizer, frames, actions, action_ranges)
    candidates = generate_candidates(model, prompt, args.K, seed=args.seed)
    images = decode_tokens(tokenizer, candidates)
    gt_indices = encode_indices(tokenizer, ground_truth.unsqueeze(0))
    reachable = decode_tokens(tokenizer, gt_indices.reshape(1, -1))[0]
    candidate_blocks = geometry.extract(images)
    target_blocks = geometry.extract(reachable.unsqueeze(0))
    block_target = geometry.block_distances(candidate_blocks, target_blocks)[:, 0]
    scales = np.maximum(np.median(block_target.cpu().numpy(), axis=0), 1e-6)
    config = make_energy_config(
        geometry.block_names, scales, metadata={"smoke_only": True, "K": args.K}
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(config, handle, indent=2)

    rewards = {}
    for kind in ("raw_energy_point", "rc_energy_point", "raw_energy", "rc_energy"):
        target = reachable if kind.startswith("rc_energy") else ground_truth
        direct = energy_candidate_reward(
            metrics.lpips,
            images,
            target,
            config,
            pairwise=not kind.endswith("_point"),
        )
        wired = gt_reward(
            kind,
            metrics,
            tokenizer,
            candidates,
            images,
            ground_truth,
            gt_indices,
            energy_config_path=args.out,
        )
        if not np.isfinite(direct).all() or np.std(direct) <= 1e-8:
            raise RuntimeError(f"{kind} smoke reward is non-finite or constant")
        if not np.allclose(direct, wired, atol=1e-6):
            raise RuntimeError(f"direct and reward_spaces paths disagree for {kind}")
        rewards[kind] = direct
    direct_certified = certified_energy_reward(
        metrics.lpips, images, ground_truth, reachable, config
    )
    wired_certified = gt_reward(
        "rc_energy_certified",
        metrics,
        tokenizer,
        candidates,
        images,
        ground_truth,
        gt_indices,
        energy_config_path=args.out,
    )
    if not np.isfinite(direct_certified).all() or np.std(direct_certified) <= 1e-8:
        raise RuntimeError("rc_energy_certified smoke reward is non-finite or constant")
    if not np.allclose(direct_certified, wired_certified, atol=1e-6):
        raise RuntimeError("direct and wired certified RC-Energy rewards disagree")
    rewards["rc_energy_certified"] = direct_certified
    print(
        f"[smoke] blocks={geometry.block_names} scales="
        + ",".join(f"{value:.6f}" for value in scales),
        flush=True,
    )
    print(
        "[smoke] "
        + " ".join(
            f"{kind}=mean:{value.mean():+.5f}/std:{value.std():.5f}"
            for kind, value in rewards.items()
        ), flush=True)
    print(f"saved {args.out}\nRC_ENERGY_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
