"""Time and compare cached/uncached HF sampling on one real RT-1 context."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import set_determinism
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt


def _sample(model, prompt, group_size, seed, use_kv_cache):
    torch.cuda.synchronize()
    started = time.perf_counter()
    candidates = generate_candidates(
        model,
        prompt,
        group_size,
        seed=seed,
        use_kv_cache=use_kv_cache,
    )
    torch.cuda.synchronize()
    return candidates, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.K < 1:
        raise ValueError("K must be positive")

    set_determinism(args.seed)
    tokenizer = load_tokenizer(args.device)
    model = load_world_model(args.device, "base", dtype=torch.float32).eval()
    action_ranges = load_action_ranges(args.device)
    path, start = sample_windows(list_episodes(), 1, seed=1)[0]
    frames, actions = get_window_tensors(path, start, args.device)
    prompt = build_prompt(tokenizer, frames, actions, action_ranges)
    model.config.use_cache = False

    cached, cached_seconds = _sample(model, prompt, args.K, args.seed, True)
    uncached, uncached_seconds = _sample(model, prompt, args.K, args.seed, False)
    hamming = float((cached != uncached).float().mean().item())
    report = {
        "protocol": "RT-1 HF generation KV-cache audit v1",
        "K": args.K,
        "seed": args.seed,
        "cached_seconds": cached_seconds,
        "uncached_seconds": uncached_seconds,
        "speedup": uncached_seconds / max(cached_seconds, 1e-12),
        "candidate_hamming_fraction": hamming,
        "config_use_cache_after": bool(model.config.use_cache),
        "candidate_shape": list(cached.shape),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"saved {output}\nRT1_GENERATION_CACHE_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
