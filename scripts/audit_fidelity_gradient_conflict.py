"""Audit real parameter-gradient conflict before training anchored GRPO."""

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
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.gradient_constraints import gradient_inner_products, projection_statistics
from dor.grpo import _bar, _hms, seq_logp, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.reward_spaces import gt_reward
from dor.rewards import shape_advantage
from dor.tokenization import build_prompt, decode_tokens, encode_indices


def _pair_flip_fraction(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    upper = np.triu_indices(len(left), k=1)
    return float(np.mean(
        np.sign(left[:, None] - left[None, :])[upper]
        != np.sign(right[:, None] - right[None, :])[upper]
    ))


def _cosine(left, right, eps=1e-12):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(left @ right / max(np.linalg.norm(left) * np.linalg.norm(right), eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--generation_seed", type=int, default=53001)
    parser.add_argument("--energy_config", required=True)
    parser.add_argument("--min_conflict_fraction", type=float, default=0.25)
    parser.add_argument("--min_retained_ratio", type=float, default=0.10)
    parser.add_argument("--min_rank_flip", type=float, default=0.10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default=f"{ROOT}/outputs/rc_energy/gradient_conflict_audit.json")
    args = parser.parse_args()
    if args.groups < 2 or args.K < 2:
        raise ValueError("gradient audit requires groups >= 2 and K >= 2")
    if args.deterministic:
        set_determinism(args.generation_seed)

    device = "cuda"
    tokenizer = load_tokenizer(device)
    model = load_world_model(device, "base", dtype=torch.float32)
    model.config.use_cache = False
    model.train()
    action_ranges = load_action_ranges(device)
    metrics = Metrics(device)
    windows = sample_windows(list_episodes(), args.groups, seed=args.window_seed)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    rows = []
    started = time.time()
    print(f"[setup] groups={len(windows)} K={args.K} parameters={len(parameters)}", flush=True)

    for index, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, device)
        ground_truth = frames[CTX]
        prompt = build_prompt(tokenizer, frames, actions, action_ranges)
        with torch.no_grad():
            candidates = generate_candidates(
                model,
                prompt,
                args.K,
                seed=args.generation_seed + index,
            )
            gt_indices = encode_indices(tokenizer, ground_truth.unsqueeze(0))
            images = decode_tokens(tokenizer, candidates)
            primary_reward = gt_reward(
                "a0faithful_tok",
                metrics,
                tokenizer,
                candidates,
                images,
                ground_truth,
                gt_indices,
            )
            auxiliary_reward = gt_reward(
                "rc_energy_certified",
                metrics,
                tokenizer,
                candidates,
                images,
                ground_truth,
                gt_indices,
                energy_config_path=args.energy_config,
            )
            primary_advantage, _ = shape_advantage(primary_reward, mode="gt_only")
            auxiliary_advantage, _ = shape_advantage(auxiliary_reward, mode="gt_only")

        logp_sum, _ = seq_logp(model, prompt, candidates)
        primary_tensor = torch.as_tensor(primary_advantage, device=device, dtype=torch.float32)
        auxiliary_tensor = torch.as_tensor(auxiliary_advantage, device=device, dtype=torch.float32)
        primary_loss = -(primary_tensor * logp_sum).mean()
        auxiliary_loss = -(auxiliary_tensor * logp_sum).mean()
        primary_gradients = torch.autograd.grad(
            primary_loss, parameters, retain_graph=True, allow_unused=True
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss, parameters, allow_unused=True
        )
        dot, primary_norm_sq, auxiliary_norm_sq = gradient_inner_products(
            primary_gradients, auxiliary_gradients
        )
        stats = projection_statistics(dot, primary_norm_sq, auxiliary_norm_sq)
        stats.update({
            "episode": os.path.basename(path),
            "start": int(start),
            "advantage_cosine": _cosine(primary_advantage, auxiliary_advantage),
            "rank_flip_fraction": _pair_flip_fraction(primary_reward, auxiliary_reward),
            "same_top": bool(np.argmax(primary_reward) == np.argmax(auxiliary_reward)),
        })
        rows.append(stats)
        del primary_gradients, auxiliary_gradients, primary_loss, auxiliary_loss, logp_sum
        torch.cuda.empty_cache()
        done = index + 1
        elapsed = time.time() - started
        eta = elapsed / done * (len(windows) - done)
        print(
            f"[audit] {_bar(done/len(windows))} {done}/{len(windows)} "
            f"gradCos={stats['cosine']:+.3f} conflict={int(stats['conflict'])} "
            f"retained={stats['retained_auxiliary_ratio']:.3f} "
            f"rankFlip={stats['rank_flip_fraction']:.3f} "
            f"elapsed={_hms(elapsed)} eta={_hms(eta)}",
            flush=True,
        )

    conflict_fraction = float(np.mean([row["conflict"] for row in rows]))
    retained = np.asarray([row["retained_auxiliary_ratio"] for row in rows])
    rank_flip = np.asarray([row["rank_flip_fraction"] for row in rows])
    gradient_cosine = np.asarray([row["cosine"] for row in rows])
    green = (
        conflict_fraction >= args.min_conflict_fraction
        and float(np.median(retained)) >= args.min_retained_ratio
        and float(np.median(rank_flip)) >= args.min_rank_flip
        and np.isfinite(gradient_cosine).all()
    )
    summary = {
        "conflict_fraction": conflict_fraction,
        "gradient_cosine_mean": float(gradient_cosine.mean()),
        "gradient_cosine_median": float(np.median(gradient_cosine)),
        "retained_auxiliary_ratio_mean": float(retained.mean()),
        "retained_auxiliary_ratio_median": float(np.median(retained)),
        "rank_flip_fraction_mean": float(rank_flip.mean()),
        "rank_flip_fraction_median": float(np.median(rank_flip)),
        "same_top_fraction": float(np.mean([row["same_top"] for row in rows])),
        "verdict": "GREEN" if green else "RED",
    }
    payload = {"args": vars(args), "summary": summary, "groups": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\n=== Fidelity Gradient Conflict Audit ===", flush=True)
    for key, value in summary.items():
        print(f"{key}={value}", flush=True)
    print(f"saved {args.out}\nFIDELITY_GRADIENT_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
