"""Audit whether RC contributes a usable raw-fidelity-safe gradient direction.

This is a zero-training gate.  Raw-GT GRPO is the anchor objective and the RC
increment is ``g_rc - g_raw``.  The audit asks whether that increment is both
non-trivial and recoverable after projecting away its first-order conflict with
the held-out evaluator-aligned raw objective.
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

from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.gradient_constraints import (
    correction_projection_statistics,
    gradient_inner_products,
)
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
    dl = (left[:, None] - left[None, :])[upper]
    dr = (right[:, None] - right[None, :])[upper]
    valid = (dl != 0.0) & (dr != 0.0)
    return float(np.mean(dl[valid] * dr[valid] < 0.0)) if np.any(valid) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--generation_seed", type=int, default=57001)
    parser.add_argument("--min_conflict_fraction", type=float, default=0.50)
    parser.add_argument("--min_retained_ratio", type=float, default=0.10)
    parser.add_argument("--min_rank_flip", type=float, default=0.05)
    parser.add_argument("--min_safe_cosine_gain", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--out",
        default=f"{ROOT}/outputs/analysis/rc_correction_gradient_gate.json",
    )
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
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    rows = []
    started = time.time()
    print(f"[setup] groups={len(windows)} K={args.K} parameters={len(parameters)}", flush=True)

    for index, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, device)
        ground_truth = frames[CTX]
        prompt = build_prompt(tokenizer, frames, actions, action_ranges)
        with torch.no_grad():
            candidates = generate_candidates(
                model, prompt, args.K, seed=args.generation_seed + index
            )
            gt_indices = encode_indices(tokenizer, ground_truth.unsqueeze(0))
            images = decode_tokens(tokenizer, candidates)
            raw_reward = gt_reward(
                "a0faithful", metrics, tokenizer, candidates, images,
                ground_truth, gt_indices,
            )
            rc_reward = gt_reward(
                "a0faithful_tok", metrics, tokenizer, candidates, images,
                ground_truth, gt_indices,
            )
            raw_advantage, _ = shape_advantage(raw_reward, mode="gt_only")
            rc_advantage, _ = shape_advantage(rc_reward, mode="gt_only")

        logp_sum, _ = seq_logp(model, prompt, candidates)
        raw_tensor = torch.as_tensor(raw_advantage, device=device, dtype=torch.float32)
        rc_tensor = torch.as_tensor(rc_advantage, device=device, dtype=torch.float32)
        raw_loss = -(raw_tensor * logp_sum).mean()
        rc_loss = -(rc_tensor * logp_sum).mean()
        raw_gradients = torch.autograd.grad(
            raw_loss, parameters, retain_graph=True, allow_unused=True
        )
        rc_gradients = torch.autograd.grad(
            rc_loss, parameters, allow_unused=True
        )
        raw_rc_dot, raw_norm_sq, rc_norm_sq = gradient_inner_products(
            raw_gradients, rc_gradients
        )
        stats = correction_projection_statistics(
            raw_norm_sq, rc_norm_sq, raw_rc_dot
        )
        stats.update({
            "episode": os.path.basename(path),
            "start": int(start),
            "rank_flip_fraction": _pair_flip_fraction(raw_reward, rc_reward),
            "same_top": bool(np.argmax(raw_reward) == np.argmax(rc_reward)),
        })
        rows.append(stats)
        del raw_gradients, rc_gradients, raw_loss, rc_loss, logp_sum
        torch.cuda.empty_cache()
        done = index + 1
        elapsed = time.time() - started
        print(
            f"[audit] {_bar(done / len(windows))} {done}/{len(windows)} "
            f"rcCos={stats['corrected_primary_cosine']:+.3f} "
            f"conflict={int(stats['conflict'])} "
            f"retained={stats['retained_auxiliary_ratio']:.3f} "
            f"safeCos={stats['safe_primary_cosine']:+.3f} "
            f"rankFlip={stats['rank_flip_fraction']:.3f} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (len(windows)-done))}",
            flush=True,
        )

    conflict_fraction = float(np.mean([row["conflict"] for row in rows]))
    retained = np.asarray([row["retained_auxiliary_ratio"] for row in rows])
    rank_flip = np.asarray([row["rank_flip_fraction"] for row in rows])
    rc_cosine = np.asarray([row["corrected_primary_cosine"] for row in rows])
    safe_cosine = np.asarray([row["safe_primary_cosine"] for row in rows])
    safe_gain = safe_cosine - rc_cosine
    green = (
        conflict_fraction >= args.min_conflict_fraction
        and float(np.median(retained)) >= args.min_retained_ratio
        and float(np.median(rank_flip)) >= args.min_rank_flip
        and float(np.median(safe_gain)) >= args.min_safe_cosine_gain
        and np.isfinite(safe_cosine).all()
    )
    summary = {
        "conflict_fraction": conflict_fraction,
        "rc_primary_cosine_mean": float(rc_cosine.mean()),
        "safe_primary_cosine_mean": float(safe_cosine.mean()),
        "safe_cosine_gain_median": float(np.median(safe_gain)),
        "retained_correction_ratio_median": float(np.median(retained)),
        "rank_flip_fraction_median": float(np.median(rank_flip)),
        "same_top_fraction": float(np.mean([row["same_top"] for row in rows])),
        "verdict": "GREEN" if green else "RED",
    }
    payload = {"args": vars(args), "summary": summary, "groups": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\n=== RC Correction Gradient Audit ===", flush=True)
    for key, value in summary.items():
        print(f"{key}={value}", flush=True)
    print(f"saved {args.out}\nRC_CORRECTION_GRADIENT_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
