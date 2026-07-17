"""P0 wiring audit for the public iVideoGPT VP2-RoboSuite checkpoint.

This is deliberately not a training script. It verifies the data/action alignment,
token framing, sampled-token teacher-forced log-probabilities, and raw/RC reward
plumbing before any external GRPO experiment is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dor.adapters.ivideogpt_vp2 import (
    CONTEXT_LENGTH,
    decoded_ground_truth,
    frame_rewards,
    load_ivideogpt,
    load_vp2_window,
    load_vp2_window_npz,
    sample_rollout,
    teacher_forced_dynamics_logp,
    tokenize_ground_truth,
)
from dor.metrics import Metrics


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hdf5")
    source.add_argument("--window_npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--episode", default="demo_1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.window_npz:
        window = load_vp2_window_npz(args.window_npz, device=device)
        if window.horizon != args.horizon:
            raise ValueError(
                f"exported window has horizon={window.horizon}, requested --horizon={args.horizon}"
            )
    else:
        window = load_vp2_window(args.hdf5, args.episode, args.start, args.horizon, device=device)
    tokenizer, model = load_ivideogpt(args.upstream, args.checkpoint, horizon=args.horizon, device=device)
    ground_truth_tokens = tokenize_ground_truth(tokenizer, window)
    reachable = decoded_ground_truth(tokenizer, ground_truth_tokens)
    rollout = sample_rollout(
        tokenizer,
        model,
        ground_truth_tokens,
        window.actions,
        horizon=args.horizon,
        group_size=args.K,
        seed=args.seed,
    )
    metrics = Metrics(device)
    rewards = frame_rewards(metrics, rollout, window, reachable)

    # P0 synthetic identity: feeding the same target to both branches must make the
    # two verifiers exactly equivalent up to float32 LPIPS evaluation noise.
    identity_window = window.__class__(window.episode, window.start, reachable[0], window.actions)
    identity = frame_rewards(metrics, rollout, identity_window, reachable)
    identity_error = float(np.max(np.abs(identity["raw"] - identity["rc"])))

    model.zero_grad(set_to_none=True)
    token_logp = teacher_forced_dynamics_logp(model, rollout, window.actions)
    (-token_logp.mean()).backward()
    grad_norm_sq = 0.0
    trainable_with_grad = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            trainable_with_grad += 1
            grad_norm_sq += float(parameter.grad.detach().float().square().sum().item())
    if not np.isfinite(grad_norm_sq) or grad_norm_sq <= 0:
        raise RuntimeError("teacher-forced dynamics log-prob produced no finite gradient")
    if identity_error > 1e-6:
        raise RuntimeError(f"raw/RC identity check failed: max reward difference {identity_error:.3e}")

    checkpoint = Path(args.checkpoint)
    payload = {
        "episode": window.episode,
        "start": window.start,
        "context_length": CONTEXT_LENGTH,
        "horizon": args.horizon,
        "group_size": args.K,
        "frames_shape": list(window.frames.shape),
        "actions_shape": list(window.actions.shape),
        "action_min": window.actions.amin(dim=0).detach().cpu().tolist(),
        "action_max": window.actions.amax(dim=0).detach().cpu().tolist(),
        "tokens_shape": list(ground_truth_tokens.shape),
        "candidate_tokens_shape": list(rollout.full_tokens.shape),
        "dynamics_tokens_shape": list(rollout.dynamics_tokens.shape),
        "decoded_shape": list(rollout.decoded.shape),
        "token_logp_shape": list(token_logp.shape),
        "token_logp_finite": bool(torch.isfinite(token_logp).all().item()),
        "backward_grad_norm": grad_norm_sq ** 0.5,
        "parameters_with_grad": trainable_with_grad,
        "identity_max_reward_error": identity_error,
        "raw_reward_mean": float(rewards["raw"].mean()),
        "rc_reward_mean": float(rewards["rc"].mean()),
        "raw_reward_std": float(rewards["raw"].std()),
        "rc_reward_std": float(rewards["rc"].std()),
        "checkpoint_transformer_sha256": sha256(checkpoint / "transformer" / "model.safetensors"),
        "checkpoint_tokenizer_sha256": sha256(checkpoint / "tokenizer" / "diffusion_pytorch_model.safetensors"),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("VP2_IVIDEOGPT_SMOKE_OK")


if __name__ == "__main__":
    main()
