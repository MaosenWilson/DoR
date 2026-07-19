#!/usr/bin/env python3
"""GPU smoke test for the official RoboNet iVideoGPT checkpoint and sample."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dor.adapters.ivideogpt_robonet import (
    ROBONET_ACTION_DIM,
    load_robonet_ivideogpt,
    load_robonet_window_npz,
)
from dor.adapters.ivideogpt_vp2 import (
    decoded_ground_truth,
    sample_rollout,
    teacher_forced_dynamics_logp,
    tokenize_ground_truth,
)


def _stage(name: str, started: float) -> None:
    print(f"[smoke] {name} elapsed={time.monotonic() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.K < 2:
        raise ValueError("K must be at least two")

    started = time.monotonic()
    device = torch.device(args.device)
    window = load_robonet_window_npz(
        args.sample, horizon=args.horizon, device=device
    )
    assert window.actions.shape == (args.horizon + 2, ROBONET_ACTION_DIM)
    assert torch.count_nonzero(window.actions[-1]).item() == 0
    _stage("data/action alignment OK", started)

    tokenizer, model = load_robonet_ivideogpt(
        args.upstream, args.checkpoint, horizon=args.horizon, device=device
    )
    _stage("checkpoint/action head OK", started)

    with torch.inference_mode():
        ground_truth = tokenize_ground_truth(tokenizer, window)
        reachable = decoded_ground_truth(tokenizer, ground_truth)
        rollout = sample_rollout(
            tokenizer,
            model,
            ground_truth,
            window.actions,
            horizon=args.horizon,
            group_size=args.K,
            seed=args.seed,
        )
    if rollout.decoded.shape != (args.K, args.horizon + 2, 3, 64, 64):
        raise RuntimeError(f"unexpected rollout shape {tuple(rollout.decoded.shape)}")
    if reachable.shape != (1, args.horizon + 2, 3, 64, 64):
        raise RuntimeError(f"unexpected reachable shape {tuple(reachable.shape)}")
    _stage("tokenize/generate/decode OK", started)

    model.zero_grad(set_to_none=True)
    logp = teacher_forced_dynamics_logp(model, rollout, window.actions)
    loss = -logp.mean()
    loss.backward()
    gradients = [
        parameter.grad.detach().float()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("teacher-forced gradient path is missing or non-finite")
    grad_norm = float(torch.sqrt(sum(value.square().sum() for value in gradients)).cpu())
    reconstruction_mse = float((reachable - window.frames.unsqueeze(0)).square().mean().cpu())
    candidate_mse = float(
        (rollout.decoded[:, 2:] - window.frames[2:].unsqueeze(0)).square().mean().cpu()
    )
    if not np.isfinite([float(loss.detach().cpu()), grad_norm, reconstruction_mse, candidate_mse]).all():
        raise RuntimeError("smoke diagnostics contain non-finite values")
    print(
        f"[smoke] loss={float(loss.detach().cpu()):.6f} grad_norm={grad_norm:.4f} "
        f"recon_mse={reconstruction_mse:.6f} candidate_mse={candidate_mse:.6f}",
        flush=True,
    )
    print("ROBONET_IVIDEOGPT_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
