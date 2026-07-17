"""Paired raw-vs-RC single-step GRPO on public IRIS Atari checkpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dor.adapters.iris_atari import (
    load_iris,
    load_iris_window_npz,
    post_quant_latent_reward,
    reachable_target,
    sample_next_frame,
    teacher_forced_next_frame_logp,
)
from dor.grpo import _bar, _hms, set_determinism
from dor.gradient_constraints import (
    accumulate_parameter_gradients,
    project_to_primary_progress,
)
from dor.kl import sampled_kl_penalty
from dor.metrics import Metrics


def _load_and_split_manifest(path: str | Path, eval_episodes: int, split_seed: int):
    payload = json.loads(Path(path).read_text())
    entries = list(payload["entries"])
    episodes = sorted({int(entry["episode"]) for entry in entries})
    if not 1 <= eval_episodes < len(episodes):
        raise ValueError("eval_episodes must leave at least one training episode")
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(episodes)[rng.permutation(len(episodes))]
    eval_ids = sorted(int(value) for value in shuffled[:eval_episodes])
    train_ids = sorted(int(value) for value in shuffled[eval_episodes:])
    train = [entry for entry in entries if int(entry["episode"]) in train_ids]
    evaluate = [entry for entry in entries if int(entry["episode"]) in eval_ids]
    if not train or not evaluate:
        raise ValueError("empty train/eval split")
    return payload, train, evaluate, train_ids, eval_ids


def _fixed_schedule(n_entries: int, steps: int, batch_windows: int, seed: int):
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, n_entries, size=(steps, batch_windows), dtype=np.int64)
    return schedule, hashlib.sha256(schedule.tobytes()).hexdigest()


def _frame_reward(metrics: Metrics, predictions, raw_target, rc_target, reward: str):
    target = raw_target if reward == "raw" else rc_target
    value = metrics.eval_batch(predictions, target)
    return -(np.asarray(value["lpips"]) + np.asarray(value["mse"]))


@torch.inference_mode()
def evaluate(model, tokenizer, metrics, entries, *, K: int, seed: int, device) -> dict:
    rows = {name: [] for name in ("lpips", "mse", "psnr", "ssim")}
    hamming, latent_rms = [], []
    model.eval()
    for index, entry in enumerate(entries):
        window = load_iris_window_npz(entry["window_npz"], device=device)
        rollout, target_tokens = sample_next_frame(
            tokenizer,
            model,
            window,
            group_size=K,
            seed=seed + 1009 * index,
        )
        quality = metrics.eval_batch(rollout.decoded, window.frames[-1])
        for name in rows:
            rows[name].append(float(np.mean(quality[name])))
        hamming.append(float((rollout.tokens != target_tokens.unsqueeze(0)).float().mean().cpu()))
        latent_rms.append(float((-post_quant_latent_reward(
            tokenizer, rollout.tokens, target_tokens
        )).mean().cpu()))
    result = {name: float(np.mean(values)) for name, values in rows.items()}
    result["token_hamming"] = float(np.mean(hamming))
    result["latent_rms"] = float(np.mean(latent_rms))
    return result


def _log_eval(log: dict, step: int, result: dict, reward: str):
    log["step"].append(int(step))
    for name, value in result.items():
        log[f"eval_{name}"].append(float(value))
    print(
        f"[{reward}] step={step} LPIPS={result['lpips']:.5f} "
        f"MSE={result['mse']:.6f} PSNR={result['psnr']:.3f} "
        f"SSIM={result['ssim']:.5f} LatRMS={result['latent_rms']:.5f} "
        f"TokHam={result['token_hamming']:.5f}",
        flush=True,
    )


def train_one(reward: str, seed: int, args) -> dict:
    if args.deterministic:
        set_determinism(seed)
    device = torch.device(args.device)
    manifest, train_entries, eval_entries, train_ids, eval_ids = _load_and_split_manifest(
        args.manifest, args.eval_episodes, args.split_seed
    )
    schedule, schedule_hash = _fixed_schedule(
        len(train_entries), args.steps, args.batch_windows, args.data_seed
    )
    tokenizer, policy = load_iris(
        args.upstream,
        args.checkpoint,
        action_vocab_size=int(manifest["action_vocab_size"]),
        device=device,
    )
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    metrics = Metrics(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    metric_names = ("lpips", "mse", "psnr", "ssim", "token_hamming", "latent_rms")
    log = {
        "step": [],
        "train_policy_loss": [],
        "train_kl": [],
        "train_grad_norm": [],
        "train_reward": [],
        "protocol": {
            "environment": manifest["environment"],
            "train_episodes": train_ids,
            "eval_episodes": eval_ids,
            "window_stride": manifest.get("window_stride"),
            "context_schedule_sha256": schedule_hash,
            "context_schedule_indices": schedule.tolist(),
            "policy_mode": "eval for sampling and teacher-forced log-prob",
            "evaluator": "raw GT for every reported metric",
        },
    }
    for name in metric_names:
        log[f"eval_{name}"] = []
    if reward == "ra_rc":
        for name in (
            "train_raw_policy_loss", "train_rc_policy_loss",
            "train_constraint_active", "train_projection_coefficient",
            "train_gradient_cosine", "train_preferred_progress_ratio",
            "train_projected_progress_ratio",
        ):
            log[name] = []
    _log_eval(
        log,
        0,
        evaluate(
            policy, tokenizer, metrics, eval_entries,
            K=args.eval_K, seed=args.eval_seed, device=device,
        ),
        reward,
    )

    started = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        policy_total = kl_total = reward_total = 0.0
        raw_policy_total = rc_policy_total = 0.0
        projection_rows = []
        for ordinal, selected_index in enumerate(schedule[step - 1]):
            entry = train_entries[int(selected_index)]
            window = load_iris_window_npz(entry["window_npz"], device=device)
            with torch.inference_mode():
                rollout, target_tokens = sample_next_frame(
                    tokenizer,
                    policy,
                    window,
                    group_size=args.K,
                    seed=(
                        seed * 1_000_003 + step * 10_007
                        + int(selected_index) * 101 + ordinal
                    ),
                )
                rc_target = reachable_target(tokenizer, target_tokens)
                if reward == "ra_rc":
                    raw_rewards = _frame_reward(
                        metrics, rollout.decoded, window.frames[-1], rc_target, "raw"
                    )
                    rc_rewards = _frame_reward(
                        metrics, rollout.decoded, window.frames[-1], rc_target, "rc"
                    )
                    raw_advantage = (
                        raw_rewards - raw_rewards.mean()
                    ) / (raw_rewards.std() + 1e-6)
                    rc_advantage = (
                        rc_rewards - rc_rewards.mean()
                    ) / (rc_rewards.std() + 1e-6)
                    rewards = rc_rewards
                else:
                    rewards = _frame_reward(
                        metrics, rollout.decoded, window.frames[-1], rc_target, reward
                    )
                    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            logp = teacher_forced_next_frame_logp(tokenizer, policy, window, rollout)
            if step == 1 and ordinal == 0:
                if rollout.sample_logp is None:
                    raise RuntimeError("IRIS rollout did not retain generation log-prob")
                logp_error = float((
                    logp.detach() - rollout.sample_logp.detach()
                ).abs().max().cpu())
                log["protocol"]["teacher_forced_sample_logp_max_error"] = logp_error
                if logp_error > args.logp_tolerance:
                    raise RuntimeError(
                        f"teacher-forced/sample log-prob mismatch {logp_error:.3e} "
                        f"> tolerance {args.logp_tolerance:.3e}"
                    )
            with torch.no_grad():
                reference_logp = teacher_forced_next_frame_logp(
                    tokenizer, reference, window, rollout
                )
            kl = sampled_kl_penalty(logp, reference_logp, "low_var_kl").mean()
            if reward == "ra_rc":
                raw_advantage_t = torch.as_tensor(
                    raw_advantage, device=device, dtype=torch.float32
                )
                rc_advantage_t = torch.as_tensor(
                    rc_advantage, device=device, dtype=torch.float32
                )
                raw_policy_loss = -(raw_advantage_t[:, None] * logp).mean()
                rc_policy_loss = -(rc_advantage_t[:, None] * logp).mean()
                raw_gradients = torch.autograd.grad(
                    raw_policy_loss, parameters, retain_graph=True, allow_unused=True
                )
                rc_gradients = torch.autograd.grad(
                    rc_policy_loss,
                    parameters,
                    retain_graph=bool(args.kl > 0.0),
                    allow_unused=True,
                )
                projected, projection = project_to_primary_progress(
                    raw_gradients, rc_gradients
                )
                accumulate_parameter_gradients(
                    parameters, projected, scale=1.0 / args.batch_windows
                )
                if args.kl > 0.0:
                    (args.kl * kl / args.batch_windows).backward()
                raw_policy_total += float(raw_policy_loss.detach().cpu())
                rc_policy_total += float(rc_policy_loss.detach().cpu())
                policy_total += float(rc_policy_loss.detach().cpu())
                projection_rows.append(projection)
            else:
                advantage_t = torch.as_tensor(
                    advantage, device=device, dtype=torch.float32
                )
                policy_loss = -(advantage_t[:, None] * logp).mean()
                loss = policy_loss + args.kl * kl
                (loss / args.batch_windows).backward()
                policy_total += float(policy_loss.detach().cpu())
            kl_total += float(kl.detach().cpu())
            reward_total += float(np.mean(rewards))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            policy.parameters(), args.grad_clip
        ).detach().cpu())
        optimizer.step()
        policy_mean = policy_total / args.batch_windows
        kl_mean = kl_total / args.batch_windows
        reward_mean = reward_total / args.batch_windows
        log["train_policy_loss"].append(policy_mean)
        log["train_kl"].append(kl_mean)
        log["train_grad_norm"].append(grad_norm)
        log["train_reward"].append(reward_mean)
        if reward == "ra_rc":
            log["train_raw_policy_loss"].append(raw_policy_total / args.batch_windows)
            log["train_rc_policy_loss"].append(rc_policy_total / args.batch_windows)
            log["train_constraint_active"].append(float(np.mean([
                row["constraint_active"] for row in projection_rows
            ])))
            for key, log_name in (
                ("coefficient", "train_projection_coefficient"),
                ("gradient_cosine", "train_gradient_cosine"),
                ("preferred_progress_ratio", "train_preferred_progress_ratio"),
                ("projected_progress_ratio", "train_projected_progress_ratio"),
            ):
                log[log_name].append(float(np.mean([row[key] for row in projection_rows])))
        elapsed = time.time() - started
        print(
            f"[{reward}] {_bar(step / args.steps)} {step}/{args.steps} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed / step * (args.steps-step))} "
            f"pg={policy_mean:.5f} kl={kl_mean:.3e} gn={grad_norm:.3f} "
            f"reward={reward_mean:.5f}"
            + (
                f" active={np.mean([row['constraint_active'] for row in projection_rows]):.2f} "
                f"cos={np.mean([row['gradient_cosine'] for row in projection_rows]):+.3f} "
                f"rawProg={np.mean([row['projected_progress_ratio'] for row in projection_rows]):.3f}"
                if reward == "ra_rc" else ""
            ),
            flush=True,
        )
        if step % args.eval_every == 0 or step == args.steps:
            _log_eval(
                log,
                step,
                evaluate(
                    policy, tokenizer, metrics, eval_entries,
                    K=args.eval_K, seed=args.eval_seed, device=device,
                ),
                reward,
            )
    if args.save_checkpoints:
        destination = Path(args.out_dir) / "ckpt" / f"iris_{reward}_s{seed}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": policy.state_dict(), "args": vars(args)}, destination)
    del policy, reference, tokenizer, metrics
    torch.cuda.empty_cache()
    return log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--rewards", default="raw,rc")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--eval_K", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_windows", type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=4)
    parser.add_argument("--split_seed", type=int, default=9413)
    parser.add_argument("--data_seed", type=int, default=9414)
    parser.add_argument("--eval_seed", type=int, default=9415)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--kl", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--logp_tolerance", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rewards = [item.strip() for item in args.rewards.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if set(rewards) - {"raw", "rc", "ra_rc"}:
        raise ValueError("rewards must be raw, rc, and/or ra_rc")
    if args.K < 2 or args.eval_K < 2 or args.steps < 1 or args.batch_windows < 1:
        raise ValueError("K/eval_K must be >=2 and training sizes must be positive")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(reward, seed) for seed in seeds for reward in rewards]
    completed, started = 0, time.time()
    for reward, seed in jobs:
        output = output_dir / f"sweep_{reward}_s{seed}.json"
        if output.exists():
            completed += 1
            print(f"[resume] {output} already exists", flush=True)
            continue
        elapsed = time.time() - started
        eta = elapsed / max(completed, 1) * (len(jobs) - completed)
        print(
            f"\n===== IRIS C1 SWEEP {_bar(completed/len(jobs))} "
            f"{completed}/{len(jobs)} next={reward}/s{seed} "
            f"elapsed={_hms(elapsed)} eta={_hms(eta)} =====",
            flush=True,
        )
        run = train_one(reward, seed, args)
        output.write_text(json.dumps({"args": vars(args), "run": run}, indent=2) + "\n")
        completed += 1
        print(f"[done] saved {output}", flush=True)
    print(
        f"\n[sweep done] {completed}/{len(jobs)} in {_hms(time.time()-started)}"
        "\nIRIS_C1_GRPO_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
