"""P2: paired raw/RC x sequence/temporal-return GRPO on public VP2-iVideoGPT.

All four arms use the same frozen episode manifests, candidate sampling, optimizer,
and KL anchor.  ``raw`` versus ``rc`` changes only the target in -(MSE + LPIPS);
``seq`` versus ``return`` changes only how those identical frame rewards attach to
sampled future-token log-probabilities.  Evaluation is always against raw GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dor.adapters.ivideogpt_vp2 import (
    DYNAMICS_GRID_TOKENS,
    decoded_ground_truth,
    frame_rewards,
    future_dynamics_latent_reward,
    future_dynamics_tokens,
    load_ivideogpt,
    load_vp2_window_npz,
    sample_rollout,
    teacher_forced_dynamics_logp,
    tokenize_ground_truth,
)
from dor.grpo import _bar, _hms, set_determinism
from dor.gradient_constraints import (
    accumulate_parameter_gradients,
    project_to_primary_progress,
)
from dor.kl import sampled_kl_penalty
from dor.metrics import Metrics
from dor.temporal_credit import (
    conservative_adaptive_temporal_advantages,
    gae_frame_advantages,
    influence_adaptive_delayed_advantages,
    influence_adaptive_temporal_advantages,
    normalize_by_horizon,
    reachability_consistent_temporal_scores,
    scale_equalized_temporal_return_advantages,
    temporal_return_advantages,
)


UATR_REWARDS = {"uatr", "uatr_shuffled", "uatr2", "uatr2_shuffled"}
ANCHORED_REWARDS = {"ra_rc", *UATR_REWARDS}


def _read_manifest(path: str | Path, horizon: int) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    if int(payload["horizon"]) != int(horizon):
        raise ValueError(f"manifest {path} does not have horizon={horizon}")
    entries = list(payload["entries"])
    if not entries:
        raise ValueError(f"manifest {path} is empty")
    for entry in entries:
        if not Path(entry["window_npz"]).is_file():
            raise FileNotFoundError(entry["window_npz"])
    return entries


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixed_context_schedule(n_entries: int, steps: int, batch_windows: int, seed: int):
    """Create one auditable context schedule shared by every arm and policy seed."""
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, n_entries, size=(steps, batch_windows), dtype=np.int64)
    digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    return schedule, digest


def _assert_episode_disjoint(train_entries: list[dict], eval_entries: list[dict]) -> None:
    train_episodes = {str(entry["episode"]) for entry in train_entries}
    eval_episodes = {str(entry["episode"]) for entry in eval_entries}
    overlap = sorted(train_episodes & eval_episodes)
    if overlap:
        raise ValueError(f"train/eval manifests overlap on episodes: {overlap}")


def _advantages(reward_frame: np.ndarray, credit: str, gamma: float) -> tuple[np.ndarray, bool]:
    """Return GRPO advantages and whether they are frame-block specific."""
    if credit == "seq":
        scalar = reward_frame.mean(axis=1)
        return (scalar - scalar.mean()) / (scalar.std() + 1e-6), False
    if credit == "return":
        return temporal_return_advantages(reward_frame, gamma), True
    if credit == "return_eq":
        return scale_equalized_temporal_return_advantages(reward_frame, gamma), True
    if credit.startswith("gae"):
        # "gae0.9" -> critic-free GAE frame-block advantage with lam=0.9.
        # lam=1 == return, lam=0 == frame-only per horizon.
        lam = float(credit[3:]) if len(credit) > 3 else 1.0
        return gae_frame_advantages(reward_frame, gamma, lam), True
    raise ValueError(f"unknown credit assignment {credit!r}")


def _select_reward(frame_values: dict[str, np.ndarray], reward: str, rc_mix: float) -> np.ndarray:
    """Select raw/full-RC reward or a raw-fidelity-constrained RC projection."""
    if reward in ("raw", "rc"):
        return frame_values[reward]
    if reward == "rc_mix":
        return (1.0 - rc_mix) * frame_values["raw"] + rc_mix * frame_values["rc"]
    raise ValueError(f"unknown reward {reward!r}")


def _policy_loss(logp: torch.Tensor, advantage: np.ndarray, blockwise: bool, device) -> torch.Tensor:
    advantage_t = torch.as_tensor(advantage, device=device, dtype=torch.float32)
    if blockwise:
        return -(advantage_t[:, :, None] * logp).mean()
    return -(advantage_t[:, None, None] * logp).mean()


def _load_catr_coefficients(path: str | Path, horizon: int) -> tuple[np.ndarray, str]:
    if not path:
        raise ValueError("reward=catr requires --catr_config")
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if int(payload["horizon"]) != int(horizon):
        raise ValueError("CATR calibration horizon does not match training horizon")
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    if coefficients.shape != (horizon,):
        raise ValueError("CATR calibration must provide one coefficient per horizon")
    if np.any(coefficients < 0.0) or np.any(coefficients > 1.0):
        raise ValueError("CATR coefficients must lie in [0,1]")
    return coefficients, hashlib.sha256(raw).hexdigest()


def _load_uatr_coefficients(path: str | Path, horizon: int) -> tuple[np.ndarray, str]:
    if not path:
        raise ValueError("reward=uatr/uatr_shuffled requires --adaptive_config")
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if int(payload["horizon"]) != int(horizon):
        raise ValueError("UATR calibration horizon does not match training horizon")
    if payload.get("verdict") != "PROVISIONAL-GREEN":
        raise ValueError("UATR training is forbidden unless calibration Gate A is green")
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    if coefficients.shape != (horizon,):
        raise ValueError("UATR calibration must provide one coefficient per horizon")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("UATR coefficients must be finite")
    if np.any(coefficients < 0.0) or np.any(coefficients > 1.0):
        raise ValueError("UATR coefficients must lie in [0,1]")
    if np.count_nonzero(coefficients > 0.0) < 2:
        raise ValueError("UATR calibration must activate at least two frame blocks")
    return coefficients, hashlib.sha256(raw).hexdigest()


def _aggregate(values: list[float], episodes: list[str], mode: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    episode_array = np.asarray(episodes).astype(str)
    if array.ndim != 1 or episode_array.shape != array.shape or not len(array):
        raise ValueError("evaluation values and episodes must be aligned non-empty vectors")
    if mode == "window_micro":
        return float(array.mean())
    if mode != "episode_macro":
        raise ValueError(f"unknown evaluation aggregation {mode!r}")
    return float(np.mean([
        array[episode_array == episode].mean()
        for episode in np.unique(episode_array)
    ]))


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    metrics,
    entries,
    *,
    horizon: int,
    group_size: int,
    device,
    seed: int,
    action_dim: int = 4,
    draws: int = 1,
    aggregation: str = "window_micro",
) -> dict:
    if draws < 1:
        raise ValueError("evaluation draws must be positive")
    rows = {"lpips": [], "mse": [], "psnr": [], "ssim": []}
    last_rows = {"lpips": [], "mse": []}
    token_rows = {"token_hamming": [], "latent_rms": []}
    token_last_rows = {"token_hamming": [], "latent_rms": []}
    row_episodes, last_episodes = [], []
    token_episodes, token_last_episodes = [], []
    model.eval()
    for draw in range(draws):
        for index, entry in enumerate(entries):
            episode = str(entry["episode"])
            window = load_vp2_window_npz(entry["window_npz"], action_dim=action_dim, device=device)
            ground_truth = tokenize_ground_truth(tokenizer, window)
            rollout = sample_rollout(
                tokenizer, model, ground_truth, window.actions,
                horizon=horizon, group_size=group_size,
                seed=seed + draw * 1_000_003 + index,
            )
            target_dynamics = future_dynamics_tokens(ground_truth, horizon)[0]
            token_hamming = (
                rollout.dynamics_tokens != target_dynamics.unsqueeze(0)
            ).float().mean(dim=-1).detach().cpu().numpy()
            latent_rms = -future_dynamics_latent_reward(
                tokenizer,
                rollout.dynamics_tokens,
                target_dynamics,
                projected=True,
            ).detach().cpu().numpy()
            for frame in range(horizon):
                token_rows["token_hamming"].append(float(np.mean(token_hamming[:, frame])))
                token_rows["latent_rms"].append(float(np.mean(latent_rms[:, frame])))
                token_episodes.append(episode)
                if frame == horizon - 1:
                    token_last_rows["token_hamming"].append(float(np.mean(token_hamming[:, frame])))
                    token_last_rows["latent_rms"].append(float(np.mean(latent_rms[:, frame])))
                    token_last_episodes.append(episode)
            prediction = rollout.decoded[:, 2:]
            for frame in range(horizon):
                quality = metrics.eval_batch(prediction[:, frame], window.frames[2 + frame])
                row_episodes.append(episode)
                for name in rows:
                    score = float(np.mean(quality[name]))
                    rows[name].append(score)
                    if frame == horizon - 1 and name in last_rows:
                        last_rows[name].append(score)
                if frame == horizon - 1:
                    last_episodes.append(episode)
    values = {
        name: _aggregate(value, row_episodes, aggregation)
        for name, value in rows.items()
    }
    values["lpips_last"] = _aggregate(last_rows["lpips"], last_episodes, aggregation)
    values["mse_last"] = _aggregate(last_rows["mse"], last_episodes, aggregation)
    values.update({
        name: _aggregate(value, token_episodes, aggregation)
        for name, value in token_rows.items()
    })
    values["token_hamming_last"] = _aggregate(
        token_last_rows["token_hamming"], token_last_episodes, aggregation
    )
    values["latent_rms_last"] = _aggregate(
        token_last_rows["latent_rms"], token_last_episodes, aggregation
    )
    return values


def _log_eval(log: dict, step: int, value: dict, reward: str, credit: str, rmean: float) -> None:
    log["step"].append(int(step))
    log["reward_mean"].append(float(rmean))
    for name, score in value.items():
        log[f"eval_{name}"].append(float(score))
    print(
        f"[{reward}/{credit}] step={step} LPIPS={value['lpips']:.5f} "
        f"LPIPS-last={value['lpips_last']:.5f} MSE={value['mse']:.6f} "
        f"PSNR={value['psnr']:.3f} SSIM={value['ssim']:.5f} "
        f"LatRMS={value['latent_rms']:.5f} TokHam={value['token_hamming']:.5f}",
        flush=True,
    )


def train_one(reward: str, credit: str, seed: int, args) -> dict:
    if args.deterministic:
        set_determinism(seed)
    device = torch.device(args.device)
    train_entries = _read_manifest(args.train_manifest, args.horizon)
    eval_entries = _read_manifest(args.eval_manifest, args.horizon)
    _assert_episode_disjoint(train_entries, eval_entries)
    schedule, schedule_sha256 = _fixed_context_schedule(
        len(train_entries), args.steps, args.batch_windows, args.data_seed
    )
    catr_coefficients, catr_sha256 = (None, None)
    if reward == "catr":
        catr_coefficients, catr_sha256 = _load_catr_coefficients(
            args.catr_config, args.horizon
        )
    uatr_coefficients, uatr_sha256 = (None, None)
    if reward in UATR_REWARDS:
        uatr_coefficients, uatr_sha256 = _load_uatr_coefficients(
            args.adaptive_config, args.horizon
        )
    tokenizer, policy = load_ivideogpt(args.upstream, args.checkpoint, horizon=args.horizon, action_dim=args.action_dim, device=device)
    _, reference = load_ivideogpt(args.upstream, args.checkpoint, horizon=args.horizon, action_dim=args.action_dim, device=device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    metrics = Metrics(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    log = {"step": [], "reward_mean": []}
    log["protocol"] = {
        "data_seed": int(args.data_seed),
        "train_manifest_sha256": _file_sha256(args.train_manifest),
        "eval_manifest_sha256": _file_sha256(args.eval_manifest),
        "context_schedule_sha256": schedule_sha256,
        "context_schedule_indices": schedule.tolist(),
        "candidate_seed_varies_by": "policy_seed,step,context_index,batch_ordinal",
        "policy_mode": "eval for both sampling and teacher-forced log-prob; gradients remain enabled",
        "train_episodes": sorted({str(entry["episode"]) for entry in train_entries}),
        "eval_episodes": sorted({str(entry["episode"]) for entry in eval_entries}),
        "eval_draws": int(args.eval_draws),
        "eval_aggregation": args.eval_aggregation,
        "eval_initial": not bool(args.skip_initial_eval),
    }
    if reward == "catr":
        log["protocol"]["catr_config"] = str(Path(args.catr_config).resolve())
        log["protocol"]["catr_config_sha256"] = catr_sha256
        log["protocol"]["catr_coefficients"] = catr_coefficients.tolist()
        log["protocol"]["catr_invariant"] = (
            "token-weighted block advantage equals sequence-raw advantage"
        )
    if reward in UATR_REWARDS:
        log["protocol"]["adaptive_config"] = str(Path(args.adaptive_config).resolve())
        log["protocol"]["adaptive_config_sha256"] = uatr_sha256
        log["protocol"]["adaptive_coefficients"] = uatr_coefficients.tolist()
        log["protocol"]["adaptive_control"] = (
            "candidate identity shuffled within each horizon"
            if reward in {"uatr_shuffled", "uatr2_shuffled"}
            else "aligned candidate identity"
        )
    for name in (
        "lpips", "lpips_last", "mse", "mse_last", "psnr", "ssim",
        "token_hamming", "token_hamming_last", "latent_rms", "latent_rms_last",
    ):
        log[f"eval_{name}"] = []
    policy.eval()
    if not args.skip_initial_eval:
        _log_eval(
            log, 0,
            evaluate(
                policy, tokenizer, metrics, eval_entries,
                horizon=args.horizon, group_size=args.eval_K, device=device,
                seed=args.eval_seed, action_dim=args.action_dim, draws=args.eval_draws,
                aggregation=args.eval_aggregation,
            ),
            reward, credit, 0.0,
        )
    # Keep the exact same deterministic policy mode for generation and
    # teacher-forced log-probability evaluation.  Gradients remain enabled in
    # eval mode; switching to train mode here would make checkpoints with
    # non-zero dropout off-policy relative to their sampled candidates.
    policy.eval()
    # Keep the context schedule fixed across arms and candidate-sampling seeds.
    # Otherwise a "seed" changes both the stochastic policy rollout and the
    # empirical training distribution, making paired four-arm effects ambiguous.
    started = time.time()
    log["train_policy_loss"] = []
    log["train_kl"] = []
    log["train_grad_norm"] = []
    if reward == "rctr":
        log["train_rctr_coverage"] = []
        log["train_rctr_score_std"] = []
    if reward == "catr":
        log["train_catr_residual_rms"] = []
        log["train_catr_conservation_error"] = []
    if reward in ANCHORED_REWARDS:
        for name in (
            "train_raw_policy_loss", "train_preferred_policy_loss",
            "train_constraint_active", "train_projection_coefficient",
            "train_gradient_cosine", "train_preferred_progress_ratio",
            "train_projected_progress_ratio",
        ):
            log[name] = []
        if reward == "ra_rc":
            log["train_rc_policy_loss"] = []
    for step in range(1, args.steps + 1):
        selected = schedule[step - 1]
        optimizer.zero_grad(set_to_none=True)
        reward_values: list[float] = []
        loss_value = 0.0
        policy_value = 0.0
        kl_value = 0.0
        raw_policy_value = 0.0
        rc_policy_value = 0.0
        projection_rows = []
        rctr_coverage_rows = []
        rctr_score_std_rows = []
        catr_residual_rows = []
        catr_conservation_rows = []
        for ordinal, selected_index in enumerate(selected):
            entry = train_entries[int(selected_index)]
            window = load_vp2_window_npz(entry["window_npz"], action_dim=args.action_dim, device=device)
            with torch.inference_mode():
                ground_truth = tokenize_ground_truth(tokenizer, window)
                reachable = decoded_ground_truth(tokenizer, ground_truth)
                rollout = sample_rollout(
                    tokenizer, policy, ground_truth, window.actions,
                    horizon=args.horizon, group_size=args.K,
                    seed=(
                        seed * 1_000_003
                        + step * 10_007
                        + int(selected_index) * 101
                        + ordinal
                    ),
                )
                frame_values = frame_rewards(metrics, rollout, window, reachable)
                if reward == "ra_rc":
                    raw_advantage, raw_blockwise = _advantages(
                        frame_values["raw"], credit, args.gamma
                    )
                    preferred_advantage, preferred_blockwise = _advantages(
                        frame_values["rc"], credit, args.gamma
                    )
                    if raw_blockwise != preferred_blockwise:
                        raise RuntimeError("raw/RC credit modes diverged")
                    reward_frame = frame_values["rc"]
                elif reward in UATR_REWARDS:
                    raw_advantage, raw_blockwise = _advantages(
                        frame_values["raw"], "seq", args.gamma
                    )
                    shuffle_seed = None
                    if reward in {"uatr_shuffled", "uatr2_shuffled"}:
                        shuffle_seed = (
                            seed * 1_000_003
                            + step * 10_007
                            + int(selected_index) * 101
                            + ordinal
                            + 97_409
                        )
                    advantage_builder = (
                        influence_adaptive_delayed_advantages
                        if reward in {"uatr2", "uatr2_shuffled"}
                        else influence_adaptive_temporal_advantages
                    )
                    preferred_advantage = advantage_builder(
                        frame_values["raw"], frame_values["rc"], args.gamma,
                        uatr_coefficients, shuffle_seed=shuffle_seed,
                    )
                    preferred_blockwise = True
                    reward_frame = frame_values["raw"]
                elif reward == "rctr":
                    score, coverage = reachability_consistent_temporal_scores(
                        frame_values["raw"], frame_values["rc"], args.gamma
                    )
                    advantage = normalize_by_horizon(score)
                    blockwise = True
                    reward_frame = frame_values["raw"]
                    rctr_coverage_rows.append(float(np.mean(coverage)))
                    rctr_score_std_rows.append(float(np.mean(np.std(score, axis=0))))
                elif reward == "catr":
                    advantage = conservative_adaptive_temporal_advantages(
                        frame_values["raw"], frame_values["rc"], args.gamma,
                        catr_coefficients,
                        token_counts=np.full(args.horizon, DYNAMICS_GRID_TOKENS),
                    )
                    blockwise = True
                    reward_frame = frame_values["raw"]
                    raw_scalar = reward_frame.mean(axis=1)
                    raw_anchor = (raw_scalar - raw_scalar.mean()) / (
                        raw_scalar.std() + 1e-6
                    )
                    catr_residual_rows.append(float(np.sqrt(np.mean(
                        (advantage - raw_anchor[:, None]) ** 2
                    ))))
                    catr_conservation_rows.append(float(np.max(np.abs(
                        advantage.mean(axis=1) - raw_anchor
                    ))))
                else:
                    reward_frame = _select_reward(frame_values, reward, args.rc_mix)
                    advantage, blockwise = _advantages(reward_frame, credit, args.gamma)
            logp = teacher_forced_dynamics_logp(policy, rollout, window.actions)
            with torch.no_grad():
                reference_logp = teacher_forced_dynamics_logp(reference, rollout, window.actions)
            kl = sampled_kl_penalty(logp, reference_logp, args.kl_type).mean()
            if reward in ANCHORED_REWARDS:
                raw_policy_loss = _policy_loss(
                    logp, raw_advantage, raw_blockwise, device
                )
                preferred_policy_loss = _policy_loss(
                    logp, preferred_advantage, preferred_blockwise, device
                )
                raw_gradients = torch.autograd.grad(
                    raw_policy_loss, parameters, retain_graph=True, allow_unused=True
                )
                preferred_gradients = torch.autograd.grad(
                    preferred_policy_loss,
                    parameters,
                    retain_graph=bool(args.kl > 0.0),
                    allow_unused=True,
                )
                projected, projection = project_to_primary_progress(
                    raw_gradients, preferred_gradients
                )
                accumulate_parameter_gradients(
                    parameters, projected, scale=1.0 / args.batch_windows
                )
                if args.kl > 0.0:
                    (args.kl * kl / args.batch_windows).backward()
                raw_policy_value += float(raw_policy_loss.detach().cpu())
                rc_policy_value += float(preferred_policy_loss.detach().cpu())
                policy_value += float(preferred_policy_loss.detach().cpu())
                loss_value += float((preferred_policy_loss + args.kl * kl).detach().cpu())
                projection_rows.append(projection)
            else:
                policy_loss = _policy_loss(logp, advantage, blockwise, device)
                loss = policy_loss + args.kl * kl
                (loss / args.batch_windows).backward()
                loss_value += float(loss.detach().cpu())
                policy_value += float(policy_loss.detach().cpu())
            kl_value += float(kl.detach().cpu())
            reward_values.append(float(reward_frame.mean()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()
        log["train_policy_loss"].append(policy_value / args.batch_windows)
        log["train_kl"].append(kl_value / args.batch_windows)
        log["train_grad_norm"].append(grad_norm)
        if reward == "rctr":
            log["train_rctr_coverage"].append(float(np.mean(rctr_coverage_rows)))
            log["train_rctr_score_std"].append(float(np.mean(rctr_score_std_rows)))
        if reward == "catr":
            log["train_catr_residual_rms"].append(float(np.mean(catr_residual_rows)))
            log["train_catr_conservation_error"].append(float(np.max(catr_conservation_rows)))
        if reward in ANCHORED_REWARDS:
            log["train_raw_policy_loss"].append(raw_policy_value / args.batch_windows)
            log["train_preferred_policy_loss"].append(rc_policy_value / args.batch_windows)
            if reward == "ra_rc":
                log["train_rc_policy_loss"].append(rc_policy_value / args.batch_windows)
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
            f"[{reward}/{credit}] {_bar(step / args.steps)} {step}/{args.steps} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed / step * (args.steps - step))} "
            f"loss={loss_value / args.batch_windows:.5f} "
            f"pg={policy_value / args.batch_windows:.5f} "
            f"kl={kl_value / args.batch_windows:.3e} gn={grad_norm:.3f} "
            f"reward={np.mean(reward_values):.5f}"
            + (
                f" active={np.mean([row['constraint_active'] for row in projection_rows]):.2f} "
                f"cos={np.mean([row['gradient_cosine'] for row in projection_rows]):+.3f} "
                f"rawProg={np.mean([row['projected_progress_ratio'] for row in projection_rows]):.3f}"
                if reward in ANCHORED_REWARDS else ""
            )
            + (
                f" coverage={np.mean(rctr_coverage_rows):.3f} "
                f"rankStd={np.mean(rctr_score_std_rows):.3f}"
                if reward == "rctr" else ""
            )
            + (
                f" residRMS={np.mean(catr_residual_rows):.3f} "
                f"consErr={np.max(catr_conservation_rows):.2e}"
                if reward == "catr" else ""
            ),
            flush=True,
        )
        if step % args.eval_every == 0 or step == args.steps:
            policy.eval()
            result = evaluate(
                policy, tokenizer, metrics, eval_entries, horizon=args.horizon,
                group_size=args.eval_K, device=device, seed=args.eval_seed,
                action_dim=args.action_dim, draws=args.eval_draws,
                aggregation=args.eval_aggregation,
            )
            _log_eval(log, step, result, reward, credit, float(np.mean(reward_values)))
            policy.eval()
    if args.save_checkpoints:
        destination = Path(args.out_dir) / "ckpt" / f"vp2_{reward}_{credit}_s{seed}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": policy.state_dict(), "args": vars(args)}, destination)
        print(f"[ckpt] saved {destination}", flush=True)
    del policy, reference, tokenizer, metrics
    torch.cuda.empty_cache()
    return log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--rewards", default="raw,rc")
    parser.add_argument(
        "--rc_mix",
        type=float,
        default=0.25,
        help="RC fraction for reward=rc_mix; selected on calibration candidates",
    )
    parser.add_argument("--credits", default="seq,return")
    parser.add_argument(
        "--catr_config",
        default="",
        help="frozen delayed-influence JSON required by reward=catr",
    )
    parser.add_argument(
        "--adaptive_config",
        default="",
        help="frozen delayed-influence JSON required by reward=uatr/uatr_shuffled",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=4,
                        help="4 for RoboSuite PushCenter, 5 for RoboDesk")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--eval_K", type=int, default=4)
    parser.add_argument("--eval_draws", type=int, default=1)
    parser.add_argument(
        "--eval_aggregation",
        choices=("window_micro", "episode_macro"),
        default="window_micro",
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="avoid repeated base evaluation; pilot decisions still use the fixed final step",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch_windows", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl", type=float, default=0.001)
    parser.add_argument("--kl_type", choices=("low_var_kl", "linear"), default="low_var_kl")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--data_seed", type=int, default=7304,
                        help="fixed context-schedule seed, independent of policy sampling seeds")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, default=9917)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.K < 2 or args.eval_K < 1 or args.eval_draws < 1 or args.steps < 1 or args.batch_windows < 1:
        raise ValueError("K, eval_K, eval_draws, steps, and batch_windows must be positive (K >= 2)")
    rewards = [item.strip() for item in args.rewards.split(",") if item.strip()]
    credits = [item.strip() for item in args.credits.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not 0.0 <= args.rc_mix <= 1.0:
        raise ValueError("rc_mix must lie in [0,1]")
    unknown_credits = {c for c in credits
                       if c not in {"seq", "return", "return_eq", "adaptive"}
                       and not c.startswith("gae")}
    if sorted(set(rewards) - {"raw", "rc", "rc_mix", "ra_rc", "rctr", "catr", *UATR_REWARDS}) or unknown_credits:
        raise ValueError("unknown reward or credit assignment")
    if "rctr" in rewards and set(credits) != {"return"}:
        raise ValueError("rctr is defined only for credits=return")
    if "catr" in rewards and set(credits) != {"adaptive"}:
        raise ValueError("catr is defined only for credits=adaptive")
    if set(rewards) & UATR_REWARDS and set(credits) != {"adaptive"}:
        raise ValueError("uatr and uatr_shuffled are defined only for credits=adaptive")
    if set(credits) == {"adaptive"} and not set(rewards) <= ({"catr"} | UATR_REWARDS):
        raise ValueError("credits=adaptive is only valid for catr/uatr rewards")
    if args.kl_type != "low_var_kl":
        raise ValueError("P2 is preregistered with the RLVR-compatible low_var_kl estimator")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(reward, credit, seed) for seed in seeds for reward in rewards for credit in credits]
    complete, started = 0, time.time()
    for reward, credit, seed in jobs:
        output = Path(args.out_dir) / f"sweep_{reward}_{credit}_s{seed}.json"
        if output.exists():
            complete += 1
            print(f"[resume] {output} already exists", flush=True)
            continue
        elapsed = time.time() - started
        eta = elapsed / max(complete, 1) * (len(jobs) - complete)
        print(
            f"\n===== VP2 P2 SWEEP {_bar(complete / len(jobs))} {complete}/{len(jobs)} "
            f"next={reward}/{credit}/s{seed} elapsed={_hms(elapsed)} eta={_hms(eta)} =====",
            flush=True,
        )
        log = train_one(reward, credit, seed, args)
        output.write_text(json.dumps({"args": vars(args), "run": log}, indent=2) + "\n")
        print(f"[done] saved {output}", flush=True)
        complete += 1
    print(f"\n[sweep done] {complete}/{len(jobs)} in {_hms(time.time() - started)}\nVP2_P2_GRPO_OK", flush=True)


if __name__ == "__main__":
    main()
