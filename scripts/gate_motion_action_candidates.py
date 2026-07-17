"""Generated-candidate command-specificity gate for the RAFT action verifier."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from dor.action_observability import motion_oracle_features
from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, load_episode
from dor.generation import generate_candidates
from dor.grpo import _flow, _get_raft, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.motion_action import action_score, episode_fold, load_payload
from dor.tokenization import build_prompt, decode_tokens, encode_indices


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def safe_spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def episode_bootstrap(values, episodes, n_boot=5000, seed=2027):
    values = np.asarray(values, np.float64)
    episodes = np.asarray(episodes, dtype=str)
    unique = np.unique(episodes)
    means = np.asarray([values[episodes == ep].mean() for ep in unique], np.float64)
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(int(n_boot), len(means)))].mean(1)
    return {
        "mean": float(means.mean()),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
        "episode_means": means.tolist(),
    }


def balanced_windows(root, episode_names, stride, n_windows, seed):
    rng = np.random.default_rng(seed)
    per_episode = []
    for name in episode_names:
        path = os.path.join(root, name)
        images, _ = load_episode(path)
        current = [(path, s) for s in range(0, len(images) - (CTX + 1), stride)]
        rng.shuffle(current)
        per_episode.append(current)
    windows = []
    depth = 0
    while n_windows <= 0 or len(windows) < n_windows:
        added = False
        for current in per_episode:
            if depth < len(current):
                windows.append(current[depth])
                added = True
                if n_windows > 0 and len(windows) == n_windows:
                    return windows
        if not added:
            return windows
        depth += 1
    return windows


def shuffled_targets(windows):
    targets, names = [], []
    for path, start in windows:
        _, actions = load_episode(path)
        targets.append(actions[start + CTX - 1])
        names.append(os.path.basename(path))
    targets = np.asarray(targets, dtype=np.float64)
    names = np.asarray(names, dtype=str)
    shuffled = np.empty_like(targets)
    for i in range(len(windows)):
        for shift in range(1, len(windows)):
            j = (i + shift) % len(windows)
            if names[j] != names[i]:
                shuffled[i] = targets[j]
                break
        else:
            raise RuntimeError("shuffled-command control requires multiple episodes")
    return targets, shuffled, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=f"{ROOT}/outputs/rcav/motion_action_payload_h1_8x10.npz")
    ap.add_argument("--data_root", default=f"{ROOT}/data/processed/fractal20220817_data")
    ap.add_argument("--n_windows", type=int, default=80)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7401,
                    help="backward-compatible default for both seeds")
    ap.add_argument("--window_seed", type=int, default=None)
    ap.add_argument("--generation_seed", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_out", default=f"{ROOT}/outputs/rcav/motion_action_candidate_gate.npz")
    ap.add_argument("--report_out", default=f"{ROOT}/outputs/rcav/motion_action_candidate_gate.json")
    args = ap.parse_args()
    window_seed = args.seed if args.window_seed is None else args.window_seed
    generation_seed = args.seed if args.generation_seed is None else args.generation_seed

    payload = load_payload(args.payload)
    pool = tuple(int(x) for x in payload["metadata"]["pool"].split("x"))
    episode_names = payload["fold_test_episode_names"].reshape(-1)
    windows = balanced_windows(
        args.data_root, episode_names, args.stride, args.n_windows, window_seed
    )
    targets, shuffled, expected_names = shuffled_targets(windows)
    if not windows:
        raise RuntimeError("no candidate-gate windows")

    set_determinism(args.seed)
    tokenizer = load_tokenizer(args.device)
    model = load_world_model(args.device, "base")
    action_ranges = load_action_ranges(args.device)
    metrics = Metrics(args.device)
    raft = _get_raft(args.device)
    keys = (
        "episode", "start", "fold", "action_std", "rho_dmotion", "rho_shuffle_dmotion",
        "delta_rho_command", "rho_state", "rho_raw", "rho_flow",
        "top_bottom_dmotion", "top_bottom_lpips", "top_bottom_mse",
        "candidate_action", "candidate_action_shuffled", "candidate_state",
        "candidate_raw", "candidate_dmotion", "candidate_flow",
        "target_action", "shuffled_action",
    )
    rows = {key: [] for key in keys}
    t0 = time.time()
    print(
        f"[setup] episodes={len(np.unique(expected_names))} windows={len(windows)} "
        f"K={args.K} pool={pool} weights="
        + ",".join(f"{x:.3f}" for x in payload["dimension_weights"]),
        flush=True,
    )
    for wi, (path, start) in enumerate(windows):
        name = os.path.basename(path)
        if name != expected_names[wi]:
            raise AssertionError("window/action order mismatch")
        fold = episode_fold(payload, name)
        frames, actions = get_window_tensors(path, start, args.device)
        current, gt = frames[CTX - 1], frames[CTX]
        prompt = build_prompt(tokenizer, frames, actions, action_ranges)
        candidates = generate_candidates(
            model, prompt, args.K, temperature=args.temperature,
            top_k=args.top_k, seed=generation_seed + wi,
        )
        images = decode_tokens(tokenizer, candidates)
        current_batch = current.unsqueeze(0).expand(args.K, -1, -1, -1)
        candidate_flow = _flow(raft, current_batch, images)
        features, _ = motion_oracle_features(current_batch, candidate_flow, pool)
        features = features.cpu().numpy()
        r_action = action_score(payload, fold, features, targets[wi])
        r_shuffled = action_score(payload, fold, features, shuffled[wi])

        gt_indices = encode_indices(tokenizer, gt.unsqueeze(0))
        reachable = decode_tokens(tokenizer, gt_indices.reshape(1, -1))[0]
        state_metrics = metrics.eval_batch(images, reachable)
        raw_metrics = metrics.eval_batch(images, gt)
        r_state = -(np.asarray(state_metrics["lpips"]) + np.asarray(state_metrics["mse"]))
        r_raw = -(np.asarray(raw_metrics["lpips"]) + np.asarray(raw_metrics["mse"]))
        delta_pred = (images - current.unsqueeze(0)).flatten(1)
        delta_gt = (gt - current).flatten().unsqueeze(0)
        dmotion = F.cosine_similarity(delta_pred, delta_gt, dim=1).cpu().numpy()
        gt_flow = _flow(raft, current.unsqueeze(0), gt.unsqueeze(0))
        weight = gt_flow.norm(dim=1)
        cosine = F.cosine_similarity(candidate_flow, gt_flow, dim=1)
        flow_fidelity = ((cosine * weight).sum((-1, -2)) /
                         (weight.sum((-1, -2)) + 1e-6)).cpu().numpy()

        best, worst = int(np.argmax(r_action)), int(np.argmin(r_action))
        rho_dyn = safe_spearman(r_action, dmotion)
        rho_shuffle = safe_spearman(r_shuffled, dmotion)
        rows["episode"].append(name)
        rows["start"].append(int(start))
        rows["fold"].append(int(fold))
        rows["action_std"].append(float(np.std(r_action)))
        rows["rho_dmotion"].append(rho_dyn)
        rows["rho_shuffle_dmotion"].append(rho_shuffle)
        rows["delta_rho_command"].append(rho_dyn - rho_shuffle)
        rows["rho_state"].append(safe_spearman(r_action, r_state))
        rows["rho_raw"].append(safe_spearman(r_action, r_raw))
        rows["rho_flow"].append(safe_spearman(r_action, flow_fidelity))
        rows["top_bottom_dmotion"].append(float(dmotion[best] - dmotion[worst]))
        rows["top_bottom_lpips"].append(
            float(raw_metrics["lpips"][worst] - raw_metrics["lpips"][best])
        )
        rows["top_bottom_mse"].append(
            float(raw_metrics["mse"][worst] - raw_metrics["mse"][best])
        )
        rows["candidate_action"].append(r_action.astype(np.float32))
        rows["candidate_action_shuffled"].append(r_shuffled.astype(np.float32))
        rows["candidate_state"].append(r_state.astype(np.float32))
        rows["candidate_raw"].append(r_raw.astype(np.float32))
        rows["candidate_dmotion"].append(dmotion.astype(np.float32))
        rows["candidate_flow"].append(flow_fidelity.astype(np.float32))
        rows["target_action"].append(targets[wi].astype(np.float32))
        rows["shuffled_action"].append(shuffled[wi].astype(np.float32))

        elapsed = time.time() - t0
        rate = (wi + 1) / max(elapsed, 1e-6)
        eta = (len(windows) - wi - 1) / max(rate, 1e-6)
        filled = 20 * (wi + 1) // len(windows)
        print(
            f"[gate {'#' * filled:<20}] {wi + 1}/{len(windows)} "
            f"rho_dyn={rho_dyn:+.3f} delta_cmd={rho_dyn - rho_shuffle:+.3f} "
            f"elapsed={hms(elapsed)} eta={hms(eta)}",
            flush=True,
        )

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    episodes = arrays["episode"].astype(str)
    dyn_boot = episode_bootstrap(arrays["rho_dmotion"], episodes, args.bootstrap, generation_seed)
    command_boot = episode_bootstrap(
        arrays["delta_rho_command"], episodes, args.bootstrap, generation_seed + 1
    )
    top_dyn_boot = episode_bootstrap(
        arrays["top_bottom_dmotion"], episodes, args.bootstrap, generation_seed + 2
    )
    nondegenerate = float(np.mean(arrays["action_std"] > 1e-4))
    median_abs_state = float(np.median(np.abs(arrays["rho_state"])))
    green = bool(
        nondegenerate >= 0.80
        and dyn_boot["q05"] > 0
        and command_boot["q05"] > 0
        and median_abs_state < 0.90
    )
    report = {
        "n_windows": len(windows),
        "K": args.K,
        "window_seed": window_seed,
        "generation_seed": generation_seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "episode_cross_fitted": True,
        "dimension_weights": payload["dimension_weights"].tolist(),
        "nondegenerate_fraction": nondegenerate,
        "median_abs_rho_state": median_abs_state,
        "dmotion_rho_bootstrap90": dyn_boot,
        "matched_minus_shuffled_rho_bootstrap90": command_boot,
        "top_bottom_dmotion_bootstrap90": top_dyn_boot,
        "mean_rho_raw": float(np.mean(arrays["rho_raw"])),
        "mean_rho_flow_secondary": float(np.mean(arrays["rho_flow"])),
        "mean_top_bottom_lpips": float(np.mean(arrays["top_bottom_lpips"])),
        "mean_top_bottom_mse": float(np.mean(arrays["top_bottom_mse"])),
        "green": green,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    np.savez_compressed(args.cache_out, **arrays)
    with open(args.report_out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== RCAV Gate B: Generated-Candidate Command Specificity ===")
    print(f"nondegenerate={nondegenerate:.3f} median|rho(action,RC)|={median_abs_state:.3f}")
    print(
        f"rho(action,dmotion)={dyn_boot['mean']:+.3f} "
        f"CI90=[{dyn_boot['q05']:+.3f},{dyn_boot['q95']:+.3f}]"
    )
    print(
        f"matched-shuffled delta={command_boot['mean']:+.3f} "
        f"CI90=[{command_boot['q05']:+.3f},{command_boot['q95']:+.3f}]"
    )
    print(
        f"top-bottom dmotion={top_dyn_boot['mean']:+.3f} "
        f"CI90=[{top_dyn_boot['q05']:+.3f},{top_dyn_boot['q95']:+.3f}]"
    )
    print(f"[verdict] {'GREEN' if green else 'RED'}")
    print(f"saved {args.cache_out} and {args.report_out}\nMOTION_ACTION_CANDIDATE_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
