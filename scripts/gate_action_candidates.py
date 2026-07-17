"""Gate A/D2: test the frozen action verifier on base-policy candidate groups."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from dor.action_verifier import action_score, load_payload, transition_features
from dor.constants import CTX, GRID, ROOT
from dor.episodes import get_window_tensors, load_episode
from dor.generation import generate_candidates
from dor.grpo import _get_raft, flow_fidelity, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
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


def episode_bootstrap(values, episodes, n_boot=2000, seed=2027):
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


def test_windows(root, episode_names, stride, n_windows, seed):
    per_episode = []
    rng = np.random.default_rng(seed)
    for name in episode_names:
        path = os.path.join(root, name)
        images, _ = load_episode(path)
        current = [(path, s) for s in range(0, len(images) - (CTX + 1), stride)]
        rng.shuffle(current)
        per_episode.append(current)
    if n_windows <= 0:
        windows = [item for current in per_episode for item in current]
        rng.shuffle(windows)
        return windows
    windows = []
    depth = 0
    while len(windows) < n_windows:
        added = False
        for current in per_episode:
            if depth < len(current):
                windows.append(current[depth])
                added = True
                if len(windows) == n_windows:
                    break
        if not added:
            break
        depth += 1
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/outputs/rcav/action_verifier.npz")
    ap.add_argument("--data_root", default=f"{ROOT}/data/processed/fractal20220817_data")
    ap.add_argument("--n_windows", type=int, default=48)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7401)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_flow", action="store_true")
    ap.add_argument("--cache_out", default=f"{ROOT}/outputs/rcav/action_candidate_gate.npz")
    ap.add_argument("--report_out", default=f"{ROOT}/outputs/rcav/action_verifier_gate_d2.json")
    args = ap.parse_args()

    payload = load_payload(args.model)
    pool_hw = tuple(payload["metadata"].get("feature_pool_hw", [4, 5]))
    windows = test_windows(
        args.data_root, payload["split"]["test"], args.stride, args.n_windows, args.seed
    )
    if not windows:
        raise RuntimeError("no test-episode windows available for candidate gate")

    set_determinism(args.seed)
    tok = load_tokenizer(args.device)
    model = load_world_model(args.device, "base")
    ar = load_action_ranges(args.device)
    metrics = Metrics(args.device)
    raft = None if args.skip_flow else _get_raft(args.device)

    rows = {name: [] for name in (
        "episode", "start", "action_std", "rho_dmotion", "rho_flow", "rho_state",
        "top_bottom_dmotion", "top_bottom_flow",
        "candidate_action", "candidate_state", "candidate_dmotion", "candidate_flow",
    )}
    t0 = time.time()
    print(
        f"[setup] test_episodes={len(payload['split']['test'])} windows={len(windows)} "
        f"K={args.K} flow={'off' if raft is None else 'on'}",
        flush=True,
    )
    for wi, (path, start) in enumerate(windows):
        frames, actions = get_window_tensors(path, start, args.device)
        cur, gt = frames[CTX - 1], frames[CTX]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(
            model, prompt, args.K, temperature=args.temperature, top_k=args.top_k,
            seed=args.seed + wi,
        )
        imgs = decode_tokens(tok, cand)
        cur_idx = encode_indices(tok, cur.unsqueeze(0))
        gt_idx = encode_indices(tok, gt.unsqueeze(0))
        z_cur = tok.indices_to_codes(cur_idx).float()
        z_cand = tok.indices_to_codes(cand.reshape(args.K, *GRID)).float()
        feat = transition_features(z_cur.expand_as(z_cand), z_cand, pool_hw).cpu().numpy()
        r_action = action_score(payload, feat, actions[CTX - 1].detach().cpu().numpy())

        reachable = decode_tokens(tok, gt_idx.reshape(1, -1))[0]
        state = metrics.eval_batch(imgs, reachable)
        r_state = -(np.asarray(state["lpips"]) + np.asarray(state["mse"]))
        dp = (imgs - cur.unsqueeze(0)).flatten(1)
        dg = (gt - cur).flatten().unsqueeze(0)
        dmotion = F.cosine_similarity(dp, dg, dim=1).detach().cpu().numpy()
        flow = (np.full(args.K, np.nan) if raft is None
                else np.asarray(flow_fidelity(raft, cur, imgs, gt), np.float64))

        best, worst = int(np.argmax(r_action)), int(np.argmin(r_action))
        rows["episode"].append(os.path.basename(path))
        rows["start"].append(int(start))
        rows["action_std"].append(float(np.std(r_action)))
        rows["rho_dmotion"].append(safe_spearman(r_action, dmotion))
        rows["rho_flow"].append(np.nan if raft is None else safe_spearman(r_action, flow))
        rows["rho_state"].append(safe_spearman(r_action, r_state))
        rows["top_bottom_dmotion"].append(float(dmotion[best] - dmotion[worst]))
        rows["top_bottom_flow"].append(
            np.nan if raft is None else float(flow[best] - flow[worst])
        )
        rows["candidate_action"].append(r_action.astype(np.float32))
        rows["candidate_state"].append(r_state.astype(np.float32))
        rows["candidate_dmotion"].append(dmotion.astype(np.float32))
        rows["candidate_flow"].append(flow.astype(np.float32))

        elapsed = time.time() - t0
        rate = (wi + 1) / max(elapsed, 1e-6)
        eta = (len(windows) - wi - 1) / max(rate, 1e-6)
        bar = "#" * (20 * (wi + 1) // len(windows))
        print(
            f"[gate {bar:<20}] {wi + 1}/{len(windows)} "
            f"rho_dyn={rows['rho_dmotion'][-1]:+.3f} "
            f"rho_flow={rows['rho_flow'][-1]:+.3f} elapsed={hms(elapsed)} eta={hms(eta)}",
            flush=True,
        )

    arrays = {k: np.asarray(v) for k, v in rows.items()}
    episodes = arrays["episode"].astype(str)
    dyn_boot = episode_bootstrap(arrays["rho_dmotion"], episodes, args.bootstrap, args.seed)
    flow_boot = None
    if raft is not None:
        flow_boot = episode_bootstrap(arrays["rho_flow"], episodes, args.bootstrap, args.seed + 1)
    nondegenerate = float(np.mean(arrays["action_std"] > 1e-4))
    median_abs_state = float(np.median(np.abs(arrays["rho_state"])))
    flow_reverse = bool(flow_boot is not None and flow_boot["q95"] < 0)
    green = bool(
        nondegenerate >= 0.80
        and dyn_boot["q05"] > 0
        and median_abs_state < 0.90
        and not flow_reverse
    )
    report = {
        "n_windows": int(len(windows)),
        "K": int(args.K),
        "generation_seed": int(args.seed),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "test_episodes": payload["split"]["test"].tolist(),
        "nondegenerate_fraction": nondegenerate,
        "median_abs_rho_state": median_abs_state,
        "dmotion_rho_bootstrap90": dyn_boot,
        "flow_rho_bootstrap90": flow_boot,
        "mean_top_bottom_dmotion": float(np.mean(arrays["top_bottom_dmotion"])),
        "mean_top_bottom_flow": (
            None if raft is None else float(np.mean(arrays["top_bottom_flow"]))
        ),
        "flow_significantly_reverse": flow_reverse,
        "green": green,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    np.savez_compressed(args.cache_out, **arrays)
    with open(args.report_out, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== RCAV Gate A / D2: Generated Candidates ===")
    print(f"nondegenerate={nondegenerate:.3f} median|rho(action,state)|={median_abs_state:.3f}")
    print(f"rho(action,dmotion)={dyn_boot['mean']:+.3f} "
          f"CI90=[{dyn_boot['q05']:+.3f},{dyn_boot['q95']:+.3f}]")
    if flow_boot is not None:
        print(f"rho(action,flow)={flow_boot['mean']:+.3f} "
              f"CI90=[{flow_boot['q05']:+.3f},{flow_boot['q95']:+.3f}]")
    print(f"[verdict] {'GREEN' if green else 'RED'}")
    print(f"saved {args.cache_out} and {args.report_out}\nACTION_VERIFIER_D2_OK", flush=True)


if __name__ == "__main__":
    main()
