"""P1: zero-training RC rank-reliability gate on public VP2-RoboSuite windows.

The primary reference follows the frozen tokenizer from categorical dynamics IDs
to the continuous latent vectors actually consumed by the decoder.  Token Hamming
and unprojected codebook distance are retained as sensitivity analyses.  Raw-GT
and reconstruction-calibrated (RC) rewards see precisely the same sampled groups,
and episode-cluster bootstrap is the unit of inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dor.adapters.ivideogpt_vp2 import (
    frame_rewards,
    future_dynamics_latent_reward,
    future_dynamics_tokens,
    load_ivideogpt,
    load_vp2_window_npz,
    sample_rollout,
    tokenize_ground_truth,
    decoded_ground_truth,
)
from dor.metrics import Metrics
from dor.rank_stats import pair_flip_fraction, rowwise_pearson, rowwise_spearman


def _progress(done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = elapsed / max(done, 1)
    eta = rate * (total - done)
    width = 24
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[VP2 P1 {bar}] {done}/{total} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", end="", flush=True)
    if done == total:
        print(flush=True)


def _episode_bootstrap(values: np.ndarray, episodes: np.ndarray, rounds: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes).astype(str)
    if values.shape[0] != len(episodes):
        raise ValueError("values and episodes must share first dimension")
    unique = np.unique(episodes)
    grouped = np.stack([np.nanmean(values[episodes == item], axis=0) for item in unique])
    per_episode = np.nanmean(grouped.reshape(len(grouped), -1), axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty(rounds, dtype=np.float64)
    for draw in range(rounds):
        selected = rng.integers(0, len(per_episode), size=len(per_episode))
        boot[draw] = np.nanmean(per_episode[selected])
    return {
        "mean": float(np.nanmean(per_episode)),
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "episodes": int(len(unique)),
    }


def _summary(raw: np.ndarray, rc: np.ndarray, reference: np.ndarray, episodes: np.ndarray, bootstrap: int, seed: int) -> dict:
    # Arrays are [context, draw, horizon, candidate].
    raw_spearman = rowwise_spearman(raw, reference)
    rc_spearman = rowwise_spearman(rc, reference)
    raw_pearson = rowwise_pearson(raw, reference)
    rc_pearson = rowwise_pearson(rc, reference)
    raw_flip = pair_flip_fraction(raw, reference)
    rc_flip = pair_flip_fraction(rc, reference)
    raw_top = np.argmax(raw, axis=-1)
    rc_top = np.argmax(rc, axis=-1)
    selected_delta = np.take_along_axis(reference, rc_top[..., None], axis=-1)[..., 0] - np.take_along_axis(reference, raw_top[..., None], axis=-1)[..., 0]
    expected_raw = np.arccos(np.clip(raw_pearson, -1.0, 1.0)) / np.pi
    expected_rc = np.arccos(np.clip(rc_pearson, -1.0, 1.0)) / np.pi
    return {
        "delta_spearman": _episode_bootstrap(rc_spearman - raw_spearman, episodes, bootstrap, seed),
        "delta_flip": _episode_bootstrap(rc_flip - raw_flip, episodes, bootstrap, seed + 1),
        "delta_reference_top": _episode_bootstrap(selected_delta, episodes, bootstrap, seed + 2),
        "raw_spearman": _episode_bootstrap(raw_spearman, episodes, bootstrap, seed + 3),
        "rc_spearman": _episode_bootstrap(rc_spearman, episodes, bootstrap, seed + 4),
        "raw_flip": _episode_bootstrap(raw_flip, episodes, bootstrap, seed + 5),
        "rc_flip": _episode_bootstrap(rc_flip, episodes, bootstrap, seed + 6),
        "raw_gaussian_flip_error": _episode_bootstrap(raw_flip - expected_raw, episodes, bootstrap, seed + 7),
        "rc_gaussian_flip_error": _episode_bootstrap(rc_flip - expected_rc, episodes, bootstrap, seed + 8),
        "per_draw_delta_spearman": [float(np.nanmean(rc_spearman[:, draw] - raw_spearman[:, draw])) for draw in range(raw.shape[1])],
        "per_draw_delta_flip": [float(np.nanmean(rc_flip[:, draw] - raw_flip[:, draw])) for draw in range(raw.shape[1])],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--primary_reference",
        choices=("post_quant", "codebook", "hamming"),
        default="post_quant",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.K < 2 or args.draws < 1:
        raise ValueError("K must be at least two and draws must be positive")
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest["entries"]
    if int(manifest["horizon"]) != args.horizon:
        raise ValueError("manifest horizon does not match --horizon")
    device = torch.device(args.device)
    tokenizer, model = load_ivideogpt(args.upstream, args.checkpoint, horizon=args.horizon, device=device)
    metrics = Metrics(device)
    n = len(entries)
    raw = np.empty((n, args.draws, args.horizon, args.K), dtype=np.float32)
    rc = np.empty_like(raw)
    hamming = np.empty_like(raw)
    codebook = np.empty_like(raw)
    post_quant = np.empty_like(raw)
    episode = np.empty(n, dtype="U32")
    start = np.empty(n, dtype=np.int64)
    started = time.monotonic()
    for index, entry in enumerate(entries):
        window = load_vp2_window_npz(entry["window_npz"], device=device)
        if window.horizon != args.horizon:
            raise ValueError(f"window {entry['window_npz']} has incompatible horizon")
        with torch.inference_mode():
            ground_truth = tokenize_ground_truth(tokenizer, window)
            reachable = decoded_ground_truth(tokenizer, ground_truth)
            target_dynamics = future_dynamics_tokens(ground_truth, args.horizon)[0]
            for draw in range(args.draws):
                rollout = sample_rollout(
                    tokenizer, model, ground_truth, window.actions,
                    horizon=args.horizon, group_size=args.K,
                    seed=args.seed + 1009 * index + 7919 * draw,
                )
                reward = frame_rewards(metrics, rollout, window, reachable)
                # Adapter reward layout is [candidate, horizon]; P1 caches
                # [horizon, candidate] so all rowwise statistics operate on a
                # candidate group in the final dimension.
                raw[index, draw] = reward["raw"].T
                rc[index, draw] = reward["rc"].T
                hamming[index, draw] = -(
                    rollout.dynamics_tokens != target_dynamics.unsqueeze(0)
                ).float().mean(dim=-1).detach().cpu().numpy().T
                codebook[index, draw] = future_dynamics_latent_reward(
                    tokenizer,
                    rollout.dynamics_tokens,
                    target_dynamics,
                    projected=False,
                ).detach().cpu().numpy().T
                post_quant[index, draw] = future_dynamics_latent_reward(
                    tokenizer,
                    rollout.dynamics_tokens,
                    target_dynamics,
                    projected=True,
                ).detach().cpu().numpy().T
        episode[index] = window.episode
        start[index] = window.start
        _progress(index + 1, n, started)
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    references = {
        "post_quant": post_quant,
        "codebook": codebook,
        "hamming": hamming,
    }
    np.savez_compressed(
        cache,
        raw_reward=raw,
        rc_reward=rc,
        post_quant_reward=post_quant,
        codebook_reward=codebook,
        hamming_reward=hamming,
        episode=episode,
        start=start,
    )
    reports = {
        name: _summary(raw, rc, value, episode, args.bootstrap, args.seed + 90000 + 100 * offset)
        for offset, (name, value) in enumerate(references.items())
    }
    # Copy the selected summary before embedding all sensitivity summaries;
    # otherwise ``report`` would contain itself through ``reports``.
    report = dict(reports[args.primary_reference])
    mechanism_green = report["delta_spearman"]["ci90"][0] > 0.0 and report["delta_flip"]["ci90"][1] < 0.0
    selection_green = report["delta_reference_top"]["ci90"][0] > 0.0
    report.update({
        "protocol": "VP2 P1 decoder-floor rank closure v2",
        "manifest": str(Path(args.manifest).resolve()),
        "cache": str(cache.resolve()),
        "shape": {"contexts": n, "draws": args.draws, "horizon": args.horizon, "group_size": args.K},
        "cluster": "episode",
        "primary_reference": args.primary_reference,
        "reference_definitions": {
            "post_quant": "negative RMS after dynamics codebook lookup and post_quant_linear (primary decoder-input latent)",
            "codebook": "negative RMS between learned dynamics codebook embeddings",
            "hamming": "negative categorical token mismatch fraction (legacy sensitivity only)",
        },
        "reference_sensitivity": reports,
        "gaussian_curve_note": "diagnostic only; the arccos curve is not used as a significance test",
        "mechanism_verdict": "GREEN" if mechanism_green else "RED",
        "selection_verdict": "GREEN" if selection_green else "RED",
        "verdict": "GREEN" if mechanism_green and selection_green else "RED",
    })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n=== VP2 RC Rank-Reliability Gate ({args.primary_reference}) ===")
    for key in ("delta_spearman", "delta_flip", "delta_reference_top", "raw_spearman", "rc_spearman", "raw_flip", "rc_flip"):
        value = report[key]
        print(f"{key:>24s} {value['mean']:+.5f} CI90=[{value['ci90'][0]:+.5f},{value['ci90'][1]:+.5f}]")
    print(f"per-draw dSpearman={report['per_draw_delta_spearman']}")
    print(f"per-draw dFlip={report['per_draw_delta_flip']}")
    print(f"[mechanism] {report['mechanism_verdict']} | [selection] {report['selection_verdict']} | [verdict] {report['verdict']}")
    print("reference sensitivity:")
    for name, sensitivity in reports.items():
        print(
            f"  {name:>10s}: dSpearman={sensitivity['delta_spearman']['mean']:+.5f} "
            f"dFlip={sensitivity['delta_flip']['mean']:+.5f} "
            f"dTop={sensitivity['delta_reference_top']['mean']:+.5f}"
        )
    print(f"saved {output}\nVP2_RC_RANK_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
