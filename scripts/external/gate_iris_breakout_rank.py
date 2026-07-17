"""Cross-architecture RC rank-reliability gate on IRIS Breakout."""
from __future__ import annotations

import argparse
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
)
from dor.metrics import Metrics
from dor.rank_stats import pair_flip_fraction, rowwise_spearman


def _progress(done: int, total: int, started: float):
    elapsed = time.monotonic() - started
    eta = elapsed / max(done, 1) * (total - done)
    width = 24
    filled = int(width * done / total)
    print(
        f"\r[IRIS gate {'#' * filled}{'-' * (width-filled)}] {done}/{total} "
        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
        end="",
        flush=True,
    )
    if done == total:
        print(flush=True)


def _cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    rounds: int,
    seed: int,
    confidence: float,
) -> dict:
    unique = np.unique(clusters)
    per_cluster = np.asarray([np.nanmean(values[clusters == item]) for item in unique])
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        np.nanmean(per_cluster[rng.integers(0, len(per_cluster), len(per_cluster))])
        for _ in range(rounds)
    ])
    tail = (1.0 - confidence) / 2.0
    interval = [float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))]
    report = {
        "mean": float(np.nanmean(per_cluster)),
        "confidence": float(confidence),
        "ci": interval,
        "episodes": int(len(unique)),
    }
    if abs(confidence - 0.90) < 1e-12:
        report["ci90"] = interval
    return report


def _summary(raw, rc, reference, episodes, rounds, seed, confidence):
    raw_s = rowwise_spearman(raw, reference)
    rc_s = rowwise_spearman(rc, reference)
    raw_f = pair_flip_fraction(raw, reference)
    rc_f = pair_flip_fraction(rc, reference)
    raw_top = np.argmax(raw, axis=-1)
    rc_top = np.argmax(rc, axis=-1)
    top_delta = (
        np.take_along_axis(reference, rc_top[..., None], -1)[..., 0]
        - np.take_along_axis(reference, raw_top[..., None], -1)[..., 0]
    )
    return {
        "delta_spearman": _cluster_bootstrap(rc_s - raw_s, episodes, rounds, seed, confidence),
        "delta_flip": _cluster_bootstrap(rc_f - raw_f, episodes, rounds, seed + 1, confidence),
        "delta_reference_top": _cluster_bootstrap(top_delta, episodes, rounds, seed + 2, confidence),
        "raw_spearman": _cluster_bootstrap(raw_s, episodes, rounds, seed + 3, confidence),
        "rc_spearman": _cluster_bootstrap(rc_s, episodes, rounds, seed + 4, confidence),
        "raw_flip": _cluster_bootstrap(raw_f, episodes, rounds, seed + 5, confidence),
        "rc_flip": _cluster_bootstrap(rc_f, episodes, rounds, seed + 6, confidence),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9203)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest["entries"]
    device = torch.device(args.device)
    tokenizer, world_model = load_iris(
        args.upstream,
        args.checkpoint,
        action_vocab_size=(
            int(manifest["action_vocab_size"])
            if "action_vocab_size" in manifest
            else None
        ),
        device=device,
    )
    metrics = Metrics(device)
    shape = (len(entries), args.draws, args.K)
    raw, rc, latent, hamming = (np.empty(shape, dtype=np.float32) for _ in range(4))
    raw_lpips, raw_mse = (np.empty(shape, dtype=np.float32) for _ in range(2))
    episodes = np.empty(len(entries), dtype=np.int64)
    steps = np.empty(len(entries), dtype=np.int64)
    started = time.monotonic()
    for index, entry in enumerate(entries):
        window = load_iris_window_npz(entry["window_npz"], device=device)
        for draw in range(args.draws):
            rollout, target_tokens = sample_next_frame(
                tokenizer,
                world_model,
                window,
                group_size=args.K,
                seed=args.seed + 1009 * index + 7919 * draw,
                temperature=args.temperature,
            )
            reachable = reachable_target(tokenizer, target_tokens)
            raw_metric = metrics.eval_batch(rollout.decoded, window.frames[-1])
            rc_metric = metrics.eval_batch(rollout.decoded, reachable)
            raw[index, draw] = -(
                np.asarray(raw_metric["mse"]) + np.asarray(raw_metric["lpips"])
            )
            raw_lpips[index, draw] = np.asarray(raw_metric["lpips"])
            raw_mse[index, draw] = np.asarray(raw_metric["mse"])
            rc[index, draw] = -(
                np.asarray(rc_metric["mse"]) + np.asarray(rc_metric["lpips"])
            )
            latent[index, draw] = post_quant_latent_reward(
                tokenizer, rollout.tokens, target_tokens
            ).cpu().numpy()
            hamming[index, draw] = -(
                rollout.tokens != target_tokens.unsqueeze(0)
            ).float().mean(dim=-1).cpu().numpy()
        episodes[index] = window.episode
        steps[index] = window.step
        _progress(index + 1, len(entries), started)

    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        raw_reward=raw,
        rc_reward=rc,
        post_quant_reward=latent,
        hamming_reward=hamming,
        raw_lpips=raw_lpips,
        raw_mse=raw_mse,
        episode=episodes,
        step=steps,
    )
    latent_report = _summary(
        raw, rc, latent, episodes, args.bootstrap, args.seed + 50000, args.confidence
    )
    hamming_report = _summary(
        raw, rc, hamming, episodes, args.bootstrap, args.seed + 60000, args.confidence
    )
    mechanism = (
        latent_report["delta_spearman"]["ci"][0] > 0
        and latent_report["delta_flip"]["ci"][1] < 0
    )
    selection = latent_report["delta_reference_top"]["ci"][0] > 0
    report = dict(latent_report)
    report.update({
        "protocol": "IRIS Atari RC rank closure v1",
        "environment": manifest["environment"],
        "manifest": str(Path(args.manifest).resolve()),
        "cache": str(cache.resolve()),
        "shape": {"contexts": len(entries), "draws": args.draws, "group_size": args.K},
        "context_length": int(manifest["context"]),
        "primary_reference": "negative RMS in tokenizer post_quant_conv latent",
        "base_raw_lpips": float(np.mean(raw_lpips)),
        "base_raw_mse": float(np.mean(raw_mse)),
        "hamming_sensitivity": hamming_report,
        "mechanism_verdict": "GREEN" if mechanism else "RED",
        "selection_verdict": "GREEN" if selection else "RED",
        "verdict": "GREEN" if mechanism and selection else "RED",
    })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n=== IRIS Atari RC Rank Gate ({manifest['environment']}) ===")
    for key in (
        "delta_spearman", "delta_flip", "delta_reference_top",
        "raw_spearman", "rc_spearman", "raw_flip", "rc_flip",
    ):
        value = report[key]
        label = int(round(100 * value["confidence"]))
        print(
            f"{key:>24s} {value['mean']:+.5f} "
            f"CI{label}=[{value['ci'][0]:+.5f},{value['ci'][1]:+.5f}]"
        )
    print(f"[mechanism] {report['mechanism_verdict']} | [selection] {report['selection_verdict']} | [verdict] {report['verdict']}")
    print(f"saved {output}\nIRIS_ATARI_RC_RANK_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
