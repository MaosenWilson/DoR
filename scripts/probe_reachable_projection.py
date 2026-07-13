"""Audit whether encode-decode GT is locally optimal among legal FSQ targets."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_tokenizer
from dor.reachable_projection import (
    decode_metric_score,
    greedy_metric_refine,
    hamming_fraction,
)
from dor.tokenization import decode_tokens, encode_indices


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=64)
    parser.add_argument("--exclude_windows", type=int, default=36)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--metric_batch", type=int, default=8)
    parser.add_argument("--min_improvement", type=float, default=1e-7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default=f"{ROOT}/outputs/analysis/reachable_projection.json")
    args = parser.parse_args()

    if args.deterministic:
        set_determinism(2701)
    device = "cuda"
    tok = load_tokenizer(device)
    metrics = Metrics(device)
    levels = [int(value) for value in tok.fsq_levels]
    windows = sample_windows(
        list_episodes(), args.exclude_windows + args.n_windows, seed=args.window_seed
    )[args.exclude_windows:]
    if len(windows) != args.n_windows:
        raise RuntimeError(f"requested {args.n_windows} windows, found {len(windows)}")

    records = []
    started = time.time()
    print(
        f"[setup] windows={len(windows)} levels={levels} positions={args.positions} "
        f"rounds={args.rounds}", flush=True,
    )
    for wi, (path, start) in enumerate(windows):
        frames, _ = get_window_tensors(path, start, device)
        target = frames[CTX]
        base = encode_indices(tok, target.unsqueeze(0))[0]
        base_image = decode_tokens(tok, base.reshape(1, -1))[0]
        base_score, base_parts = decode_metric_score(
            tok, metrics, base.unsqueeze(0), target, args.metric_batch
        )
        current, trace, evaluated = greedy_metric_refine(
            tok, metrics, base, target, levels,
            positions=args.positions, rounds=args.rounds,
            batch_size=args.metric_batch, min_improvement=args.min_improvement,
        )
        accepted = len(trace)
        final_score, final_parts = decode_metric_score(
            tok, metrics, current.unsqueeze(0), target, args.metric_batch
        )
        records.append({
            "episode": os.path.basename(path),
            "start": int(start),
            "base_objective": float(base_score[0]),
            "projected_objective": float(final_score[0]),
            "base_mse": float(base_parts[0, 0]),
            "base_lpips": float(base_parts[0, 1]),
            "projected_mse": float(final_parts[0, 0]),
            "projected_lpips": float(final_parts[0, 1]),
            "absolute_gain": float(base_score[0] - final_score[0]),
            "relative_gain": float((base_score[0] - final_score[0]) / max(base_score[0], 1e-12)),
            "hamming_fraction": hamming_fraction(base, current),
            "accepted_moves": accepted,
            "evaluated_neighbors": evaluated,
        })
        done = wi + 1
        elapsed = time.time() - started
        print(
            f"[projection] {_bar(done / len(windows))} {done}/{len(windows)} "
            f"gain={records[-1]['relative_gain']:+.4f} moves={accepted} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (len(windows)-done))}",
            flush=True,
        )

    gains = np.asarray([row["relative_gain"] for row in records])
    report = {
        "protocol": vars(args),
        "levels": levels,
        "windows": records,
        "summary": {
            "n": len(records),
            "improved": int(np.sum(gains > 0)),
            "mean_relative_gain": float(gains.mean()),
            "median_relative_gain": float(np.median(gains)),
            "q05_relative_gain": float(np.quantile(gains, 0.05)),
            "mean_hamming_fraction": float(np.mean([
                row["hamming_fraction"] for row in records
            ])),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"saved {args.out}\nREACHABLE_PROJECTION_OK", flush=True)


if __name__ == "__main__":
    main()
