"""Cache fixed-budget MRRT and matched-random targets for single-step GRPO."""

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
    highest_error_positions,
    matched_random_legal_target,
)
from dor.tokenization import decode_tokens, encode_indices


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_windows", type=int, default=24)
    parser.add_argument("--eval_windows", type=int, default=12)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--metric_batch", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default=f"{ROOT}/outputs/mrrt/train_targets.npz")
    args = parser.parse_args()

    if args.deterministic:
        set_determinism(2701)
    device = "cuda"
    tok = load_tokenizer(device)
    metrics = Metrics(device)
    levels = [int(value) for value in tok.fsq_levels]
    all_windows = sample_windows(
        list_episodes(), args.train_windows + args.eval_windows, seed=args.window_seed
    )
    windows = all_windows[:args.train_windows]
    episodes, starts, base_rows, mrrt_rows, random_rows, records = [], [], [], [], [], []
    started = time.time()
    for wi, (path, start) in enumerate(windows):
        frames, _ = get_window_tensors(path, start, device)
        target = frames[CTX]
        base = encode_indices(tok, target.unsqueeze(0))[0]
        base_image = decode_tokens(tok, base.reshape(1, -1))[0]
        mrrt, trace, evaluated = greedy_metric_refine(
            tok, metrics, base, target, levels,
            positions=args.positions, rounds=args.rounds, batch_size=args.metric_batch,
        )
        traced_positions = sorted({
            position
            for step_record in trace
            for position in step_record["candidate_positions"]
        })
        candidate_positions = torch.as_tensor(traced_positions, device=base.device)
        if not len(candidate_positions):
            candidate_positions = highest_error_positions(
                target, base_image, tuple(base.shape), args.positions
            )
        changed_cells = int((base != mrrt).sum().item())
        random_target = matched_random_legal_target(
            base, levels, candidate_positions, changed_cells, seed=7301 + wi
        )
        stacked = torch.stack([base, mrrt, random_target])
        scores, parts = decode_metric_score(
            tok, metrics, stacked, target, args.metric_batch
        )
        records.append({
            "base_objective": float(scores[0]),
            "mrrt_objective": float(scores[1]),
            "random_objective": float(scores[2]),
            "mrrt_lpips": float(parts[1, 1]),
            "random_lpips": float(parts[2, 1]),
            "mrrt_hamming": hamming_fraction(base, mrrt),
            "random_hamming": hamming_fraction(base, random_target),
            "accepted_moves": len(trace),
            "evaluated_neighbors": evaluated,
        })
        episodes.append(os.path.basename(path))
        starts.append(int(start))
        base_rows.append(base.cpu().numpy())
        mrrt_rows.append(mrrt.cpu().numpy())
        random_rows.append(random_target.cpu().numpy())
        done = wi + 1
        elapsed = time.time() - started
        print(
            f"[mrrt-cache] {_bar(done / len(windows))} {done}/{len(windows)} "
            f"gain={scores[0]-scores[1]:+.6f} random={scores[0]-scores[2]:+.6f} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed/done*(len(windows)-done))}",
            flush=True,
        )

    mrrt_gain = np.asarray([r["base_objective"] - r["mrrt_objective"] for r in records])
    random_gain = np.asarray([r["base_objective"] - r["random_objective"] for r in records])
    summary = {
        "n": len(records),
        "mrrt_improved": int(np.sum(mrrt_gain > 0)),
        "random_improved": int(np.sum(random_gain > 0)),
        "mrrt_mean_gain": float(mrrt_gain.mean()),
        "random_mean_gain": float(random_gain.mean()),
        "mrrt_better_than_random": int(np.sum(mrrt_gain > random_gain)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        episodes=np.asarray(episodes), starts=np.asarray(starts, dtype=np.int64),
        base=np.stack(base_rows), mrrt=np.stack(mrrt_rows),
        mrrt_random=np.stack(random_rows),
        protocol_json=np.asarray(json.dumps(vars(args))),
        records_json=np.asarray(json.dumps(records)),
        summary_json=np.asarray(json.dumps(summary)),
    )
    print(json.dumps(summary, indent=2))
    print(f"saved {args.out}\nMRRT_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
