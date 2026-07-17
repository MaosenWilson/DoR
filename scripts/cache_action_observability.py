"""Cache frame-level FSQ codes for the RCAV observability audit."""
import argparse
import os
import time

import numpy as np
import torch

from dor.constants import DATA_DIR, ROOT
from dor.episodes import list_episodes, load_episode
from dor.models import load_tokenizer
from dor.tokenization import encode_indices


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_episodes", type=int, default=0, help="smoke only; 0 uses all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/action_observability_codes.npz")
    args = ap.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    paths = list_episodes()
    if args.max_episodes > 0:
        paths = paths[:args.max_episodes]
    if len(paths) < 5:
        raise RuntimeError(f"observability audit requires >=5 episodes under {DATA_DIR}")

    tokenizer = load_tokenizer(args.device)
    all_codes, all_actions, all_episode, all_steps = [], [], [], []
    names = []
    t0 = time.time()
    print(f"[setup] episodes={len(paths)} batch={args.batch_size} device={args.device}", flush=True)
    for ei, path in enumerate(paths):
        images, actions = load_episode(path)
        chunks = []
        for start in range(0, len(images), args.batch_size):
            raw = images[start:start + args.batch_size]
            frames = torch.from_numpy(raw).float().div(255).permute(0, 3, 1, 2).to(args.device)
            indices = encode_indices(tokenizer, frames)
            chunks.append(tokenizer.indices_to_codes(indices).float().cpu().numpy())
        codes = np.concatenate(chunks).astype(np.float32)
        all_codes.append(codes)
        all_actions.append(actions.astype(np.float32))
        all_episode.append(np.full(len(codes), ei, dtype=np.int32))
        all_steps.append(np.arange(len(codes), dtype=np.int32))
        names.append(os.path.basename(path))
        elapsed = time.time() - t0
        rate = (ei + 1) / max(elapsed, 1e-6)
        eta = (len(paths) - ei - 1) / max(rate, 1e-6)
        done = 20 * (ei + 1) // len(paths)
        print(
            f"[cache {'#' * done:<20}] {ei + 1}/{len(paths)} {names[-1]} "
            f"frames={len(codes)} elapsed={hms(elapsed)} eta={hms(eta)}",
            flush=True,
        )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        codes=np.concatenate(all_codes),
        actions=np.concatenate(all_actions),
        episode_id=np.concatenate(all_episode),
        step=np.concatenate(all_steps),
        episode_names=np.asarray(names, dtype=str),
    )
    print(f"saved {out}\nACTION_OBSERVABILITY_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
