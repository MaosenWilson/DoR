"""Cache episode-labelled FSQ transition features for RCAV Gate A."""
import argparse
import os
import time

import numpy as np
import torch

from dor.action_verifier import transition_features
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
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pool_h", type=int, default=4)
    ap.add_argument("--pool_w", type=int, default=5)
    ap.add_argument("--max_episodes", type=int, default=0,
                    help="smoke only; 0 uses every episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/action_transitions.npz")
    args = ap.parse_args()
    if args.stride < 1 or args.batch_size < 1:
        raise ValueError("stride and batch_size must be positive")

    paths = list_episodes()
    if args.max_episodes > 0:
        paths = paths[:args.max_episodes]
    if len(paths) < 5:
        raise RuntimeError(f"Gate A requires >=5 episodes under {DATA_DIR}, found {len(paths)}")
    tok = load_tokenizer(args.device)
    features, actions, episode_ids, steps = [], [], [], []
    episode_names = []
    t0 = time.time()
    print(f"[setup] episodes={len(paths)} stride={args.stride} batch={args.batch_size}", flush=True)
    for ei, path in enumerate(paths):
        images, act = load_episode(path)
        name = os.path.basename(path)
        episode_names.append(name)
        frame_codes = []
        for start in range(0, len(images), args.batch_size):
            raw = images[start:start + args.batch_size]
            frames = torch.from_numpy(raw).float().div(255.0).permute(0, 3, 1, 2).to(args.device)
            idx = encode_indices(tok, frames)
            frame_codes.append(tok.indices_to_codes(idx).float().cpu())
        codes = torch.cat(frame_codes, 0)
        ids = np.arange(0, len(images) - 1, args.stride, dtype=int)
        feat = transition_features(
            codes[ids], codes[ids + 1], (args.pool_h, args.pool_w)
        ).numpy().astype(np.float32)
        features.append(feat)
        actions.append(act[ids].astype(np.float32))
        episode_ids.append(np.full(len(ids), ei, dtype=np.int32))
        steps.append(ids.astype(np.int32))
        elapsed = time.time() - t0
        rate = (ei + 1) / max(elapsed, 1e-6)
        eta = (len(paths) - ei - 1) / max(rate, 1e-6)
        bar = "#" * (20 * (ei + 1) // len(paths))
        print(
            f"[cache {bar:<20}] {ei + 1}/{len(paths)} {name} "
            f"transitions={len(ids)} elapsed={hms(elapsed)} eta={hms(eta)}",
            flush=True,
        )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        features=np.concatenate(features),
        actions=np.concatenate(actions),
        episode_id=np.concatenate(episode_ids),
        step=np.concatenate(steps),
        episode_names=np.asarray(episode_names, dtype=str),
        pool_hw=np.asarray([args.pool_h, args.pool_w], dtype=np.int64),
        stride=np.asarray(args.stride, dtype=np.int64),
    )
    print(f"saved {out}\nACTION_VERIFIER_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
