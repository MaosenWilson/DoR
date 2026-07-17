"""Cache frozen RAFT motion features for RCAV Gate A2.2."""
import argparse
import os
import time

import numpy as np
import torch

from dor.action_observability import motion_oracle_features
from dor.constants import DATA_DIR, ROOT
from dor.episodes import list_episodes, load_episode
from dor.grpo import _flow, _get_raft


def csv_ints(value):
    return tuple(int(x) for x in value.split(",") if x)


def pool_sizes(value):
    result = []
    for item in value.split(","):
        h, w = item.lower().split("x")
        result.append((int(h), int(w)))
    return tuple(result)


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,2,3,4")
    ap.add_argument("--pools", default="4x5,8x10")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_episodes", type=int, default=0, help="smoke only; 0 uses all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/outputs/rcav/action_motion_oracle.npz")
    args = ap.parse_args()
    horizons = csv_ints(args.horizons)
    pools = pool_sizes(args.pools)
    if not horizons or not pools or args.batch_size < 1:
        raise ValueError("horizons, pools and batch_size must be non-empty/positive")

    paths = list_episodes()
    if args.max_episodes > 0:
        paths = paths[:args.max_episodes]
    if len(paths) < 5:
        raise RuntimeError(f"motion oracle requires >=5 episodes under {DATA_DIR}")
    raft = _get_raft(args.device)

    features = {(h, f"{p[0]}x{p[1]}"): [] for h in horizons for p in pools}
    starts = {h: [] for h in horizons}
    row_episodes = {h: [] for h in horizons}
    row_steps = {h: [] for h in horizons}
    state_dims = {}
    actions_all, frame_episode, frame_step, names = [], [], [], []
    global_offset = 0
    completed = 0
    total = len(paths) * len(horizons)
    t0 = time.time()
    print(
        f"[setup] episodes={len(paths)} horizons={horizons} pools={pools} "
        f"batch={args.batch_size} device={args.device}",
        flush=True,
    )
    for ei, path in enumerate(paths):
        images, actions = load_episode(path)
        names.append(os.path.basename(path))
        actions_all.append(actions.astype(np.float32))
        frame_episode.append(np.full(len(images), ei, dtype=np.int32))
        frame_step.append(np.arange(len(images), dtype=np.int32))
        for horizon in horizons:
            local_steps = np.arange(1, len(images) - horizon, dtype=int)
            starts[horizon].append(global_offset + local_steps)
            row_episodes[horizon].append(np.full(len(local_steps), ei, dtype=np.int32))
            row_steps[horizon].append(local_steps.astype(np.int32))
            per_pool = {f"{p[0]}x{p[1]}": [] for p in pools}
            for first in range(0, len(local_steps), args.batch_size):
                t = local_steps[first:first + args.batch_size]
                current = torch.from_numpy(images[t]).float().div(255).permute(0, 3, 1, 2).to(args.device)
                future = torch.from_numpy(images[t + horizon]).float().div(255).permute(0, 3, 1, 2).to(args.device)
                flow = _flow(raft, current, future)
                for pool in pools:
                    pool_name = f"{pool[0]}x{pool[1]}"
                    x, state_dim = motion_oracle_features(current, flow, pool)
                    per_pool[pool_name].append(x.cpu().numpy().astype(np.float32))
                    state_dims[(horizon, pool_name)] = state_dim
            for pool_name, chunks in per_pool.items():
                features[(horizon, pool_name)].append(np.concatenate(chunks))
            completed += 1
            elapsed = time.time() - t0
            eta = elapsed / completed * (total - completed)
            filled = 20 * completed // total
            print(
                f"[flow {'#' * filled:<20}] {completed}/{total} {names[-1]} h={horizon} "
                f"pairs={len(local_steps)} elapsed={hms(elapsed)} eta={hms(eta)}",
                flush=True,
            )
        global_offset += len(images)

    payload = {
        "actions": np.concatenate(actions_all),
        "episode_id": np.concatenate(frame_episode),
        "step": np.concatenate(frame_step),
        "episode_names": np.asarray(names, dtype=str),
        "horizons": np.asarray(horizons, dtype=np.int64),
        "pools": np.asarray([f"{p[0]}x{p[1]}" for p in pools], dtype=str),
    }
    for horizon in horizons:
        payload[f"starts_h{horizon}"] = np.concatenate(starts[horizon])
        payload[f"episodes_h{horizon}"] = np.concatenate(row_episodes[horizon])
        payload[f"steps_h{horizon}"] = np.concatenate(row_steps[horizon])
        for pool in pools:
            pool_name = f"{pool[0]}x{pool[1]}"
            prefix = f"h{horizon}_{pool_name}"
            payload[f"features_{prefix}"] = np.concatenate(features[(horizon, pool_name)])
            payload[f"state_dim_{prefix}"] = np.asarray(state_dims[(horizon, pool_name)])

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"saved {out}\nACTION_MOTION_ORACLE_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
