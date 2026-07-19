"""Export a deterministic, episode-disjoint RoboDesk evaluation manifest.

RoboDesk differs from RoboSuite PushCenter in three ways this exporter handles:
  * actions are 5-dimensional (not 4);
  * frames live under ``obs/camera_image`` and are stored with a compression
    filter that rejects sliced reads, so each episode is read in full then sliced
    (matching upstream ``preprocess_vp2``);
  * episode-disjoint contexts are drawn from the official ``mask/valid`` split so
    they never overlap the 2250 training demos.

Each exported window also stores the drawer task-state scalar
``states[:, drawer_index]`` (Stage-0 found index 18 correlates 0.974 with the
task reward) so downstream branch-value diagnostics can use a task-relevant
utility instead of only whole-frame pixel distance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


CONTEXT_LENGTH = 2
IMAGE_KEY = "camera_image"
ACTION_DIM = 5


def _progress(done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed / max(done, 1) * (total - done)
    bar = "#" * int(24 * done / total) + "-" * (24 - int(24 * done / total))
    print(f"\r[RoboDesk export {bar}] {done}/{total} "
          f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", end="", flush=True)
    if done == total:
        print(flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n_contexts", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--drawer_index", type=int, default=18,
                        help="states[:, drawer_index]; Stage-0 default 18")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--split", default="valid", choices=["valid", "train"],
                        help="episode-disjoint contexts come from mask/valid")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_contexts < 1 or args.horizon < 1:
        raise ValueError("n_contexts and horizon must be positive")
    output_dir = Path(args.out_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    length = CONTEXT_LENGTH + args.horizon
    entries: list[dict] = []
    started = time.monotonic()
    with h5py.File(args.hdf5, "r", swmr=False, libver="latest") as handle:
        demos = [e.decode("utf-8") if isinstance(e, bytes) else str(e)
                 for e in np.asarray(handle[f"mask/{args.split}"][()])]
        if args.n_contexts > len(demos):
            raise ValueError(f"requested {args.n_contexts} > {len(demos)} {args.split} demos")
        chosen = [demos[i] for i in rng.permutation(len(demos))[: args.n_contexts]]
        for index, episode in enumerate(chosen):
            group = handle[f"data/{episode}"]
            total = int(group["actions"].shape[0])
            if total < length:
                raise ValueError(f"{episode} shorter than one context window")
            start = int(rng.integers(0, total - length + 1))
            path = output_dir / f"{episode}_s{start:02d}_h{args.horizon}.npz"
            if path.exists() and not args.overwrite:
                entries.append({"episode": episode, "start": start, "window_npz": str(path)})
                _progress(index + 1, len(chosen), started)
                continue
            image = np.asarray(group[f"obs/{IMAGE_KEY}"][()])[start:start + length]
            action = np.asarray(group["actions"][start:start + length], dtype=np.float32)
            states = np.asarray(group["states"][()])[start:start + length]
            reward = np.asarray(group["rewards"][()])[start:start + length]
            if image.shape != (length, 256, 256, 3) or action.shape != (length, ACTION_DIM):
                raise RuntimeError(f"unexpected shapes for {episode}: {image.shape}, {action.shape}")
            np.savez_compressed(
                path,
                image=image,
                action=action,
                episode=np.asarray(episode),
                start=np.asarray(start, dtype=np.int64),
                task_state=states[:, args.drawer_index].astype(np.float32),
                reward=reward.astype(np.float32),
            )
            entries.append({"episode": episode, "start": start, "window_npz": str(path)})
            _progress(index + 1, len(chosen), started)
    payload = {
        "protocol": "RoboDesk episode-disjoint fixed-context export v1",
        "hdf5": str(Path(args.hdf5).resolve()),
        "image_key": IMAGE_KEY,
        "action_dim": ACTION_DIM,
        "drawer_index": args.drawer_index,
        "split": args.split,
        "context_length": CONTEXT_LENGTH,
        "horizon": args.horizon,
        "seed": args.seed,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {manifest_path}\nROBODESK_MANIFEST_EXPORT_OK", flush=True)


if __name__ == "__main__":
    main()
