"""Export a deterministic, episode-disjoint VP2 evaluation manifest.

Run with ``external_wm/venv_h5compat/bin/python``.  This process is the only
component that touches the released HDF5 file; exported NPZ windows are then read
by the normal GPU environment.  Progress includes elapsed time and ETA without
requiring extra dependencies in the isolated HDF5 environment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


CONTEXT_LENGTH = 2
IMAGE_KEY = "agentview_shift_2_image"


def _progress(done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = elapsed / max(done, 1)
    eta = rate * (total - done)
    width = 24
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[VP2 export {bar}] {done}/{total} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", end="", flush=True)
    if done == total:
        print(flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episode_start", type=int, default=4900)
    parser.add_argument("--n_contexts", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7301)
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
        for index in range(args.n_contexts):
            episode = f"demo_{args.episode_start + index}"
            group = handle[f"data/{episode}"]
            total = int(group["actions"].shape[0])
            if total < length:
                raise ValueError(f"{episode} is shorter than one context window")
            start = int(rng.integers(0, total - length + 1))
            path = output_dir / f"{episode}_s{start:02d}_h{args.horizon}.npz"
            if path.exists() and not args.overwrite:
                with np.load(path, allow_pickle=False) as cached:
                    image = np.asarray(cached["image"])
                    action = np.asarray(cached["action"])
            else:
                image = np.asarray(group[f"obs/{IMAGE_KEY}"][start:start + length])
                action = np.asarray(group["actions"][start:start + length], dtype=np.float32)
                np.savez_compressed(
                    path,
                    image=image,
                    action=action,
                    episode=np.asarray(episode),
                    start=np.asarray(start, dtype=np.int64),
                )
            if image.shape != (length, 256, 256, 3) or action.shape != (length, 4):
                raise RuntimeError(f"unexpected export shapes for {episode}: {image.shape}, {action.shape}")
            entries.append({"episode": episode, "start": start, "window_npz": str(path)})
            _progress(index + 1, args.n_contexts, started)
    payload = {
        "protocol": "VP2 episode-disjoint fixed-context export v1",
        "hdf5": str(Path(args.hdf5).resolve()),
        "context_length": CONTEXT_LENGTH,
        "horizon": args.horizon,
        "seed": args.seed,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {manifest_path}\nVP2_MANIFEST_EXPORT_OK", flush=True)


if __name__ == "__main__":
    main()
