"""Export one VP2 RoboSuite window through the isolated HDF5-1.12 reader.

Run this script only with ``external_wm/venv_h5compat/bin/python``. It intentionally
depends on h5py/numpy only and writes a small immutable NPZ consumed by the regular
DoR/iVideoGPT environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


CONTEXT_LENGTH = 2
IMAGE_KEY = "agentview_shift_2_image"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    length = CONTEXT_LENGTH + args.horizon
    with h5py.File(args.hdf5, "r", swmr=False, libver="latest") as handle:
        group = handle[f"data/{args.episode}"]
        total = int(group["actions"].shape[0])
        if args.start < 0 or args.start + length > total:
            raise ValueError(f"window [{args.start}, {args.start + length}) exceeds episode length {total}")
        image = np.asarray(group[f"obs/{IMAGE_KEY}"][args.start:args.start + length])
        action = np.asarray(group["actions"][args.start:args.start + length], dtype=np.float32)
    if image.shape != (length, 256, 256, 3) or action.shape != (length, 4):
        raise RuntimeError(f"unexpected export shapes image={image.shape}, action={action.shape}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image=image,
        action=action,
        episode=np.asarray(args.episode),
        start=np.asarray(args.start, dtype=np.int64),
    )
    print(f"VP2_WINDOW_EXPORT_OK episode={args.episode} start={args.start} horizon={args.horizon} out={output}")


if __name__ == "__main__":
    main()
