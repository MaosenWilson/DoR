#!/usr/bin/env python3
"""Summarize flattened RT-1 action statistics from converted `.npz` episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vote2world.data.action_schema import ActionSchema


QUANTILES = [0, 10, 25, 50, 75, 90, 95, 99, 100]


def stats(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "exact_zero_ratio": float(np.mean(values == 0)),
        "quantiles": {f"p{q:02d}": float(np.percentile(values, q)) for q in QUANTILES},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/vote2world/action_schema.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/vote2world_action_statistics.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    schema = ActionSchema.from_file(args.schema)
    files = sorted(args.npz_dir.glob("*.npz"))
    if args.limit is not None:
        files = files[: args.limit]
    actions = []
    for path in files:
        with np.load(path) as data:
            if "action" in data:
                arr = np.asarray(data["action"], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == schema.action_dim:
                    actions.append(arr)
    if not actions:
        raise FileNotFoundError(f"no converted npz action arrays found under {args.npz_dir}")
    all_actions = np.concatenate(actions, axis=0)
    result: dict[str, object] = {"num_files": len(files), "num_steps": int(all_actions.shape[0]), "fields": {}}
    for name, field in schema.fields.items():
        result["fields"][name] = stats(all_actions[:, field.slice()])
    result["groups"] = {
        "arm_translation_norm": stats(np.linalg.norm(schema.slice_action(all_actions, "world_vector"), axis=1)),
        "arm_rotation_norm": stats(np.linalg.norm(schema.slice_action(all_actions, "rotation_delta"), axis=1)),
        "gripper_abs": stats(np.abs(schema.slice_action(all_actions, "gripper_closedness_action"))),
        "base_translation_norm": stats(np.linalg.norm(schema.slice_action(all_actions, "base_displacement_vector"), axis=1)),
        "base_rotation_abs": stats(np.abs(schema.slice_action(all_actions, "base_displacement_vertical_rotation"))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "num_steps": int(all_actions.shape[0])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

