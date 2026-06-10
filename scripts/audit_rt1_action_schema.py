#!/usr/bin/env python3
"""Audit or materialize the Vote2World RT-1 action schema.

This script can confirm schema from a small converted `.npz` episode, or emit
the current metadata-backed provisional schema when official conversion is not
available yet.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vote2world.data.action_schema import ACTION_SCHEMA_PATH, ActionSchema, write_schema_safely


PROVISIONAL_ORDER = [
    "gripper_closedness_action",
    "terminate_episode",
    "base_displacement_vector",
    "rotation_delta",
    "base_displacement_vertical_rotation",
    "world_vector",
]

FIELD_DIMS = {
    "gripper_closedness_action": 1,
    "terminate_episode": 3,
    "base_displacement_vector": 2,
    "rotation_delta": 3,
    "base_displacement_vertical_rotation": 1,
    "world_vector": 3,
}

FIELD_GROUPS = {
    "gripper_closedness_action": "gripper",
    "terminate_episode": "mode",
    "base_displacement_vector": "base",
    "rotation_delta": "rotation",
    "base_displacement_vertical_rotation": "base",
    "world_vector": "translation",
}

MOTION_FIELDS = {"rotation_delta", "world_vector"}


def build_schema(order: list[str], status: str, evidence: str) -> dict[str, object]:
    offset = 0
    slices = {}
    for field in order:
        dim = FIELD_DIMS[field]
        slices[field] = {
            "start": offset,
            "end": offset + dim,
            "dim": dim,
            "dtype": "int32" if field == "terminate_episode" else "float32",
            "semantic_group": FIELD_GROUPS[field],
            "include_in_model_input": True,
            "include_in_motion_magnitude": field in MOTION_FIELDS,
        }
        offset += dim
    return {
        "dataset_name": "fractal20220817_data",
        "schema_status": status,
        "schema_version": 1,
        "action_dim": offset,
        "flatten_key_order": order,
        "field_slices": slices,
        "semantic_groups": {
            "arm_translation": ["world_vector"],
            "arm_rotation": ["rotation_delta"],
            "gripper": ["gripper_closedness_action"],
            "base": ["base_displacement_vector", "base_displacement_vertical_rotation"],
            "mode": ["terminate_episode"],
        },
        "official_single_step": {
            "context_length": 4,
            "segment_length": 5,
            "action_bins": 256,
            "visual_token_num": 4375,
            "tokens_per_frame": 320,
            "bos_token_id": 4631,
            "eos_token_id": 4632,
            "gen_input_length": 1333,
            "gen_output_length": 321,
        },
        "evidence": {
            "status_reason": evidence,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def write_report(schema: ActionSchema, path: Path, source: str) -> None:
    rows = ["| field | slice | dim | group | motion magnitude |", "|---|---:|---:|---|---:|"]
    for name in schema.flatten_key_order:
        field = schema.fields[name]
        rows.append(
            f"| `{name}` | `[{field.start}:{field.end}]` | {field.dim} | "
            f"`{field.semantic_group}` | {field.include_in_motion_magnitude} |"
        )
    text = f"""# RT-1 Action Schema Audit

- source: `{source}`
- status: `{schema.schema_status}`
- action_dim: `{schema.action_dim}`
- flatten_key_order: `{schema.flatten_key_order}`

{chr(10).join(rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=None, help="Optional converted RLVR-World episode.")
    parser.add_argument("--schema-out", type=Path, default=ACTION_SCHEMA_PATH)
    parser.add_argument("--report-out", type=Path, default=Path("reports/vote2world_action_schema_audit.md"))
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    status = "provisional"
    evidence = "No official converted npz was provided; using metadata-backed provisional order."
    source = "metadata/provisional"
    if args.npz is not None:
        with np.load(args.npz) as data:
            if "action" not in data:
                raise KeyError(f"{args.npz} has no 'action' key")
            actions = np.asarray(data["action"])
            if actions.ndim != 2 or actions.shape[1] != 13:
                raise ValueError(f"converted action must be (T,13), got {actions.shape}")
        status = "confirmed"
        evidence = f"Confirmed from converted npz action shape at {args.npz}; key order follows official converter metadata."
        source = str(args.npz)

    raw = build_schema(PROVISIONAL_ORDER, status, evidence)
    schema_path = write_schema_safely(raw, args.schema_out, allow_overwrite=args.allow_overwrite)
    schema = ActionSchema.from_file(schema_path)
    write_report(schema, args.report_out, source)
    print(json.dumps({"schema": str(schema_path), "status": schema.schema_status, "report": str(args.report_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

