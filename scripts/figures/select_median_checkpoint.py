#!/usr/bin/env python3
"""Freeze a representative median-seed checkpoint before qualitative export."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def _seed_from_report(report: dict, path: str) -> int:
    if report.get("seed") is not None:
        return int(report["seed"])
    match = re.search(r"_s(\d+)\.json$", path)
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_glob", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--aggregation", default="window_macro")
    parser.add_argument("--metric", default="lpips")
    parser.add_argument("--checkpoint_pattern", required=True, help="must contain {seed}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if "{seed}" not in args.checkpoint_pattern:
        raise ValueError("--checkpoint_pattern must contain {seed}")

    rows = []
    for path in sorted(glob.glob(args.eval_glob)):
        report = json.loads(Path(path).read_text())
        if report.get("arm") != args.arm:
            continue
        seed = _seed_from_report(report, path)
        try:
            value = float(report["aggregate"][args.aggregation][args.metric])
        except KeyError as error:
            raise KeyError(f"{path} lacks aggregate/{args.aggregation}/{args.metric}") from error
        rows.append({"seed": seed, "value": value, "evaluation": str(Path(path).resolve())})
    if not rows:
        raise ValueError(f"no reports for arm={args.arm!r} under {args.eval_glob!r}")
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("duplicate seeds in evaluation reports")

    median = float(np.median([row["value"] for row in rows]))
    selected = min(rows, key=lambda row: (abs(row["value"] - median), row["seed"]))
    checkpoint = Path(args.checkpoint_pattern.format(seed=selected["seed"])).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    payload = {
        "protocol": "global held-out median-seed checkpoint selection v1",
        "arm": args.arm,
        "aggregation": args.aggregation,
        "metric": args.metric,
        "median": median,
        "rows": sorted(rows, key=lambda row: row["seed"]),
        "selected_seed": selected["seed"],
        "selected_value": selected["value"],
        "checkpoint": str(checkpoint),
        "rule": "closest to the median held-out metric; ties resolved by lower seed",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"selected seed={selected['seed']} {args.metric}={selected['value']:.6f} "
        f"median={median:.6f}\ncheckpoint={checkpoint}\nsaved {output}\n"
        "MEDIAN_CHECKPOINT_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
