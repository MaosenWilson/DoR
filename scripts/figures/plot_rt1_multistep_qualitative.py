#!/usr/bin/env python3
"""Compose fixed-protocol RT-1 qualitative image plates for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_labels(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _build_plate(
    archive,
    manifest: dict,
    scene_indices: list[int],
    method_labels: list[str],
    horizons: list[int],
    output: Path,
) -> None:
    methods = {item["label"]: item for item in manifest["methods"]}
    missing = [label for label in method_labels if label not in methods]
    if missing:
        raise KeyError(f"methods absent from manifest: {missing}")
    scenes = manifest["scenes"]
    for index in scene_indices:
        if not 0 <= index < len(scenes):
            raise IndexError(f"scene index {index} outside [0,{len(scenes)})")
    for horizon in horizons:
        if horizon not in manifest["horizons_exported"]:
            raise ValueError(f"horizon {horizon} was not exported")

    row_labels = ["Ground truth", *method_labels]
    rows_per_scene = len(row_labels)
    ncols = len(horizons)
    fig = plt.figure(figsize=(7.2, max(2.4, 0.72 * rows_per_scene * len(scene_indices))))
    outer = fig.add_gridspec(
        nrows=len(scene_indices),
        ncols=1,
        left=0.13,
        right=0.995,
        bottom=0.025,
        top=0.965,
        hspace=0.16,
    )
    gt = archive["ground_truth"]
    for block_index, scene_index in enumerate(scene_indices):
        scene = scenes[scene_index]
        grid = outer[block_index, 0].subgridspec(
            rows_per_scene, ncols, hspace=0.025, wspace=0.025
        )
        first_axis = None
        for row_offset, label in enumerate(row_labels):
            frames = gt[scene_index] if label == "Ground truth" else archive[methods[label]["array_key"]][scene_index]
            for column, horizon in enumerate(horizons):
                axis = fig.add_subplot(grid[row_offset, column])
                if first_axis is None:
                    first_axis = axis
                axis.imshow(frames[horizon - 1])
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if row_offset == 0:
                    axis.set_title(f"$t={horizon}$", fontsize=7, pad=2)
                if column == 0:
                    color = "#1565c0" if label.lower().startswith("ours") else "#111111"
                    axis.set_ylabel(
                        label,
                        fontsize=7,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=5,
                        color=color,
                        fontweight="bold" if color != "#111111" else "normal",
                    )
        assert first_axis is not None
        first_axis.text(
            -0.52,
            1.15,
            scene.get("scene", f"Scene {scene_index + 1}"),
            transform=first_axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    _save(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--methods", required=True, help="comma-separated labels in display order")
    parser.add_argument("--horizons", default="2,4,6,7")
    parser.add_argument("--main_scenes", default="0,1")
    parser.add_argument("--supp_scenes", default="0,1,2,3")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir)
    with np.load(args.archive, allow_pickle=False) as archive:
        _build_plate(
            archive,
            manifest,
            _parse_ints(args.main_scenes),
            _parse_labels(args.methods),
            _parse_ints(args.horizons),
            output / "fig_rt1_multistep_qualitative",
        )
        _build_plate(
            archive,
            manifest,
            _parse_ints(args.supp_scenes),
            _parse_labels(args.methods),
            _parse_ints(args.horizons),
            output / "fig_rt1_multistep_qualitative_supp",
        )
    print(f"saved figures under {output}\nRT1_QUAL_PLOT_OK", flush=True)


if __name__ == "__main__":
    main()
