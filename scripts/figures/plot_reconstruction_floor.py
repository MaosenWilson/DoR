#!/usr/bin/env python3
"""Plot cross-tokenizer reconstruction floors and matched qualitative examples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _assignments(values: list[str], value_type=str) -> dict:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=VALUE, got {value!r}")
        label, raw = value.split("=", 1)
        if label in parsed:
            raise ValueError(f"duplicate label {label!r}")
        parsed[label] = value_type(raw)
    return parsed


def _save(fig: plt.Figure, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="append", default=[], help="repeat LABEL=NPZ")
    parser.add_argument("--floor", action="append", default=[], help="repeat LABEL=LPIPS")
    parser.add_argument("--example_index", type=int, default=0)
    parser.add_argument("--residual_max", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    examples = _assignments(args.example, Path)
    floors = _assignments(args.floor, float)
    if list(examples) != list(floors):
        raise ValueError("--example and --floor labels must match and preserve the same order")
    if not examples:
        raise ValueError("provide at least one tokenizer")
    if args.residual_max <= 0:
        raise ValueError("residual_max must be positive")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    labels = list(examples)
    fig = plt.figure(figsize=(7.2, 2.25 + 0.55 * len(labels)))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.9, 2.5],
        left=0.07,
        right=0.98,
        bottom=0.12,
        top=0.90,
        wspace=0.35,
    )

    bar_axis = fig.add_subplot(outer[0, 0])
    y = np.arange(len(labels))
    values = np.asarray([floors[label] for label in labels])
    colors = ["#9aa6b2", "#5b8db8", "#d18f62"][: len(labels)]
    if len(colors) < len(labels):
        colors.extend(["#7d8b99"] * (len(labels) - len(colors)))
    bar_axis.barh(y, values, color=colors, height=0.62)
    bar_axis.set_yticks(y, labels)
    bar_axis.invert_yaxis()
    bar_axis.set_xlabel("Reconstruction floor (LPIPS-VGG)")
    bar_axis.set_title("(a) Non-zero floor", loc="left", fontsize=8, fontweight="bold")
    bar_axis.grid(axis="x", alpha=0.22, linewidth=0.5)
    for index, value in enumerate(values):
        bar_axis.text(value + values.max() * 0.025, index, f"{value:.3f}", va="center", fontsize=7)
    bar_axis.set_xlim(0, values.max() * 1.24)

    image_grid = outer[0, 1].subgridspec(len(labels), 3, hspace=0.08, wspace=0.04)
    image_handle = None
    source_rows = []
    for row, label in enumerate(labels):
        with np.load(examples[label], allow_pickle=False) as payload:
            if not 0 <= args.example_index < len(payload["gt"]):
                raise IndexError(f"example_index={args.example_index} invalid for {label}")
            gt = payload["gt"][args.example_index].astype(np.float32) / 255.0
            reachable = payload["reachable"][args.example_index].astype(np.float32) / 255.0
            residual = np.abs(gt - reachable).mean(axis=-1)
            source_rows.append(
                {
                    "tokenizer": label,
                    "reported_floor_lpips": floors[label],
                    "example_lpips": float(payload["lpips"][args.example_index]),
                    "example_mse": float(payload["mse"][args.example_index]),
                    "scene": str(payload["scene"][args.example_index]),
                    "episode": str(payload["episode"][args.example_index]),
                    "start": int(payload["start"][args.example_index]),
                    "horizon": int(payload["horizon"][args.example_index]),
                }
            )
        for column, image in enumerate((gt, reachable, residual)):
            axis = fig.add_subplot(image_grid[row, column])
            if column < 2:
                axis.imshow(image)
            else:
                image_handle = axis.imshow(
                    image,
                    cmap="magma",
                    vmin=0.0,
                    vmax=args.residual_max,
                    interpolation="nearest",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(("Raw GT", "Reachable target", r"$|GT-RC|$ mean RGB")[column], fontsize=7)
            if column == 0:
                axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=5, fontsize=7)
    title_axis = fig.add_subplot(outer[0, 1], frameon=False)
    title_axis.set_xticks([])
    title_axis.set_yticks([])
    title_axis.set_title("(b) Matched reachable targets", loc="left", fontsize=8, fontweight="bold", pad=10)
    if image_handle is not None:
        colorbar = fig.colorbar(image_handle, ax=title_axis, fraction=0.035, pad=0.02)
        colorbar.ax.tick_params(labelsize=6)
        colorbar.set_label("absolute error", fontsize=6)

    output = Path(args.output)
    _save(fig, output)
    source = output.with_name(output.name + "_source.csv")
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    print(f"saved {output}.[svg|pdf|png|tiff]\nsaved {source}\nFLOOR_FIGURE_OK", flush=True)


if __name__ == "__main__":
    main()
