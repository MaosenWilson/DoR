#!/usr/bin/env python3
"""Generate manuscript figures from audited AAAI-2027 result JSON files."""

from __future__ import annotations

import glob
import json
import math
import re
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tmp" / "aaai_results_raw"
FIGS = ROOT / "docs" / "AuthorKit27" / "Figures"
SOURCE = FIGS / "source_data"


def seed_of(path: str) -> int:
    match = re.search(r"_s(\d+)\.json$", path)
    return int(match.group(1)) if match else -1


def load_sweep(pattern: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(str(pattern)), key=seed_of):
        data = json.load(open(path))
        run = next(iter(data["run"].values()))
        runs.append(run)
    return runs


def mean_std_curves(runs: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.array(runs[0]["step"], dtype=float)
    values = np.array([run[metric] for run in runs], dtype=float)
    return steps, values.mean(axis=0), values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(steps)


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves() -> None:
    arms = [
        ("Seq raw", DATA / "msp_step30_seq" / "sweep_raw_msp_s*.json", "#8c8c8c", "o"),
        ("Seq RC (Ours)", DATA / "msp_step30_seq" / "sweep_rc_msp_s*.json", "#3b82f6", "s"),
        ("Return RC (Ours, full)", DATA / "msp_step30_return_hkl00" / "sweep_rc_msp_s*.json", "#f59e0b", "^"),
        ("Gain return (Ablation)", DATA / "msp_step30_gain_return_a05" / "sweep_rc_msp_s*.json", "#10b981", "D"),
    ]
    rlvr = {"eval_lpips": 0.21154583369692165, "eval_lpips_last": 0.21547726728022099}
    base = {"eval_lpips": 0.21570572862401605, "eval_lpips_last": 0.22598793730139732}

    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)
    panels = [("eval_lpips", "LPIPS ↓"), ("eval_lpips_last", "LPIPS-last ↓")]
    for ax, (metric, ylabel) in zip(axes, panels):
        for label, pattern, color, marker in arms:
            runs = load_sweep(pattern)
            steps, mu, sd = mean_std_curves(runs, metric)
            for run_index, run in enumerate(runs):
                for step, value in zip(run["step"], run[metric]):
                    source_rows.append({
                        "figure": "fig_msp_learning_curves",
                        "panel_metric": metric,
                        "arm": label,
                        "run_index": run_index,
                        "step": step,
                        "value": value,
                    })
            ax.plot(steps, mu, color=color, marker=marker, lw=1.8, ms=4, label=label)
            ax.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)
        ax.axhline(rlvr[metric], color="#ef4444", ls="--", lw=1.2, label="RLVR ckpt" if metric == "eval_lpips" else None)
        ax.axhline(base[metric], color="#6b7280", ls=":", lw=1.1, label="Base ckpt" if metric == "eval_lpips" else None)
        ax.set_xlabel("Post-training steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.22, lw=0.5)
        ax.set_xlim(-1, 31)
        ax.set_xticks([0, 10, 20, 30])
    axes[0].set_title("(a) Full rollout fidelity", loc="left", fontsize=10)
    axes[1].set_title("(b) Last-frame fidelity", loc="left", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    for metric, value in rlvr.items():
        source_rows.append({
            "figure": "fig_msp_learning_curves",
            "panel_metric": metric,
            "arm": "RLVR checkpoint",
            "run_index": "eval-only",
            "step": "constant",
            "value": value,
        })
    for metric, value in base.items():
        source_rows.append({
            "figure": "fig_msp_learning_curves",
            "panel_metric": metric,
            "arm": "Base checkpoint",
            "run_index": "eval-only",
            "step": "constant",
            "value": value,
        })
    write_csv(
        SOURCE / "fig_msp_learning_curves_source.csv",
        ["figure", "panel_metric", "arm", "run_index", "step", "value"],
        source_rows,
    )
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.88))
    save(fig, "fig_msp_learning_curves")


def plot_horizon_delta() -> None:
    rows = json.load(open(DATA / "analysis" / "msp_horizon_fullmetrics.json"))["rows"]
    by = {(row["label"], str(row["seed"]), int(row["horizon"])): row for row in rows}
    horizons = sorted({int(row["horizon"]) for row in rows if row["label"] == "seq_rc"})
    metrics = [
        ("lpips", "LPIPS Δ ↓", "#3b82f6"),
        ("mse", "MSE Δ ↓", "#ef4444"),
        ("psnr", "PSNR Δ ↑", "#10b981"),
        ("ssim", "SSIM Δ ↑", "#a855f7"),
    ]

    deltas: dict[str, list[float]] = defaultdict(list)
    for h in horizons:
        seeds = sorted({str(row["seed"]) for row in rows if row["label"] == "seq_rc" and int(row["horizon"]) == h})
        for metric, _, _ in metrics:
            vals = []
            for seed in seeds:
                seq = by[("seq_rc", seed, h)]
                ret = by[("return_rc", seed, h)]
                vals.append(ret[metric] - seq[metric])
            deltas[metric].append(float(np.mean(vals)))

    source_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), sharex=True)
    for ax, (metric, ylabel, color) in zip(axes, metrics):
        vals = np.array(deltas[metric])
        ax.axhline(0, color="#111827", lw=0.8)
        ax.plot(horizons, vals, marker="o", color=color, lw=1.8, ms=4)
        ax.fill_between(horizons, vals, 0, color=color, alpha=0.12)
        ax.set_title(ylabel, fontsize=9)
        ax.set_xlabel("Horizon")
        ax.grid(True, axis="y", alpha=0.22, lw=0.5)
        if metric in {"lpips", "mse"}:
            ax.text(0.04, 0.08, "lower is better", transform=ax.transAxes, fontsize=6.5, color="#4b5563")
        else:
            ax.text(0.04, 0.08, "higher is better", transform=ax.transAxes, fontsize=6.5, color="#4b5563")
    for h in horizons:
        seeds = sorted({str(row["seed"]) for row in rows if row["label"] == "seq_rc" and int(row["horizon"]) == h})
        for seed in seeds:
            seq = by[("seq_rc", seed, h)]
            ret = by[("return_rc", seed, h)]
            for metric, _, _ in metrics:
                source_rows.append({
                    "figure": "fig_horizon_delta",
                    "seed": seed,
                    "horizon": h,
                    "metric": metric,
                    "seq_rc": seq[metric],
                    "return_rc": ret[metric],
                    "delta_return_minus_seq": ret[metric] - seq[metric],
                })
    write_csv(
        SOURCE / "fig_horizon_delta_source.csv",
        ["figure", "seed", "horizon", "metric", "seq_rc", "return_rc", "delta_return_minus_seq"],
        source_rows,
    )
    axes[0].set_ylabel("Return RC - Seq RC")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_horizon_delta")


def plot_rank_flip() -> None:
    files = sorted((DATA / "analysis").glob("*rankflip.json"))
    aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rename = {"pixel(-LPIPS)": "LPIPS", "-MSE": "MSE", "SSIM": "SSIM"}
    for path in files:
        data = json.load(open(path))
        for arm, stats in data["arms"].items():
            name = rename.get(arm, arm)
            aggregate[name]["observed"].append(float(stats["emp_flip"]))
            aggregate[name]["predicted"].append(float(stats["bound"]))
            aggregate[name]["rho"].append(float(stats["rho"]))

    names = ["LPIPS", "MSE", "SSIM"]
    obs = np.array([np.mean(aggregate[name]["observed"]) for name in names])
    pred = np.array([np.mean(aggregate[name]["predicted"]) for name in names])
    rho = np.array([np.mean(aggregate[name]["rho"]) for name in names])
    obs_sd = np.array([np.std(aggregate[name]["observed"], ddof=1) for name in names])

    source_rows = []
    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    colors = ["#3b82f6", "#ef4444", "#10b981"]
    offsets = {"LPIPS": (0.002, 0.004), "MSE": (0.002, -0.005), "SSIM": (0.002, 0.001)}
    for i, name in enumerate(names):
        ax.errorbar(pred[i], obs[i], yerr=obs_sd[i], fmt="o", color=colors[i], capsize=2.5, ms=6)
        dx, dy = offsets[name]
        ax.text(pred[i] + dx, obs[i] + dy, f"{name}\nρ={rho[i]:.2f}", fontsize=7)
        for seed_idx, (observed, predicted, rho_value) in enumerate(zip(
            aggregate[name]["observed"],
            aggregate[name]["predicted"],
            aggregate[name]["rho"],
        )):
            source_rows.append({
                "figure": "fig_rank_flip",
                "metric": name,
                "seed_index": seed_idx,
                "rho": rho_value,
                "predicted_flip": predicted,
                "observed_flip": observed,
            })
    lo = min(pred.min(), obs.min()) - 0.015
    hi = max(pred.max(), obs.max()) + 0.018
    ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Predicted flip $\arccos(\rho)/\pi$")
    ax.set_ylabel("Observed pairwise flip")
    ax.set_title("Rank-flip diagnostic", fontsize=10)
    ax.grid(True, alpha=0.22, lw=0.5)
    write_csv(
        SOURCE / "fig_rank_flip_source.csv",
        ["figure", "metric", "seed_index", "rho", "predicted_flip", "observed_flip"],
        source_rows,
    )
    fig.tight_layout()
    save(fig, "fig_rank_flip")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plot_learning_curves()
    plot_horizon_delta()
    plot_rank_flip()
    (SOURCE / "README.md").write_text(
        "# Source Data for AAAI-2027 Figures\n\n"
        "These files contain the processed source data used to draw the manuscript figures.\n\n"
        "## Files\n"
        "- `fig_rank_flip_source.csv`: per-seed rank-flip diagnostics for Fig. rank-flip.\n"
        "- `fig_msp_learning_curves_source.csv`: per-run post-training curves and eval-only baselines for Fig. learning curves.\n"
        "- `fig_horizon_delta_source.csv`: per-seed, per-horizon values for sequence-level RC and temporal-return RC.\n\n"
        "## Provenance\n"
        "The CSV files were generated by `scripts/figures/plot_aaai_results.py` from audited JSON files under `tmp/aaai_results_raw/`, which mirror server outputs from `/root/autodl-tmp/vote2world/outputs`.\n\n"
        "The five raw sequence-GRPO runs and five calibrated runs are all present locally. The same-candidate RC diagnostic is stored separately as `tmp/aaai_results_raw/analysis/rc_flip_closure.npz`; its 920 rows comprise 184 contexts repeated under five independent candidate-generation seeds and must not be analyzed as 920 independent contexts.\n\n"
        "## Variables\n"
        "- `step`: post-training step.\n"
        "- `horizon`: rollout horizon used in multi-step evaluation.\n"
        "- `delta_return_minus_seq`: `return_rc - seq_rc`; negative is better for LPIPS/MSE/MAE, positive is better for PSNR/SSIM.\n"
        "- `predicted_flip`: theoretical pairwise flip rate `arccos(rho)/pi`.\n"
        "- `observed_flip`: empirical pairwise rank-flip rate.\n\n"
        "- Learning-curve bands and tabulated uncertainty use sample standard deviation (`ddof=1`).\n\n"
        "## Access and licence\n"
        "For submission, these processed figure source data can be included as source data or supplementary files. Raw model outputs remain in the project `outputs/` directory.\n",
    )
    print(f"Wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
