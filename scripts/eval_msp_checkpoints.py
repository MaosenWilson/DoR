"""Re-evaluate frozen RT-1 multi-step checkpoints on unseen episodes.

The original training harness evaluates eight held-out windows sampled after the
24 training windows.  Those windows do not repeat training windows, but some
come from episodes represented in training.  This script reconstructs the
training manifest and evaluates every stride-aligned window from episodes that
were never used for training.  It never updates model parameters.
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from dor.constants import ROOT
from dor.episodes import list_episodes
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges
from dor.multistep import (
    MSP_BASE_DIR,
    MSP_RLVR_DIR,
    MSP_TOK_DIR,
    V_MSP,
    detok_chunked,
    discretize_actions,
    msp_rollout,
    msp_sample_windows,
    msp_window,
)


METRICS = ("lpips", "mse", "psnr", "ssim")


def build_episode_disjoint_windows(paths, T, train_windows, split_seed, stride):
    """Reconstruct training episodes, then return all windows from unseen episodes."""
    train = msp_sample_windows(paths, train_windows, T, seed=split_seed, stride=stride)
    train_episodes = {path for path, _ in train}
    heldout_episodes = [path for path in sorted(paths) if path not in train_episodes]
    windows = []
    for path in heldout_episodes:
        length = np.load(path, allow_pickle=True)["image"].shape[0]
        windows.extend((path, start) for start in range(0, length - T, stride))
    return train, windows


def _manifest(train, windows):
    def rows(items):
        return [
            {"episode": os.path.basename(path), "start": int(start)}
            for path, start in items
        ]

    payload = {"train": rows(train), "evaluation": rows(windows)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def aggregate_rows(rows):
    """Return window-macro and episode-macro summaries without inflating n."""
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    keys = (*METRICS, "lpips_last")
    window_macro = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)
    episode_values = {
        episode: {
            key: float(np.mean([row[key] for row in episode_rows]))
            for key in keys
        }
        for episode, episode_rows in by_episode.items()
    }
    episode_macro = {
        key: float(np.mean([value[key] for value in episode_values.values()]))
        for key in keys
    }
    return {
        "window_macro": window_macro,
        "episode_macro": episode_macro,
        "per_episode": episode_values,
    }


def _load_tokenizer(device):
    import dor.compat  # noqa: F401
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ

    return CompressiveVQModelFSQ.from_pretrained(MSP_TOK_DIR).to(device).eval()


def _load_model(path, device):
    import dor.compat  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32
        )
    except AttributeError as error:
        # Some public safetensors checkpoints were exported without the optional
        # {"format": "pt"} metadata. Transformers 4.42 dereferences that missing
        # dictionary before loading otherwise-valid weights.
        weight_path = os.path.join(path, "model.safetensors")
        metadata_error = "'NoneType' object has no attribute 'get'" in str(error)
        if not metadata_error or not os.path.isfile(weight_path):
            raise
        from safetensors.torch import load_model

        print(
            f"[load] {path}: safetensors metadata is absent; "
            "using strict state-dict fallback",
            flush=True,
        )
        config = AutoConfig.from_pretrained(path)
        model = AutoModelForCausalLM.from_config(config)
        load_model(model, weight_path, strict=True, device="cpu")

    model = model.to(device)
    model.config.use_cache = True
    return model.eval()


@torch.no_grad()
def evaluate_model(model, tokenizer, action_ranges, metric_fn, windows, args, label):
    rows = []
    started = time.time()
    total = len(windows) * args.eval_draws
    completed = 0
    for window_index, (path, start) in enumerate(windows):
        draw_rows = []
        frames, actions = msp_window(path, start, args.T, args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context_tokens, _ = tokenizer.tokenize(frames.unsqueeze(0))
        context = (context_tokens.reshape(1, -1) + V_MSP).long()
        action_tokens = discretize_actions(actions, action_ranges)[1 : args.T] + 2 * V_MSP
        for draw in range(args.eval_draws):
            generation_seed = args.eval_seed + draw * 100_000 + window_index
            dynamics = msp_rollout(
                model,
                context,
                action_tokens,
                args.T - 1,
                args.K,
                seed=generation_seed,
            )
            predictions = detok_chunked(
                tokenizer, context_tokens.expand(args.K, -1, -1), dynamics
            )
            real = frames[1:]
            horizon_rows = []
            # Match RLVR-World's convention: the first predicted frame has no
            # direct verifier reward, so readouts cover horizons 2..T-1.
            for horizon_index in range(1, args.T - 1):
                measured = metric_fn.eval_batch(
                    predictions[:, horizon_index], real[horizon_index]
                )
                horizon_rows.append(
                    {
                        "horizon": horizon_index + 1,
                        **{
                            key: float(np.mean(measured[key]))
                            for key in METRICS
                        },
                    }
                )
            draw_rows.append(horizon_rows)
            completed += 1
            elapsed = time.time() - started
            eta = elapsed / completed * (total - completed)
            print(
                f"[{label}] {_bar(completed / total)} {completed}/{total} "
                f"window={window_index + 1}/{len(windows)} draw={draw + 1}/"
                f"{args.eval_draws} elapsed={_hms(elapsed)} eta={_hms(eta)}",
                flush=True,
            )

        averaged_horizons = []
        for horizon_offset in range(args.T - 2):
            averaged_horizons.append(
                {
                    "horizon": draw_rows[0][horizon_offset]["horizon"],
                    **{
                        key: float(
                            np.mean(
                                [draw[horizon_offset][key] for draw in draw_rows]
                            )
                        )
                        for key in METRICS
                    },
                }
            )
        rows.append(
            {
                "episode": os.path.basename(path),
                "start": int(start),
                "lpips": float(np.mean([row["lpips"] for row in averaged_horizons])),
                "lpips_last": float(averaged_horizons[-1]["lpips"]),
                "mse": float(np.mean([row["mse"] for row in averaged_horizons])),
                "psnr": float(np.mean([row["psnr"] for row in averaged_horizons])),
                "ssim": float(np.mean([row["ssim"] for row in averaged_horizons])),
                "horizons": averaged_horizons,
            }
        )
    return rows


def _parse_arms(values, seeds):
    arms = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--arm must be NAME=PATH_PATTERN, got {value!r}")
        name, pattern = value.split("=", 1)
        if "{seed}" not in pattern:
            raise ValueError(f"checkpoint pattern must contain {{seed}}: {pattern}")
        arms.extend((name, seed, pattern.format(seed=seed)) for seed in seeds)
    return arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="repeatable NAME=CHECKPOINT_PATTERN; pattern must contain {seed}",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--include_baselines", action="store_true")
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--eval_draws", type=int, default=1)
    parser.add_argument("--train_windows", type=int, default=24)
    parser.add_argument("--split_seed", type=int, default=1)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=999)
    parser.add_argument("--max_windows", type=int, default=0, help="smoke only")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_dir", default=f"{ROOT}/outputs/analysis/msp_episode_disjoint_eval"
    )
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    jobs = _parse_arms(args.arm, seeds)
    if args.include_baselines:
        jobs = [("base", None, MSP_BASE_DIR), ("rlvr", None, MSP_RLVR_DIR), *jobs]
    if not jobs:
        raise ValueError("provide at least one --arm or --include_baselines")
    if args.deterministic:
        set_determinism(args.eval_seed)

    train, windows = build_episode_disjoint_windows(
        list_episodes(), args.T, args.train_windows, args.split_seed, args.stride
    )
    if args.max_windows:
        windows = windows[: args.max_windows]
    if not windows:
        raise RuntimeError("episode-disjoint evaluation set is empty")
    manifest = _manifest(train, windows)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"[manifest] train={len(train)} windows/{len(set(p for p, _ in train))} episodes "
        f"eval={len(windows)} windows/{len(set(p for p, _ in windows))} episodes "
        f"sha256={manifest['sha256']}",
        flush=True,
    )

    tokenizer = _load_tokenizer(args.device)
    action_ranges = load_action_ranges(args.device)
    metric_fn = Metrics(args.device)
    sweep_started = time.time()
    completed_jobs = 0
    for name, seed, checkpoint in jobs:
        label = name if seed is None else f"{name}/s{seed}"
        filename = f"{name}_eval.json" if seed is None else f"{name}_s{seed}.json"
        output = os.path.join(args.out_dir, filename)
        if os.path.exists(output):
            print(f"[skip] {label}: {output} exists", flush=True)
            completed_jobs += 1
            continue
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(checkpoint)
        print(
            f"\n===== EVAL {_bar(completed_jobs / len(jobs))} "
            f"{completed_jobs}/{len(jobs)} next={label} =====",
            flush=True,
        )
        model = _load_model(checkpoint, args.device)
        rows = evaluate_model(
            model, tokenizer, action_ranges, metric_fn, windows, args, label
        )
        report = {
            "label": label,
            "arm": name,
            "seed": seed,
            "checkpoint": checkpoint,
            "protocol": {
                "T": args.T,
                "K": args.K,
                "eval_draws": args.eval_draws,
                "train_windows": args.train_windows,
                "split_seed": args.split_seed,
                "stride": args.stride,
                "eval_seed": args.eval_seed,
                "manifest_sha256": manifest["sha256"],
                "selection": "all windows from episodes absent from training",
            },
            "aggregate": aggregate_rows(rows),
            "rows": rows,
        }
        with open(output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"[saved] {output}", flush=True)
        del model
        torch.cuda.empty_cache()
        completed_jobs += 1
        elapsed = time.time() - sweep_started
        eta = elapsed / completed_jobs * (len(jobs) - completed_jobs)
        print(
            f"[sweep] {completed_jobs}/{len(jobs)} elapsed={_hms(elapsed)} "
            f"eta={_hms(eta)}",
            flush=True,
        )
    print("MSP_EPISODE_DISJOINT_EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
