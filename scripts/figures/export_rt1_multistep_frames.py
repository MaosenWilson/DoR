#!/usr/bin/env python3
"""Export auditable RT-1 multi-step qualitative trajectories.

The workflow has two mandatory stages:

1. ``shortlist`` selects scenes from ground-truth-only episode-disjoint windows.
   No model or reward is loaded.
2. ``generate`` reads the frozen scene manifest and samples candidate index zero
   with K=1 and identical generation seeds for every checkpoint.

This separation prevents scene-level and candidate-level cherry-picking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dor.constants import ROOT
from dor.episodes import list_episodes, load_episode
from dor.grpo import _bar, _hms, set_determinism
from dor.metrics import Metrics
from dor.models import load_action_ranges
from dor.multistep import (
    MSP_BASE_DIR,
    MSP_RLVR_DIR,
    V_MSP,
    detok_chunked,
    discretize_actions,
    msp_rollout,
    msp_sample_windows,
    msp_window,
)
from dor.qualitative import SceneCandidate, scene_feature, select_diverse_scenes, temporal_motion


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    if not cleaned:
        raise ValueError(f"label {value!r} has no filename-safe characters")
    return cleaned


def _save_rgb(array: np.ndarray, path: Path) -> None:
    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = np.rint(image * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _episode_disjoint_paths(T: int, train_windows: int, split_seed: int, stride: int):
    paths = list_episodes()
    train = msp_sample_windows(paths, train_windows, T, seed=split_seed, stride=stride)
    train_episodes = {path for path, _ in train}
    evaluation = [path for path in paths if path not in train_episodes]
    if not evaluation:
        raise RuntimeError("episode-disjoint evaluation set is empty")
    return train, evaluation


def _contact_sheet(rows: list[dict], output: Path, T: int) -> None:
    horizons = sorted(set([0, min(2, T - 1), min(4, T - 1), T - 1]))
    thumb_w, thumb_h, label_w, pad = 240, 192, 210, 8
    canvas = Image.new(
        "RGB",
        (label_w + len(horizons) * (thumb_w + pad), len(rows) * (thumb_h + 30) + 34),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, horizon in enumerate(horizons):
        draw.text((label_w + column * (thumb_w + pad), 8), f"GT t={horizon}", fill="black")
    for row_index, row in enumerate(rows):
        images, _ = load_episode(row["path"])
        y = 34 + row_index * (thumb_h + 30)
        label = f"{row_index:02d} {row['episode']}\nstart={row['start']} motion={row['motion']:.4f}"
        draw.multiline_text((4, y + 4), label, fill="black", spacing=3)
        for column, horizon in enumerate(horizons):
            image = Image.fromarray(images[row["start"] + horizon]).resize(
                (thumb_w, thumb_h), Image.Resampling.LANCZOS
            )
            canvas.paste(image, (label_w + column * (thumb_w + pad), y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def shortlist(args) -> None:
    train, evaluation_episodes = _episode_disjoint_paths(
        args.T, args.train_windows, args.split_seed, args.stride
    )
    candidates: list[SceneCandidate] = []
    rows = []
    for path in evaluation_episodes:
        images, _ = load_episode(path)
        starts = list(range(0, len(images) - args.T, args.stride))
        if not starts:
            continue
        scores = [temporal_motion(images[start : start + args.T]) for start in starts]
        best_offset = int(np.argmax(scores))
        start = starts[best_offset]
        feature = scene_feature(images[start : start + args.T])
        candidate = SceneCandidate(
            episode=os.path.basename(path),
            start=int(start),
            motion=float(scores[best_offset]),
            feature=feature,
        )
        candidates.append(candidate)
        rows.append(
            {
                "episode": candidate.episode,
                "path": path,
                "start": candidate.start,
                "motion": candidate.motion,
            }
        )
    rows.sort(key=lambda row: row["episode"])
    if args.scene_indices:
        scene_indices = [int(item) for item in args.scene_indices.split(",") if item.strip()]
        if len(scene_indices) != len(set(scene_indices)):
            raise ValueError("--scene_indices must not contain duplicates")
        if any(index < 0 or index >= len(rows) for index in scene_indices):
            raise IndexError(f"scene index outside [0,{len(rows)})")
        selected = [
            next(
                item
                for item in candidates
                if item.episode == rows[index]["episode"] and item.start == rows[index]["start"]
            )
            for index in scene_indices
        ]
        selection_rule = (
            "ground-truth-only manual semantic coverage using frozen contact-sheet row indices "
            f"{scene_indices}; no model prediction or reward is loaded"
        )
    else:
        selected = select_diverse_scenes(
            candidates, args.scene_count, motion_weight=args.motion_weight
        )
        selection_rule = (
            "one maximum-GT-motion window per unseen episode, followed by deterministic "
            "farthest-first GT appearance diversity; no model prediction or reward is loaded"
    )
    selected_keys = {(item.episode, item.start) for item in selected}
    for row in rows:
        row["selected"] = (row["episode"], row["start"]) in selected_keys
    selected_rows = []
    for order, item in enumerate(selected):
        source = next(row for row in rows if row["episode"] == item.episode and row["start"] == item.start)
        selected_rows.append(
            {
                "scene": f"Scene {chr(ord('A') + order)}",
                "episode": item.episode,
                "path": source["path"],
                "start": item.start,
                "motion": item.motion,
            }
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    contact = output / "gt_only_shortlist.png"
    _contact_sheet(rows, contact, args.T)
    manifest = {
        "protocol": "RT-1 qualitative scene selection v1",
        "selection_rule": selection_rule,
        "T": args.T,
        "train_windows": args.train_windows,
        "split_seed": args.split_seed,
        "stride": args.stride,
        "motion_weight": args.motion_weight,
        "train_episodes": sorted(os.path.basename(path) for path, _ in train),
        "candidate_rows": rows,
        "selected": selected_rows,
        "contact_sheet": str(contact.resolve()),
    }
    manifest_path = output / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[shortlist] eval episodes={len(rows)} selected={len(selected_rows)}")
    for row in selected_rows:
        print(
            f"  {row['scene']}: {row['episode']} start={row['start']} "
            f"motion={row['motion']:.5f}"
        )
    print(f"saved {manifest_path}\nsaved {contact}\nRT1_QUAL_SHORTLIST_OK", flush=True)


def _parse_checkpoints(values: list[str], selections: list[str]) -> list[dict]:
    parsed = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in labels:
            raise ValueError(f"duplicate or empty checkpoint label {label!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        labels.add(label)
        parsed.append(
            {
                "label": label,
                "path": path,
                "selection_note": "fixed public checkpoint or checkpoint fixed before scene generation",
                "selection_manifest": None,
            }
        )
    for value in selections:
        if "=" not in value:
            raise ValueError(f"--checkpoint_selection must be LABEL=JSON, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in labels:
            raise ValueError(f"duplicate or empty checkpoint label {label!r}")
        selection_path = Path(raw_path).expanduser().resolve()
        selection = json.loads(selection_path.read_text())
        path = Path(selection["checkpoint"]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        labels.add(label)
        parsed.append(
            {
                "label": label,
                "path": path,
                "selection_note": selection["rule"],
                "selection_manifest": str(selection_path),
            }
        )
    if not parsed:
        raise ValueError("provide at least one --checkpoint LABEL=PATH")
    return parsed


def _load_model(path: Path, device: str):
    import dor.compat  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
    except AttributeError as error:
        weight_path = path / "model.safetensors"
        metadata_error = "'NoneType' object has no attribute 'get'" in str(error)
        if not metadata_error or not weight_path.is_file():
            raise
        from safetensors.torch import load_model

        config = AutoConfig.from_pretrained(path)
        model = AutoModelForCausalLM.from_config(config)
        load_model(model, str(weight_path), strict=True, device="cpu")
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model = model.to(device).eval()
    model.config.use_cache = True
    return model


def _load_tokenizer(device: str):
    import dor.compat  # noqa: F401
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ

    from dor.multistep import MSP_TOK_DIR

    return CompressiveVQModelFSQ.from_pretrained(MSP_TOK_DIR).to(device).eval()


@torch.no_grad()
def generate(args) -> None:
    manifest_path = Path(args.scene_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    scenes = manifest.get("selected", [])
    if not scenes:
        raise ValueError("scene manifest has no selected scenes")
    checkpoints = _parse_checkpoints(args.checkpoint, args.checkpoint_selection)
    if args.deterministic:
        set_determinism(args.generation_seed)

    device = args.device
    tokenizer = _load_tokenizer(device)
    action_ranges = load_action_ranges(device)
    metrics = Metrics(device)
    output = Path(args.out_dir)
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    gt_uint8 = []
    loaded_scenes = []
    for scene in scenes:
        frames, actions = msp_window(scene["path"], int(scene["start"]), args.T, device)
        gt_uint8.append(
            np.rint(frames[1:].permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        )
        loaded_scenes.append((frames, actions))
        for horizon, image in enumerate(gt_uint8[-1], start=1):
            _save_rgb(image, frame_dir / _safe_name(scene["scene"]) / "Ground_truth" / f"t{horizon}.png")

    arrays: dict[str, np.ndarray] = {"ground_truth": np.stack(gt_uint8)}
    method_reports = []
    total = len(checkpoints) * len(scenes)
    done = 0
    started = time.monotonic()
    for checkpoint_spec in checkpoints:
        label = checkpoint_spec["label"]
        checkpoint = checkpoint_spec["path"]
        print(f"\n[load] {label}: {checkpoint}", flush=True)
        model = _load_model(checkpoint, device)
        predictions = []
        scene_metrics = []
        for scene_index, (scene, (frames, actions)) in enumerate(zip(scenes, loaded_scenes)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                context_tokens, _ = tokenizer.tokenize(frames.unsqueeze(0))
            context = (context_tokens.reshape(1, -1) + V_MSP).long()
            action_tokens = discretize_actions(actions, action_ranges)[1 : args.T] + 2 * V_MSP
            seed = args.generation_seed + scene_index
            dynamics = msp_rollout(
                model,
                context,
                action_tokens,
                args.T - 1,
                1,
                seed=seed,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            decoded = detok_chunked(tokenizer, context_tokens, dynamics)[0]
            pred_uint8 = np.rint(
                decoded.permute(0, 2, 3, 1).cpu().numpy() * 255.0
            ).astype(np.uint8)
            predictions.append(pred_uint8)
            horizons = []
            for horizon_index in range(args.T - 1):
                measured = metrics.eval_batch(decoded[horizon_index : horizon_index + 1], frames[horizon_index + 1])
                horizons.append(
                    {
                        "horizon": horizon_index + 1,
                        "lpips": float(np.asarray(measured["lpips"]).mean()),
                        "mse": float(np.asarray(measured["mse"]).mean()),
                        "psnr": float(np.asarray(measured["psnr"]).mean()),
                        "ssim": float(np.asarray(measured["ssim"]).mean()),
                    }
                )
                _save_rgb(
                    pred_uint8[horizon_index],
                    frame_dir / _safe_name(scene["scene"]) / _safe_name(label) / f"t{horizon_index + 1}.png",
                )
            scene_metrics.append(
                {
                    "scene": scene["scene"],
                    "episode": scene["episode"],
                    "start": scene["start"],
                    "generation_seed": seed,
                    "horizons": horizons,
                }
            )
            done += 1
            elapsed = time.monotonic() - started
            eta = elapsed / done * (total - done)
            print(
                f"[{label}] {_bar(done / total)} {done}/{total} "
                f"scene={scene['scene']} elapsed={_hms(elapsed)} eta={_hms(eta)}",
                flush=True,
            )
        slug = _safe_name(label)
        arrays[f"prediction__{slug}"] = np.stack(predictions)
        method_reports.append(
            {
                "label": label,
                "array_key": f"prediction__{slug}",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": None if args.skip_checkpoint_hash else _tree_hash(checkpoint),
                "selection_note": checkpoint_spec["selection_note"],
                "selection_manifest": checkpoint_spec["selection_manifest"],
                "scenes": scene_metrics,
            }
        )
        del model
        torch.cuda.empty_cache()

    output.mkdir(parents=True, exist_ok=True)
    archive = output / "trajectories.npz"
    np.savez_compressed(archive, **arrays)
    report = {
        "protocol": "RT-1 fixed-candidate qualitative export v1",
        "scene_manifest": str(manifest_path),
        "scene_selection_rule": manifest["selection_rule"],
        "candidate_selection": "K=1, candidate index 0; no best-of-K or per-scene checkpoint selection",
        "T": args.T,
        "horizons_exported": list(range(1, args.T)),
        "generation_seed": args.generation_seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "deterministic_requested": bool(args.deterministic),
        "scenes": scenes,
        "methods": method_reports,
        "archive": str(archive.resolve()),
    }
    report_path = output / "generation_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {archive}\nsaved {report_path}\nRT1_QUAL_GENERATE_OK", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    shortlist_parser = subparsers.add_parser("shortlist")
    shortlist_parser.add_argument("--T", type=int, default=8)
    shortlist_parser.add_argument("--train_windows", type=int, default=24)
    shortlist_parser.add_argument("--split_seed", type=int, default=1)
    shortlist_parser.add_argument("--stride", type=int, default=8)
    shortlist_parser.add_argument("--scene_count", type=int, default=4)
    shortlist_parser.add_argument(
        "--scene_indices",
        default="",
        help=(
            "optional comma-separated GT contact-sheet row indices in display order; "
            "use only to freeze semantic scene coverage before generating predictions"
        ),
    )
    shortlist_parser.add_argument("--motion_weight", type=float, default=0.10)
    shortlist_parser.add_argument(
        "--out_dir", default=f"{ROOT}/outputs/figures/rt1_multistep_qualitative"
    )

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--scene_manifest", required=True)
    generate_parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "repeat LABEL=PATH; recommended rows are "
            f"Base={MSP_BASE_DIR}, RLVR-World={MSP_RLVR_DIR}, and a preselected median-seed ours checkpoint"
        ),
    )
    generate_parser.add_argument(
        "--checkpoint_selection",
        action="append",
        default=[],
        help="repeat LABEL=median-checkpoint-selection.json for trained methods",
    )
    generate_parser.add_argument("--T", type=int, default=8)
    generate_parser.add_argument("--generation_seed", type=int, default=92027)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top_k", type=int, default=100)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--deterministic", action="store_true")
    generate_parser.add_argument("--skip_checkpoint_hash", action="store_true")
    generate_parser.add_argument(
        "--out_dir", default=f"{ROOT}/outputs/figures/rt1_multistep_qualitative"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "shortlist":
        shortlist(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
