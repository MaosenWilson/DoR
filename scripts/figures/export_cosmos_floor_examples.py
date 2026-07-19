#!/usr/bin/env python3
"""Export Cosmos DV-FSQ reconstructions for the frozen RT-1 scene manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dor.grpo import _bar, _hms
from dor.metrics import Metrics


def _uint8(frames: torch.Tensor) -> np.ndarray:
    return np.rint(frames.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos_dir", required=True, help="directory with encoder.jit and decoder.jit")
    parser.add_argument("--scene_manifest", required=True)
    parser.add_argument("--horizons", default="2,4,6,7")
    parser.add_argument("--clip_len", type=int, default=9, help="Cosmos expects 8k+1 temporal clips")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    if any(horizon < 0 or horizon >= args.clip_len for horizon in horizons):
        raise ValueError(f"horizons must lie in [0,{args.clip_len - 1}]")
    manifest_path = Path(args.scene_manifest).expanduser().resolve()
    scenes = json.loads(manifest_path.read_text())["selected"]
    device = torch.device(args.device)
    encoder = torch.jit.load(str(Path(args.cosmos_dir) / "encoder.jit")).to(device).eval()
    decoder = torch.jit.load(str(Path(args.cosmos_dir) / "decoder.jit")).to(device).eval()
    metrics = Metrics(device)

    gt_rows, reconstruction_rows, metadata = [], [], []
    started = time.monotonic()
    for scene_index, scene in enumerate(scenes):
        images = np.asarray(np.load(scene["path"], allow_pickle=True)["image"])
        start = int(scene["start"])
        clip = images[start : start + args.clip_len]
        if len(clip) != args.clip_len:
            raise ValueError(
                f"{scene['episode']} start={start} has only {len(clip)} frames; "
                f"Cosmos requires {args.clip_len}"
            )
        frames = torch.from_numpy(clip).permute(0, 3, 1, 2).float().div_(255.0)
        frames = F.interpolate(
            frames,
            size=(args.resolution, args.resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        model_input = (frames.permute(1, 0, 2, 3) * 2.0 - 1.0).unsqueeze(0).to(device)
        encoded = encoder(model_input)
        decoded = decoder(encoded[0] if isinstance(encoded, (tuple, list)) else encoded)
        decoded = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
        if decoded.ndim != 5 or decoded.shape[0] != 1:
            raise ValueError(f"Cosmos decoder must return [1,3,T,H,W], got {tuple(decoded.shape)}")
        reconstruction = (decoded[0].permute(1, 0, 2, 3).float().clamp(-1, 1) + 1.0) / 2.0
        if reconstruction.shape[0] != args.clip_len:
            raise ValueError(
                f"Cosmos returned {reconstruction.shape[0]} frames for clip_len={args.clip_len}"
            )
        for horizon in horizons:
            target = frames[horizon].to(device)
            reachable = reconstruction[horizon]
            measured = metrics.eval_batch(reachable.unsqueeze(0), target)
            gt_rows.append(_uint8(target.unsqueeze(0).cpu())[0])
            reconstruction_rows.append(_uint8(reachable.unsqueeze(0).cpu())[0])
            metadata.append(
                {
                    "scene": scene["scene"],
                    "episode": scene["episode"],
                    "start": start,
                    "horizon": horizon,
                    "lpips": float(np.asarray(measured["lpips"]).mean()),
                    "mse": float(np.asarray(measured["mse"]).mean()),
                }
            )
        done = scene_index + 1
        elapsed = time.monotonic() - started
        print(
            f"[Cosmos DV-FSQ] {_bar(done / len(scenes))} {done}/{len(scenes)} "
            f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (len(scenes) - done))}",
            flush=True,
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gt=np.stack(gt_rows),
        reachable=np.stack(reconstruction_rows),
        scene=np.asarray([row["scene"] for row in metadata]),
        episode=np.asarray([row["episode"] for row in metadata]),
        start=np.asarray([row["start"] for row in metadata], dtype=np.int64),
        horizon=np.asarray([row["horizon"] for row in metadata], dtype=np.int64),
        lpips=np.asarray([row["lpips"] for row in metadata], dtype=np.float32),
        mse=np.asarray([row["mse"] for row in metadata], dtype=np.float32),
    )
    report = {
        "protocol": "Cosmos DV-FSQ matched RT-1 floor examples v1",
        "scene_manifest": str(manifest_path),
        "clip_len": args.clip_len,
        "resolution": args.resolution,
        "horizons": horizons,
        "cache": str(output.resolve()),
        "mean_lpips": float(np.mean([row["lpips"] for row in metadata])),
        "mean_mse": float(np.mean([row["mse"] for row in metadata])),
        "note": "RT-1 frames are bilinearly resized to the square Cosmos input protocol",
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {output}\nsaved {report_path}\nCOSMOS_FLOOR_EXAMPLES_OK", flush=True)


if __name__ == "__main__":
    main()
