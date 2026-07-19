#!/usr/bin/env python3
"""Export matched RT-1 reconstruction-floor examples for two project tokenizers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dor.grpo import _bar, _hms
from dor.metrics import Metrics
from dor.models import load_tokenizer
from dor.multistep import detok_chunked, msp_window
from dor.tokenization import decode_tokens, encode_indices


def _uint8(frames: torch.Tensor) -> np.ndarray:
    return np.rint(frames.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).astype(np.uint8)


def _write(path: Path, gt: list[np.ndarray], reachable: list[np.ndarray], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gt=np.stack(gt),
        reachable=np.stack(reachable),
        scene=np.asarray([row["scene"] for row in rows]),
        episode=np.asarray([row["episode"] for row in rows]),
        start=np.asarray([row["start"] for row in rows], dtype=np.int64),
        horizon=np.asarray([row["horizon"] for row in rows], dtype=np.int64),
        lpips=np.asarray([row["lpips"] for row in rows], dtype=np.float32),
        mse=np.asarray([row["mse"] for row in rows], dtype=np.float32),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_manifest", required=True)
    parser.add_argument("--horizons", default="2,4,6,7")
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    if any(horizon < 1 or horizon >= args.T for horizon in horizons):
        raise ValueError(f"horizons must lie in [1,{args.T - 1}]")
    manifest_path = Path(args.scene_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    scenes = manifest["selected"]
    output = Path(args.out_dir)
    metrics = Metrics(args.device)

    print("[load] CNN-FSQ tokenizer", flush=True)
    cnn = load_tokenizer(args.device)
    gt_cnn, reachable_cnn, rows_cnn = [], [], []
    started = time.monotonic()
    total = len(scenes) * len(horizons) * 2
    done = 0
    for scene in scenes:
        frames, _ = msp_window(scene["path"], int(scene["start"]), args.T, args.device)
        chosen = frames[horizons]
        indices = encode_indices(cnn, chosen).reshape(len(horizons), -1)
        reconstruction = decode_tokens(cnn, indices)
        for offset, horizon in enumerate(horizons):
            measured = metrics.eval_batch(reconstruction[offset : offset + 1], chosen[offset])
            row = {
                "scene": scene["scene"],
                "episode": scene["episode"],
                "start": scene["start"],
                "horizon": horizon,
                "lpips": float(np.asarray(measured["lpips"]).mean()),
                "mse": float(np.asarray(measured["mse"]).mean()),
            }
            rows_cnn.append(row)
            gt_cnn.append(_uint8(chosen[offset : offset + 1])[0])
            reachable_cnn.append(_uint8(reconstruction[offset : offset + 1])[0])
            done += 1
            elapsed = time.monotonic() - started
            print(
                f"[CNN-FSQ] {_bar(done / total)} {done}/{total} "
                f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (total - done))}",
                flush=True,
            )
    del cnn
    torch.cuda.empty_cache()

    print("[load] compressive FSQ tokenizer", flush=True)
    import dor.compat  # noqa: F401
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
    from dor.multistep import MSP_TOK_DIR

    compressive = CompressiveVQModelFSQ.from_pretrained(MSP_TOK_DIR).to(args.device).eval()
    gt_compressive, reachable_compressive, rows_compressive = [], [], []
    for scene in scenes:
        frames, _ = msp_window(scene["path"], int(scene["start"]), args.T, args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context_tokens, dynamics_tokens = compressive.tokenize(frames.unsqueeze(0))
        reconstruction = detok_chunked(compressive, context_tokens, dynamics_tokens)[0]
        for horizon in horizons:
            target = frames[horizon]
            reachable = reconstruction[horizon - 1]
            measured = metrics.eval_batch(reachable.unsqueeze(0), target)
            row = {
                "scene": scene["scene"],
                "episode": scene["episode"],
                "start": scene["start"],
                "horizon": horizon,
                "lpips": float(np.asarray(measured["lpips"]).mean()),
                "mse": float(np.asarray(measured["mse"]).mean()),
            }
            rows_compressive.append(row)
            gt_compressive.append(_uint8(target.unsqueeze(0))[0])
            reachable_compressive.append(_uint8(reachable.unsqueeze(0))[0])
            done += 1
            elapsed = time.monotonic() - started
            print(
                f"[Compressive-FSQ] {_bar(done / total)} {done}/{total} "
                f"elapsed={_hms(elapsed)} eta={_hms(elapsed / done * (total - done))}",
                flush=True,
            )

    cnn_path = output / "cnn_fsq_floor_examples.npz"
    compressive_path = output / "compressive_fsq_floor_examples.npz"
    _write(cnn_path, gt_cnn, reachable_cnn, rows_cnn)
    _write(compressive_path, gt_compressive, reachable_compressive, rows_compressive)
    report = {
        "protocol": "matched RT-1 reconstruction-floor examples v1",
        "scene_manifest": str(manifest_path),
        "horizons": horizons,
        "CNN-FSQ": {
            "cache": str(cnn_path.resolve()),
            "mean_lpips": float(np.mean([row["lpips"] for row in rows_cnn])),
            "mean_mse": float(np.mean([row["mse"] for row in rows_cnn])),
        },
        "Compressive-FSQ": {
            "cache": str(compressive_path.resolve()),
            "mean_lpips": float(np.mean([row["lpips"] for row in rows_compressive])),
            "mean_mse": float(np.mean([row["mse"] for row in rows_compressive])),
        },
    }
    report_path = output / "rt1_floor_examples.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"saved {cnn_path}\nsaved {compressive_path}\nsaved {report_path}\n"
        "RT1_FLOOR_EXAMPLES_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
