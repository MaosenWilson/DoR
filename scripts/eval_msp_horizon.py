"""Per-horizon evaluation for multi-step MSP checkpoints.

Examples:
  python scripts/eval_msp_horizon.py \
    --arms base=BASE,rlvr=RLVR,return=outputs/msp_step30_return_hkl00/ckpt/rc_msp_s* \
    --K 16 --T 8 --eval_windows 8 --out outputs/analysis/msp_horizon.json
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from dor.constants import ROOT
from dor.episodes import list_episodes
from dor.grpo import _bar, _hms
from dor.metrics import Metrics
from dor.models import load_action_ranges
from dor.multistep import (MSP_BASE_DIR, MSP_RLVR_DIR, MSP_TOK_DIR, V_MSP,
                           detok_chunked, discretize_actions, msp_rollout,
                           msp_sample_windows, msp_window)


def load_tokenizer(dev):
    import dor.compat  # noqa: F401
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
    return CompressiveVQModelFSQ.from_pretrained(MSP_TOK_DIR).to(dev).eval()


def load_model(path_or_alias, dev):
    import dor.compat  # noqa: F401
    from transformers import AutoModelForCausalLM
    key = path_or_alias.upper()
    if key == "BASE":
        path = MSP_BASE_DIR
    elif key == "RLVR":
        path = MSP_RLVR_DIR
    else:
        path = path_or_alias
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(dev).eval()


def expand_arms(spec):
    arms = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad arm spec {item!r}; expected label=path")
        label, path = item.split("=", 1)
        label, path = label.strip(), path.strip()
        if path.upper() in ("BASE", "RLVR"):
            arms.append((label, path.upper(), "baseline"))
            continue
        matches = sorted(glob.glob(path))
        if not matches:
            raise FileNotFoundError(f"arm {label}: no matches for {path}")
        for m in matches:
            seed = os.path.basename(m).split("_s")[-1] if "_s" in os.path.basename(m) else os.path.basename(m)
            arms.append((label, m, seed))
    return arms


@torch.no_grad()
def eval_per_horizon(model, tok, ar, metrics, wins, T, K, dev, seed=999):
    F = T - 1
    buckets = {
        "lpips": [[] for _ in range(F)],
        "mse": [[] for _ in range(F)],
        "mae": [[] for _ in range(F)],
        "psnr": [[] for _ in range(F)],
        "ssim": [[] for _ in range(F)],
        "dmotion": [[] for _ in range(F)],
    }
    for wi, (p, s) in enumerate(wins):
        frames, actions = msp_window(p, s, T, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            idx_c, _ = tok.tokenize(frames.unsqueeze(0))
        ctx_off = (idx_c.reshape(1, -1) + V_MSP).long()
        act_off = discretize_actions(actions, ar)[1:T] + 2 * V_MSP
        dyn = msp_rollout(model, ctx_off, act_off, F, K, seed=seed + wi)
        pred = detok_chunked(tok, idx_c.expand(K, -1, -1), dyn)
        real_f = frames[1:]
        for h in range(1, F):
            q = metrics.eval_batch(pred[:, h], real_f[h])
            for key in ("lpips", "mse", "mae", "psnr"):
                buckets[key][h].append(float(np.mean(q[key])))
            if "ssim" in q:
                buckets["ssim"][h].append(float(np.mean(q["ssim"])))
            pred_delta = (pred[:, h] - pred[:, h - 1]).flatten(1)
            gt_delta = (real_f[h] - real_f[h - 1]).flatten().unsqueeze(0).expand_as(pred_delta)
            cos = torch.nn.functional.cosine_similarity(pred_delta, gt_delta, dim=1)
            buckets["dmotion"][h].append(float(cos.mean().detach().cpu()))
    horizons = []
    for h in range(1, F):
        row = {
            "horizon": h + 1,
            "n_windows": len(buckets["lpips"][h]),
        }
        for key, vals in buckets.items():
            row[key] = float(np.mean(vals[h])) if vals[h] else float("nan")
        horizons.append(row)
    return horizons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma list label=BASE|RLVR|ckpt_glob")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--train_windows", type=int, default=24)
    ap.add_argument("--eval_windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1, help="held-out window sampling seed")
    ap.add_argument("--sample_seed", type=int, default=999, help="rollout sampling seed")
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/msp_horizon.json")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    arms = expand_arms(args.arms)
    tok = load_tokenizer(dev)
    ar = load_action_ranges(dev)
    metrics = Metrics(dev)
    allw = msp_sample_windows(list_episodes(), args.train_windows + args.eval_windows,
                              args.T, seed=args.seed)
    wins = allw[args.train_windows:]

    rows = []
    t0 = time.time()
    for i, (label, path, seed_label) in enumerate(arms, 1):
        print(f"\n===== HORIZON {_bar((i - 1) / len(arms))} {i - 1}/{len(arms)} "
              f"next: {label} {seed_label} =====", flush=True)
        model = load_model(path, dev)
        horizons = eval_per_horizon(model, tok, ar, metrics, wins, args.T, args.K, dev,
                                    seed=args.sample_seed)
        for h in horizons:
            row = {"label": label, "seed": seed_label, "path": path, **h}
            rows.append(row)
        del model
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[horizon] {_bar(i / len(arms))} {i}/{len(arms)} elapsed={_hms(el)} "
              f"eta={_hms(el / i * (len(arms) - i))}", flush=True)

    payload = {"args": vars(args), "rows": rows}
    json.dump(payload, open(args.out, "w"), indent=2)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "seed", "horizon", "lpips", "mse",
                                          "mae", "psnr", "ssim", "dmotion",
                                          "n_windows", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {args.out}\nsaved {csv_path}\nMSP_HORIZON_OK", flush=True)


if __name__ == "__main__":
    main()
