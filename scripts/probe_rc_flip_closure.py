"""Close the C1 diagnostic chain: does reconstruction calibration REDUCE rank
corruption on the very same candidate groups?

For each held-out window we generate K candidates once, then score them twice:
  raw verifier   -d(cand, s)         d in {LPIPS, MSE, MSE+LPIPS}
  RC verifier    -d(cand, D(E(s)))   same d, reachable target
Reference ranking = -code_rms in the pre-decode FSQ code space (floor-free).
Per window we report the within-group Pearson correlation rho and the pairwise
flip rate vs the reference, for raw and RC; the paper's claim closes if
rho_RC > rho_raw and flip_RC < flip_raw with arccos(rho)/pi still matching.

Zero training. Mirrors the candidate recipe of cache_reward_spaces.py
(same sampler, temperature, top_k, per-window seed).

Example (smoke):
  python scripts/probe_rc_flip_closure.py --n_windows 4 --seeds 0
Formal:
  python scripts/probe_rc_flip_closure.py --n_windows 184 --seeds 0,1,2,3,4
"""
import argparse
import math
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch

from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import code_rms
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens, encode_indices


def _hms(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    return f"{h:d}:{r // 60:02d}:{r % 60:02d}"


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-12
    return float((a * b).sum()) / den


def flip_rate(a, b):
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    m = np.triu(np.ones_like(da, bool), 1)
    return float((da[m] != db[m]).mean())


SPACES = ("lpips", "mse", "faithful")  # faithful = MSE + LPIPS (paper's verifier)


def scores_from(q):
    return {"lpips": -q["lpips"], "mse": -q["mse"], "faithful": -(q["mse"] + q["lpips"])}


def run_seed(seed, args, dev="cuda"):
    tok = load_tokenizer(dev)
    model = load_world_model(dev, "base")
    ar = load_action_ranges(dev)
    M = Metrics(dev)
    wins = sample_windows(list_episodes(), args.n_windows, seed=seed)
    per = {f"rho_{s}_{v}": [] for s in SPACES for v in ("raw", "rc")}
    per.update({f"flip_{s}_{v}": [] for s in SPACES for v in ("raw", "rc")})
    floors = []
    t0 = time.time()
    for wi, (p, s) in enumerate(wins):
        frames, actions = get_window_tensors(p, s, dev)
        gt = frames[CTX]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, args.K, temperature=args.temperature,
                                   top_k=args.top_k, seed=seed * 100000 + wi)
        imgs = decode_tokens(tok, cand)
        gt_idx = encode_indices(tok, gt.unsqueeze(0))
        s_tilde = decode_tokens(tok, gt_idx.reshape(1, -1))[0]          # reachable target
        q_raw = M.eval_batch(imgs, gt)
        q_rc = M.eval_batch(imgs, s_tilde)
        ref = -code_rms(tok, cand, gt_idx)                              # floor-free reference
        floors.append(float(M.eval_batch(s_tilde.unsqueeze(0), gt)["lpips"][0]))
        raw_s, rc_s = scores_from(q_raw), scores_from(q_rc)
        for sp in SPACES:
            for v, sc in (("raw", raw_s[sp]), ("rc", rc_s[sp])):
                per[f"rho_{sp}_{v}"].append(pearson(np.asarray(sc), np.asarray(ref)))
                per[f"flip_{sp}_{v}"].append(flip_rate(np.asarray(sc), np.asarray(ref)))
        if (wi + 1) % 10 == 0 or wi + 1 == len(wins):
            el = time.time() - t0
            print(f"[s{seed} {wi + 1}/{len(wins)}] elapsed={_hms(el)} "
                  f"eta={_hms(el / (wi + 1) * (len(wins) - wi - 1))}", flush=True)
    out = {k: np.array(v, np.float32) for k, v in per.items()}
    out["floor_lpips"] = np.array(floors, np.float32)
    del model, tok
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=184)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/rc_flip_closure.npz")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    allres = None
    for sd in seeds:
        r = run_seed(sd, args)
        allres = r if allres is None else {k: np.concatenate([allres[k], r[k]]) for k in r}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **allres)

    n = len(allres["floor_lpips"])
    print(f"\n=== RC flip-closure over {n} windows x K={args.K} "
          f"(seeds {seeds}) ===")
    print(f"{'space':10s} {'rho_raw':>8s} {'rho_rc':>8s} | {'flip_raw':>8s} "
          f"{'flip_rc':>8s} | {'th_raw':>7s} {'th_rc':>7s}")
    for sp in SPACES:
        rr, rc = allres[f"rho_{sp}_raw"].mean(), allres[f"rho_{sp}_rc"].mean()
        fr, fc = allres[f"flip_{sp}_raw"].mean(), allres[f"flip_{sp}_rc"].mean()
        tr = float(np.arccos(np.clip(allres[f"rho_{sp}_raw"], -1, 1)).mean() / math.pi)
        tc = float(np.arccos(np.clip(allres[f"rho_{sp}_rc"], -1, 1)).mean() / math.pi)
        verdict = "CLOSED" if (rc > rr and fc < fr) else "NOT-CLOSED"
        print(f"{sp:10s} {rr:8.3f} {rc:8.3f} | {fr:8.3f} {fc:8.3f} | "
              f"{tr:7.3f} {tc:7.3f}  {verdict}")
    print(f"mean floor LPIPS = {allres['floor_lpips'].mean():.4f}")
    print("RC_FLIP_CLOSURE_DONE")


if __name__ == "__main__":
    main()
