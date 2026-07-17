"""Audit whether cumulative policy drift explains post-training fidelity loss."""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from dor.constants import ROOT, TPF
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.grpo import _bar, _hms, set_determinism
from dor.models import load_action_ranges, load_tokenizer
from dor.tokenization import build_prompt


DEFAULT_TARGETS = ("rc_energy_point", "rc_energy_certified", "rc_energy")


def _load_checkpoint(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32
    ).to(device).eval()


def _token_logits(model, prompt, candidates):
    batch_prompt = prompt.unsqueeze(0).expand(len(candidates), -1)
    tokens = torch.cat([batch_prompt, candidates], dim=1)
    logits = model(input_ids=tokens[:, :-1]).logits
    start = prompt.shape[0] - 1
    return logits[:, start:start + TPF].float()


def _conditional_divergence(reference_logits, target_logits):
    reference_logp = reference_logits.log_softmax(dim=-1)
    target_logp = target_logits.log_softmax(dim=-1)
    reference_p = reference_logp.exp()
    target_p = target_logp.exp()
    kl_reference_target = (reference_p * (reference_logp - target_logp)).sum(dim=-1)
    kl_target_reference = (target_p * (target_logp - reference_logp)).sum(dim=-1)
    symmetric = 0.5 * (kl_reference_target + kl_target_reference)
    return {
        "kl_reference_target": float(kl_reference_target.mean().item()),
        "kl_target_reference": float(kl_target_reference.mean().item()),
        "symmetric_kl": float(symmetric.mean().item()),
    }


def _load_final_metrics(results_dir, arm, seed):
    path = os.path.join(results_dir, f"sweep_{arm}_gt_only_s{seed}.json")
    with open(path) as handle:
        run = next(iter(json.load(handle)["run"].values()))
    return {
        key: float(run[key][-1])
        for key in ("eval_lpips", "eval_mse", "eval_psnr", "eval_ssim", "eval_flow", "eval_dmotion")
    }


def _spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--window_seed", type=int, default=1)
    parser.add_argument("--generation_seed", type=int, default=74001)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default=f"{ROOT}/outputs/rc_energy/policy_drift_audit.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    targets = [value for value in args.targets.split(",") if value]
    if args.windows < 1 or args.K < 2 or not seeds or not targets:
        raise ValueError("policy drift audit received an empty or degenerate protocol")
    if args.deterministic:
        set_determinism(args.generation_seed)

    device = "cuda"
    tokenizer = load_tokenizer(device)
    action_ranges = load_action_ranges(device)
    windows = sample_windows(list_episodes(), args.windows, seed=args.window_seed)
    prompt_data = []
    for path, start in windows:
        frames, actions = get_window_tensors(path, start, device)
        prompt_data.append((os.path.basename(path), int(start), frames, actions))

    rows = []
    total = len(seeds) * len(targets)
    completed = 0
    started = time.time()
    print(
        f"[setup] seeds={seeds} targets={targets} windows={len(windows)} K={args.K}",
        flush=True,
    )
    for seed in seeds:
        reference_path = os.path.join(
            args.results_dir, "ckpt", f"a0faithful_tok_gt_only_s{seed}"
        )
        reference = _load_checkpoint(reference_path, device)
        fixed = []
        for window_index, (episode, start, frames, actions) in enumerate(prompt_data):
            prompt = build_prompt(tokenizer, frames, actions, action_ranges)
            candidates = generate_candidates(
                reference,
                prompt,
                args.K,
                seed=args.generation_seed + 1000 * seed + window_index,
            )
            reference_logits = _token_logits(reference, prompt, candidates).cpu()
            fixed.append((episode, start, prompt.cpu(), candidates.cpu(), reference_logits))
        reference_metrics = _load_final_metrics(args.results_dir, "a0faithful_tok", seed)

        for target_name in targets:
            target_path = os.path.join(
                args.results_dir, "ckpt", f"{target_name}_gt_only_s{seed}"
            )
            target = _load_checkpoint(target_path, device)
            divergences = []
            for episode, start, prompt_cpu, candidates_cpu, reference_logits_cpu in fixed:
                target_logits = _token_logits(
                    target, prompt_cpu.to(device), candidates_cpu.to(device)
                )
                divergences.append(
                    _conditional_divergence(reference_logits_cpu.to(device), target_logits)
                )
            target_metrics = _load_final_metrics(args.results_dir, target_name, seed)
            row = {
                "seed": seed,
                "target": target_name,
                **{
                    key: float(np.mean([value[key] for value in divergences]))
                    for key in divergences[0]
                },
                **{
                    f"{key}_delta": target_metrics[key] - reference_metrics[key]
                    for key in target_metrics
                },
            }
            rows.append(row)
            del target
            torch.cuda.empty_cache()
            completed += 1
            elapsed = time.time() - started
            eta = elapsed / completed * (total - completed)
            print(
                f"[drift] {_bar(completed/total)} {completed}/{total} "
                f"seed={seed} target={target_name} sKL={row['symmetric_kl']:.6f} "
                f"dLPIPS={row['eval_lpips_delta']:+.5f} "
                f"dMSE={row['eval_mse_delta']:+.6f} "
                f"elapsed={_hms(elapsed)} eta={_hms(eta)}",
                flush=True,
            )
        del reference
        torch.cuda.empty_cache()

    drift = [row["symmetric_kl"] for row in rows]
    lpips_delta = [row["eval_lpips_delta"] for row in rows]
    mse_delta = [row["eval_mse_delta"] for row in rows]
    ordered = 0
    for seed in seeds:
        by_target = {row["target"]: row for row in rows if row["seed"] == seed}
        if all(name in by_target for name in DEFAULT_TARGETS):
            ordered += int(
                by_target["rc_energy_point"]["symmetric_kl"]
                < by_target["rc_energy_certified"]["symmetric_kl"]
                < by_target["rc_energy"]["symmetric_kl"]
            )
    summary = {
        "spearman_kl_lpips_degradation": _spearman(drift, lpips_delta),
        "spearman_kl_mse_degradation": _spearman(drift, mse_delta),
        "ordered_seeds": ordered,
        "total_seeds": len(seeds),
    }
    green = (
        summary["spearman_kl_lpips_degradation"] >= 0.5
        and summary["spearman_kl_mse_degradation"] >= 0.5
        and ordered >= max(1, len(seeds) - 1)
    )
    summary["verdict"] = "GREEN" if green else "RED"
    payload = {"args": vars(args), "summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\n=== RC-Reference Policy Drift Audit ===", flush=True)
    for key, value in summary.items():
        print(f"{key}={value}", flush=True)
    print(f"saved {args.out}\nRC_POLICY_DRIFT_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
