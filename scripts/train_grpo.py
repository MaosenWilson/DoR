"""Train GRPO under one or more (reward, mode) designs and save held-out curves.

Reward distance D (--rewards):
  pixel   RLVR-World perceptual GT (-LPIPS)            baseline
  code    FSQ code-space RMS (no decode)               Dynamics-over-Reconstruction
  hybrid  alpha*z(pixel) + (1-alpha)*z(code)           fusion
Advantage shaping (--modes, dor.rewards.MODES):
  gt_only / hybrid_add / hybrid_mult  (consensus only reshapes advantage).

Example:
  python scripts/train_grpo.py --rewards pixel,code,hybrid --modes gt_only \
         --steps 40 --K 16 --temperature 0.5 --alpha 0.5 --seed 0
"""
import argparse
import json
import os
import time

# Must be set before the CUDA context is created (i.e. before torch is imported via
# the dor.* modules below). Deterministic cuBLAS GEMM + less fragmentation. Harmless off.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dor.constants import ROOT
from dor.grpo import _hms, train
from dor.reward_spaces import ARMS as REWARDS  # space-sweep superset (A0-A6 + floorpc + hybrid)
from dor.rewards import MODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", default="pixel,code,hybrid")
    ap.add_argument("--modes", default="gt_only")
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid weight on z(pixel)")
    ap.add_argument("--phi_tok", type=float, default=0.0518,
                    help="constant reward-noise floor for the 'floor' arm (B0 floor_lpips mean)")
    ap.add_argument("--weight_temp", type=float, default=1.0,
                    help="dorw weight temperature w_m^tau (0=equal weight, 1=designed)")
    ap.add_argument("--rank_weight", action="store_true",
                    help="Apply rank-reliability soft advantage weights to any mode (mode=rankrel does this automatically)")
    ap.add_argument("--rank_sigma", type=float, default=-1.0,
                    help="Reward-noise std in raw reward units; <=0 infers from reward_weights.json")
    ap.add_argument("--rank_sigma_scale", type=float, default=1.0,
                    help="Multiplier on inferred/provided rank_sigma for sensitivity checks")
    ap.add_argument("--rank_min_weight", type=float, default=0.05,
                    help="Minimum rank-reliability weight so uncertain groups do not silently stop learning")
    ap.add_argument("--dyn_lambda", type=float, default=0.25,
                    help="Weight of code-motion residual term in *_dyn rewards after within-group z-score")
    ap.add_argument("--dyn_gamma", type=float, default=0.25,
                    help="Magnitude-ratio penalty in code-motion residual reward")
    ap.add_argument("--dyn_tau", type=float, default=0.0,
                    help="Disable dynamic auxiliary when GT code-motion norm <= dyn_tau")
    ap.add_argument("--rcmg_pre_weight", type=float, default=0.5,
                    help="MG-RC balance between pre-decode and post-decode domains; fixed paper default=0.5")
    ap.add_argument("--rankcal_weights", default="",
                    help="Frozen JSON from calibrate_rank_reward.py for rankcal_* rewards")
    ap.add_argument("--reachable_target_cache", default="",
                    help="NPZ from cache_mrrt_targets.py, required by mrrt/mrrt_random")
    ap.add_argument("--energy_config", default="",
                    help="Frozen JSON from cache_rc_energy.py, required by *_energy rewards")
    ap.add_argument("--floor_filter", action="store_true",
                    help="Floor-aware GRPO: skip groups with std_k(code)/s_code <= tau*sigma_eta_norm")
    ap.add_argument("--tau", type=float, default=1.0, help="floor-filter threshold multiple")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--batch_windows", type=int, default=2)
    ap.add_argument("--train_windows", type=int, default=24)
    ap.add_argument("--eval_windows", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=0.5, help="additive consensus weight (hybrid_add)")
    ap.add_argument("--beta", type=float, default=0.5, help="multiplicative consensus weight (hybrid_mult)")
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="",
                    help="comma list e.g. 0,1,2,3,4 -> SWEEP mode (per-run json + global ETA + resume)")
    ap.add_argument("--adv_estimator", default="grpo", choices=["grpo", "seg_grpo", "gpseg", "real", "gspo"],
                    help="seg_grpo = v1-v3 segment credit (SUPERSEDED: pooled pseudo-global + "
                         "(1-lam) shrinks the global signal, kept for comparison); "
                         "gpseg = v4 GP-SegGRPO (method.md Sec.3.4): true-global backbone + "
                         "lambda * zero-mean segment residual; lambda=0 or K=1 degenerates "
                         "exactly to standard global GRPO; "
                         "real = Rank-Label VPO, a REAL-style classification objective using "
                         "top/bottom reward-ranked candidates as labels; "
                         "gspo = GSPO-style sequence-level importance-ratio objective.")
    ap.add_argument("--seg_grid", default="2x2", help="spatial segmentation gHxgW (e.g. 2x2->K=4, 4x4->K=16)")
    ap.add_argument("--seg_lambda", type=float, default=0.7, help="blend: lam*segment_adv + (1-lam)*global_adv")
    ap.add_argument("--seg_gamma", type=float, default=0.0, help="codec-aware denominator weight on per-seg floor")
    ap.add_argument("--seg_reward", default="code",
                    choices=["code", "pixtok", "pixtok_dyn", "codec_fused"],
                    help="code=pre-decode (floor~0, gamma off); pixtok=post-decode+per-seg codec floor; "
                         "pixtok_dyn=legacy Phase-C reward (floor-cancelled pix + motion residual, "
                         "superseded for CAST-GRPO, kept for comparison); "
                         "codec_fused=CANONICAL CAST-GRPO reward v3 (segment decomposition + "
                         "reliability q=relu(corr(r_dec,r_tok))^2, zero free hyperparameters, "
                         "+ motion residual; use with --dyn_lambda 0.10 --dyn_gamma 0.25)")
    ap.add_argument("--real_tau", type=float, default=0.5,
                    help="Rank-Label VPO temperature; REAL paper/code use tau=0.5 as the stable default")
    ap.add_argument("--real_pos_frac", type=float, default=0.25,
                    help="Fraction of top-ranked candidates labelled positive for adv_estimator=real")
    ap.add_argument("--real_neg_frac", type=float, default=0.25,
                    help="Fraction of bottom-ranked candidates labelled negative for adv_estimator=real")
    ap.add_argument("--real_min_gap", type=float, default=0.0,
                    help="Skip a group for adv_estimator=real if max(reward)-min(reward) is below this")
    ap.add_argument("--gspo_clip_low", type=float, default=3e-4,
                    help="GSPO lower clipping range for sequence-level ratio")
    ap.add_argument("--gspo_clip_high", type=float, default=4e-4,
                    help="GSPO upper clipping range for sequence-level ratio")
    ap.add_argument("--ppo_epochs", type=int, default=1,
                    help="gspo only: total update epochs per rollout batch. MUST be >=2 for "
                         "GSPO to differ from vanilla GRPO in this harness -- with 1 epoch the "
                         "ratio is identically 1 (fixed-length on-policy), clip never fires, and "
                         "gspo is gradient-identical to the vanilla baseline")
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise-reproducible per seed (det. cuBLAS/cuDNN/SDPA); slower, for final runs")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument(
        "--no_save_checkpoints",
        action="store_true",
        help="keep metric JSONs but skip final model checkpoints (sweep storage control)",
    )
    ap.add_argument("--out", default=f"{ROOT}/outputs/grpo/curves.json", help="legacy single-run output")
    ap.add_argument("--out_dir", default=f"{ROOT}/outputs/grpo", help="sweep-mode per-run output dir")
    args = ap.parse_args()
    _seg_grid = tuple(int(x) for x in args.seg_grid.lower().split("x"))
    if len(_seg_grid) != 2:
        raise SystemExit(f"--seg_grid must be gHxgW, got {args.seg_grid!r}")

    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad_r = set(rewards) - set(REWARDS)
    bad_m = set(modes) - set(MODES)
    if bad_r:
        raise SystemExit(f"unknown rewards {bad_r}; valid: {REWARDS}")
    if bad_m:
        raise SystemExit(f"unknown modes {bad_m}; valid: {MODES}")

    def _train(reward, mode, seed):
        ckpt = (
            None
            if args.no_save_checkpoints
            else os.path.join(args.out_dir, "ckpt", f"{reward}_{mode}_s{seed}")
        )
        return train(mode, reward=reward, alpha=args.alpha, phi_tok=args.phi_tok,
                     weight_temp=args.weight_temp, rank_weight=args.rank_weight,
                     rank_sigma=args.rank_sigma, rank_sigma_scale=args.rank_sigma_scale,
                     rank_min_weight=args.rank_min_weight,
                     dyn_lambda=args.dyn_lambda, dyn_gamma=args.dyn_gamma, dyn_tau=args.dyn_tau,
                     rcmg_pre_weight=args.rcmg_pre_weight,
                     rankcal_weights_path=args.rankcal_weights or None,
                     reachable_target_cache=args.reachable_target_cache or None,
                     energy_config_path=args.energy_config or None,
                     floor_filter=args.floor_filter, tau=args.tau,
                     steps=args.steps, K=args.K, batch_windows=args.batch_windows,
                     train_windows=args.train_windows, eval_windows=args.eval_windows,
                     lr=args.lr, lam=args.lam, beta=args.beta, kl=args.kl,
                     eval_every=args.eval_every, seed=seed, ckpt_dir=ckpt,
                     deterministic=args.deterministic,
                     adv_estimator=args.adv_estimator, seg_grid=_seg_grid,
                     seg_lambda=args.seg_lambda, seg_gamma=args.seg_gamma,
                     seg_reward=args.seg_reward,
                     real_tau=args.real_tau, real_pos_frac=args.real_pos_frac,
                     real_neg_frac=args.real_neg_frac, real_min_gap=args.real_min_gap,
                     gspo_clip_low=args.gspo_clip_low, gspo_clip_high=args.gspo_clip_high,
                     ppo_epochs=args.ppo_epochs)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:  # legacy single-run: all rewards x modes into one --out json
        runs = {f"{r}-{m}": _train(r, m, args.seed) for r in rewards for m in modes}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(dict(args=vars(args), runs=runs), open(args.out, "w"), indent=2)
        print(f"[done] saved {args.out}", flush=True)
        print("GRPO_OK", flush=True)
        return

    # SWEEP mode: reward x mode x seed, per-run json, global progress bar + ETA, resumable
    combos = [(r, m, s) for s in seeds for r in rewards for m in modes]
    total, t0, done = len(combos), time.time(), 0
    os.makedirs(args.out_dir, exist_ok=True)
    for reward, mode, seed in combos:
        out = os.path.join(args.out_dir, f"sweep_{reward}_{mode}_s{seed}.json")
        el = time.time() - t0
        eta = _hms(el / done * (total - done)) if done else "estimating"
        avg = f"{el / done / 60:.1f}min/run" if done else "--"
        bar = "[" + "#" * int(done / total * 20) + "-" * (20 - int(done / total * 20)) + f"] {done}/{total}"
        print(f"\n===== SWEEP {bar}  next: arm={reward} seed={seed} | "
              f"elapsed={_hms(el)} eta={eta} (avg {avg}) =====", flush=True)
        if os.path.exists(out):
            print(f"[skip] {out} exists", flush=True)
            done += 1
            continue
        log = _train(reward, mode, seed)
        json.dump(dict(args=vars(args), run={f"{reward}-{mode}": log}), open(out, "w"), indent=2)
        print(f"[done] saved {out}", flush=True)
        done += 1
    print(f"\n[sweep done] {done}/{total} runs in {_hms(time.time() - t0)}", flush=True)
    print("GRPO_OK", flush=True)


if __name__ == "__main__":
    main()
