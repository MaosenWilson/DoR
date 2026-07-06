# Exp B1+B2 — code-space reward beats pixel; rank-flip mechanism validated

**Date**: 2026-06-25  **Blocks**: B1 (pilot, pixel vs code) + B2 (rank-flip theory) + phi fix
**Story**: `docs/aaai2027/story_spine.md` 贡献三(方法有用)+ 贡献二(机理)。

## Environment / setup

- Hardware: RTX 5090 32G (Blackwell sm_120), AutoDL; `torch 2.8.0+cu128`, `transformers 4.51.1`.
- Repo: `/root/autodl-tmp/vote2world`; python `/root/miniconda3/bin/python`.
- Base ckpt (same start for every arm): `thuml/rt1-world-model-single-step-base`
  (Llama, hidden 768, 12 layers, vocab 4633). Tokenizer: `CNNFSQModel256` (FSQ).
- Generation aligned to RLVR-World eval: `temperature=1.0`, `top_k=100`, ignore EOS,
  generate TPF=320 visual tokens. (RLVR-World 视频 WM GRPO 原版用 `rollout.n=5`,
  训练 rollout `top_k=-1`,`kl_loss_coef=0.001`;此处 K=16、kl=0,差异已记录。)
- Domain: RT-1 single-step (fractal20220817, 20 episodes).

### B1 hyperparams
`scripts/train_grpo.py --rewards pixel,code --modes gt_only --steps 150 --K 16
--seed {0..4} --eval_every 10`  (lr=1e-5, batch_windows=2, train/eval_windows=24/12,
kl=0). Eval metrics independent of code objective: LPIPS, PSNR (held-out windows).

### B0 cache (for B2)
`scripts/cache_reward_spaces.py --n_windows 200 --K 16 --seed {0..4}`; 184 usable
windows/seed. phi_tok (reward-noise floor) = LPIPS(decode(encode(gt)),gt).

## Results

### B1 — final (step 150), mean ± std over 5 seeds
| metric | pixel (A0) | **code (A6, DoR)** | paired test |
|---|---|---|---|
| LPIPS ↓ | 0.0816 ± 0.0026 | **0.0751 ± 0.0017** | code 5/5, t=−5.54, bands DISJOINT |
| PSNR ↑ | 24.80 ± 0.24 | **25.34 ± 0.13** | code 5/5, t=+5.02, bands DISJOINT |

- Both metrics are **independent of code's training objective** (code_rms excluded from headline).
- Curve shape (seed 0, representative): pixel PSNR peaks ~step 40 (24.77) then **degrades**
  to 24.59; code improves **monotonically** to 25.42. Same lr → difference is the reward
  space, not optimization. The 40-step proof (`exp_codereward_multiseed_20260616.md`)
  stopped at the pixel peak; extending to 150 exposes the divergence. → Fig 4.

### B2 — within-group rank-flip vs closed-form bound (reference-free)
Clean ordering = code reward (pre-decode, ~zero floor); measured = post-decode rewards.
Per-pair flip prob bound = arccos(ρ)/π, ρ = within-group corr(code, measured).

| seed | emp_flip (pixel) | bound | \|gap\| | ρ |
|---|---|---|---|---|
| 0 | 0.1875 | 0.1879 | 0.0004 | 0.790 |
| 1 | 0.1843 | 0.1863 | 0.0020 | 0.789 |
| 2 | 0.1840 | 0.1838 | 0.0002 | 0.797 |
| 3 | 0.1833 | 0.1846 | 0.0013 | 0.796 |
| 4 | 0.1853 | 0.1884 | 0.0031 | 0.784 |

- Empirical flip rate matches the Gaussian closed-form bound to ~0.002 on **every seed**
  → the floor-noise model is essentially exact. → Fig 5.
- Other post-decode metrics also match (seed 0): −MSE emp 0.178/bound 0.183; SSIM 0.216/0.220.
- **Floor-dominated regime (seed 0)**: pixel flips 29.5% of candidate pairs at low clean
  spread σ⋆=0.012, vs 9.6% at high σ⋆=0.054; `spearman(flip, σ⋆) = −0.79`.
  → GRPO consumes the corrupted ranking; ~1/3 of pairs scrambled when signal is weak.
- phi_tok ≈ 0.0518 LPIPS (Fig 1 floor value).

### A5 vs A6 (quantization ablation), after phi GAP fix
Within-group rank-pres vs DINOv2 (seed 0): A5 phi = 0.287 ≈ A6 code = 0.297 (was
0.114 before fixing the GAP-pooling bug in `phi_rms`). → quantization barely costs
signal; the win is **pre-decode**, not the discrete codes specifically.

## Conclusions (mapped to story spine)
- **贡献三 (方法)**: code-space reward → better INDEPENDENT quality, 5/5 seeds, disjoint
  error bars, p<0.01. Solid headline.
- **贡献二 (机理)**: rank-flip rate = arccos(ρ)/π validated across 5 seeds; floor scrambles
  up to ~30% of within-group orderings in the weak-signal regime. Reference-free.
- **A5≈A6**: isolates that the gain is from leaving the decoder, not from quantization.

## Caveats / known issues (for Limitations)
- Single domain (RT-1 single-step), single-step prediction. FVD / 2nd domain / multi-step
  → B3 (H800). FVD on single-step needs short rollout (R1, open).
- The **aggregate "rank-pres vs DINOv2" figure is dropped**: its post-decode reference is
  itself floor-corrupted in the regime of interest (informativeness 0.25 @ low σ⋆ vs 0.55
  @ high σ⋆), structurally biasing toward pixel. Mechanism evidence uses B2 (reference-free)
  + σ⋆ stratification instead. Two WEAK aggregate runs are explained by this, not a refutation.
- **A3 floor-calibrated baseline still a strawman**: per-window floor subtraction is
  rank-invariant within group (Spearman unchanged); needs per-candidate floor (R2, open).
- B1 used kl=0 (RLVR-World GRPO used kl=0.001); comparability axis, not yet ablated.

## Update 2026-06-26 — full arm sweep (MAIN Table 1), metric alignment, SSIM instability

### Full A0–A6 sweep done (8 arms × 5 seeds, 150 steps, K16, gt_only), eval LPIPS-alex
Independent metrics (LPIPS-alex/PSNR), mean±std over 5 seeds:
- A6 code 0.0751±0.0017 / 25.34±0.13  (best, most stable)
- A2 SSIM 0.0752±0.0014 / 25.39±0.13  (ties code on LPIPS/PSNR — see instability below)
- A0 pixel 0.0816±0.0026 / 24.80±0.24 (ref)
- A1 mse 0.0853±0.0088 ; A3 floor(const) 0.0815±0.0036 ≈ A0 (t=−0.07, confirms rank-invariance);
  A3 floorpc(per-cand) 0.1336±0.1157 (worse + high variance); A4 multi 0.1163±0.0826; A5 phi 0.1066±0.0624.
- Rebuttals堵住: const-floor=A0 (theory+exp); per-cand floor worse; MSE/multi worse → not "just better pixel reward".
- A5 phi unstable as a *training* reward (good rank-pres but bad/unstable training) → **quantization
  helps training stability**; refines A5-vs-A6: pre-decode necessary, FSQ code (pre-decode + quantized) is the sweet spot.

### SSIM instability (reproduction, key finding)
Re-running pixel/ssim/code ×5 seeds (independent run) reproduced code/pixel within CUDA noise, but **SSIM
diverged on 2/5 seeds** (s2 0.251, s4 0.286 vs ~0.075 in the first run). → SSIM's apparent tie with code is
**non-reproducible**; SSIM-as-reward is fragile (bounded metric, tiny within-group spread, z-score amplifies noise).
**code's real edge over SSIM = reproducible stability** (smallest std on every metric, both runs).

### Dynamics eval (RAFT flow + frame-delta cosine), single-step
flow: pixel 0.257±0.066, ssim 0.223±0.042, code 0.252±0.019 (t vs pixel ns); dmotion: pixel highest.
→ **single-step dynamics metrics do NOT cleanly separate code from pixel** (only inject the floor once).
code is most *stable* (smallest flow std). Multi-step rollout deferred (focus single-step per advisor; RLVR-World
splits single/multi-step as separate works).

### Metric alignment to RLVR-World (decided + implemented)
RLVR-World single-step `Evaluator` reports **MAE/MSE/PSNR/SSIM/LPIPS(net='vgg')** (+ FVD/FID via I3D, i3d_path
default None). Their GRPO **reward** uses LPIPS-**vgg** (`verl/ivideogpt/lpips.py`). We were on LPIPS-**alex** and
not logging MAE/MSE/SSIM → not one-to-one. Fixes (opt-in, code/phi/mse/ssim arms' training unaffected):
- **eval** now uses their exact `Evaluator` (`ivideogpt.utils.video_metric`) → MAE/MSE/PSNR/SSIM/LPIPS-vgg, bit-comparable.
- **reward** `dor.metrics.Metrics` LPIPS alex→**vgg** → A0 is now the *faithful* RLVR-World baseline (affects LPIPS-using
  arms: pixel/floor/floorpc/multi; mse/ssim/phi/code reward unchanged).
- **all training now saves a final checkpoint** (`{out_dir}/ckpt/{arm}_{mode}_s{seed}`, model.safetensors).
- Circularity hygiene: each arm's reward metric = one eval column (home advantage, marked `*`); headline reads code's
  wins on NON-home columns (all LPIPS/PSNR/SSIM/FID are independent of code's code_rms reward).
- NB: const-floor `phi_tok=0.0518` is alex-scale/nominal — rank-invariant anyway (= A0); floorpc uses live vgg floor (self-consistent).

### MAIN experiment now running → `outputs/grpo_full`
`pixel,code,ssim,floorpc,multi,phi,mse,floor × seeds 0-4`, 150 steps, K16, vgg reward + Evaluator + ckpt + RAFT flow.
~21–23h (vgg heavier than alex). Resumable (skip existing). **This is the canonical Table 1** (supersedes the
alex-based `outputs/grpo` and `outputs/grpo_flow`). Aggregate: `scripts/aggregate_table1.py --out_dir outputs/grpo_full`.

### FID — eval-only, no retraining
Because checkpoints are saved, FID/FVD are an **eval-only** add-on over `grpo_full/ckpt/*` (load → generate → I3D
features → Fréchet). Pending: locate a trusted I3D torchscript. No training re-run needed.

## Update 2026-06-26b — 资源可行性 + RLVR-World 放出的模型 + FID 定位

**约束**：项目单人单卡（RTX 5090）。**绝不从头训 base/tokenizer**；一切都是在 RLVR-World 放出的预训练 base 上做轻量 GRPO 微调（~25min/run）。

**RLVR-World 实际放出的模型（决定哪些实验可行）**：
- 视频（与本文相关）**只有 RT-1 一个域**：`rt1-frame-tokenizer`、`rt1-world-model-single-step-{base,rlvr}`（在用）、**`rt1-world-model-multi-step-{base,rlvr}`（多步 base 已放出）**、**`rt1-compressive-tokenizer`（第二个 tokenizer，更激进压缩）**。
- 其它"域"（bytesized32 text-game、webarena web-nav）全是**语言世界模型**，无图像有损解码、与地板 thesis 无关。

**可行性结论**：
- **第二个视频域 = 不可行且不必要**：RLVR-World 视频本身就单域；单人单卡无法训新视频 WM。汇报口径：对齐 base 工作的单域设定。
- **多步 = 唯一可行的"动态"强化**：微调 `rt1-world-model-multi-step-base`（同配方，生成长 8×、序列变长、可能 K 降到 8），不是从头训。仅在导师认为"动态必须坐实"时才投入（几天）。
- **两张几乎免费的普遍性牌（无训练）**：① 跨度量地板（MSE/SSIM/LPIPS 各算 decode∘encode 往返）；② 跨 tokenizer 地板（用 `rt1-compressive-tokenizer` 再测，证明地板随压缩程度变）。

**FID 定位修正**：RLVR-World **视频 RT-1 评测并不上报 FID/FVD**（`eval_vgpt.py` 未调用 `compute_fid/fvd`，i3d_path 默认 None；上报的是 MAE/MSE/PSNR/SSIM/LPIPS）。→ 我们已对齐其**全部上报指标**，**FID 不是对标缺口**，只是可选的额外分布级指标，优先级下调。

**需导师拍板的唯一问题**：单步（问题→机理→方法 + 跨度量/跨tokenizer 普遍性）是否够投 AAAI；还是"动态"必须用多步坐实（多步可微调其 base、但单人单卡要几天）。第二域已排除。

## Next
1. Let `grpo_full` finish → `aggregate_table1.py` = Table 1 (watch whether SSIM diverges again).
2. （可选/低优先）FID = eval-only on saved checkpoints once I3D is sourced — 非对标必需。
3. Fig 4 (curves) + Fig 5 (flip vs bound) + Table 1 plotting.
4. 几乎免费的普遍性补强：跨度量地板 + 跨 tokenizer 地板（无训练）。
5. （待导师拍板）多步 rollout：微调 `rt1-world-model-multi-step-base`。第二视频域已排除（不可行且 RLVR-World 本身单域）。
