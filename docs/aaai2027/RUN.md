# AAAI-2027 当前实验运行手册

> 2026-07-13 起的唯一运行手册。全部前台运行并显示进度/ETA。旧多步结果使用线性 sampled log-ratio，不得与本页 `low_var_kl` 结果混用。

## 0. 固定环境与协议

```bash
cd /root/autodl-tmp/vote2world
P=/root/miniconda3/bin/python
```

正式 multi-step 协议：`T=8, K=16, steps=30, batch_windows=2, train/eval=24/8, lr=1e-5, kl=0.001, kl_type=low_var_kl, deterministic`。正式结果统一写入带 `_lvkl` 的新目录。

## 1. C1：Local Metric-Target Gate（先跑，约数分钟）

```bash
PYTHONPATH=src $P scripts/probe_reachable_projection.py \
  --n_windows 64 --exclude_windows 36 --window_seed 1 \
  --positions 8 --rounds 2 --metric_batch 8 \
  --deterministic \
  --out outputs/analysis/reachable_projection_w64.json
```

固定判决：报告 improved fraction、mean/median/q05 gain 与 Hamming fraction，不根据结果改搜索预算。若多数窗口改善且 mean gain 为正，`D(E(s'))` 不能再称 metric projection；下一步才实现 calibration-only MPRT cache、same-candidate replay 与 preprocessing controls。若门控为红，保留 encoder-RC，停止 MPRT 分支。

## 2. C1：缓存 MRRT 并运行四臂训练（约 9 小时）

```bash
PYTHONPATH=src $P scripts/cache_mrrt_targets.py \
  --train_windows 24 --eval_windows 12 --window_seed 1 \
  --positions 8 --rounds 2 --metric_batch 8 --deterministic \
  --out outputs/mrrt/train_targets.npz
```

缓存必须报告 `MRRT_CACHE_OK`。随后在原始单步 GRPO 下同时运行四臂：

```bash
PYTHONPATH=src $P scripts/train_grpo.py \
  --rewards a0faithful,a0faithful_tok,mrrt,mrrt_random \
  --modes gt_only --seeds 0,1,2,3,4 \
  --steps 150 --K 16 --batch_windows 2 \
  --train_windows 24 --eval_windows 12 --lr 1e-5 --eval_every 10 \
  --reachable_target_cache outputs/mrrt/train_targets.npz \
  --deterministic --out_dir outputs/mrrt/four_arm
```

```bash
PYTHONPATH=src $P scripts/analyze_mrrt_training.py \
  --raw 'outputs/mrrt/four_arm/sweep_a0faithful_gt_only_s*.json' \
  --encoder_rc 'outputs/mrrt/four_arm/sweep_a0faithful_tok_gt_only_s*.json' \
  --mrrt 'outputs/mrrt/four_arm/sweep_mrrt_gt_only_s*.json' \
  --random 'outputs/mrrt/four_arm/sweep_mrrt_random_gt_only_s*.json' \
  --expected_n 5 --out outputs/analysis/mrrt_four_arm_s0_4.json
```

Primary：held-out raw-GT LPIPS。MRRT 必须同时优于 encoder-RC 和 matched-random；MSE/SSIM/flow
作为 secondary/boundary 全量报告。若只改善自己的 target objective 而不改善 held-out raw-GT 指标，
MRRT 降级为诊断，不进入 headline。

## 3. C2：官方 KL 下的 2×2 复核（第一阶段 seeds 0--4，约 5--6 小时）

### 2.1 Sequence-level，raw + RC

```bash
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards raw,rc --adv_temporal seq --seeds 0,1,2,3,4 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --return_horizon 0 --horizon_kl_alpha 0.0 --eval_every 10 \
  --which rlvr --deterministic --out_dir outputs/msp_lvkl_seq
```

### 2.2 Temporal return，raw + RC

```bash
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards raw,rc --adv_temporal return --seeds 0,1,2,3,4 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --return_horizon 0 --horizon_kl_alpha 0.0 --eval_every 10 \
  --which rlvr --deterministic --out_dir outputs/msp_lvkl_return
```

### 2.3 因子分析

```bash
PYTHONPATH=src $P scripts/analyze_msp_factorial.py \
  --seq_raw 'outputs/msp_lvkl_seq/sweep_raw_msp_s*.json' \
  --seq_rc 'outputs/msp_lvkl_seq/sweep_rc_msp_s*.json' \
  --return_raw 'outputs/msp_lvkl_return/sweep_raw_msp_s*.json' \
  --return_rc 'outputs/msp_lvkl_return/sweep_rc_msp_s*.json' \
  --expected_n 5 --out outputs/analysis/msp_lvkl_factorial_s0_4.json
```

扩展规则：只有 `return effect under RC` 至少 4/5 同向且均值改善，才原样补 seeds 5--9；不因某个 seed 好看改变协议。完整矩阵同时回答 RC 主效应、Temporal Return 主效应和 interaction。

## 4. C2：时间对应控制（2×2 通过后；第一阶段 paired 3 seeds，约 3 小时）

Full arm 可直接复用 `outputs/msp_lvkl_return` 的 RC seeds 0--2。只新增三组控制：

```bash
# L=1：仅当前 frame reward
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards rc --adv_temporal return --return_horizon 1 --seeds 0,1,2 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --eval_every 10 --which rlvr --deterministic --out_dir outputs/msp_lvkl_return_L1

# L=3：局部 future credit
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards rc --adv_temporal return --return_horizon 3 --seeds 0,1,2 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --eval_every 10 --which rlvr --deterministic --out_dir outputs/msp_lvkl_return_L3

# 每个 horizon 保持 reward multiset，只打乱 candidate 的时间身份
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards rc --adv_temporal shuffled_return --return_horizon 0 --seeds 0,1,2 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --eval_every 10 --which rlvr --deterministic --out_dir outputs/msp_lvkl_return_shuffled
```

```bash
PYTHONPATH=src $P scripts/analyze_temporal_controls.py \
  --trunc1 'outputs/msp_lvkl_return_L1/sweep_rc_msp_s*.json' \
  --trunc3 'outputs/msp_lvkl_return_L3/sweep_rc_msp_s*.json' \
  --full 'outputs/msp_lvkl_return/sweep_rc_msp_s[0-2].json' \
  --shuffled 'outputs/msp_lvkl_return_shuffled/sweep_rc_msp_s*.json' \
  --expected_n 3 --out outputs/analysis/msp_lvkl_temporal_controls_s0_2.json
```

判决：primary 为 full aligned 相对 shuffled 的 LPIPS 配对差；同时要求 full 不差于 L=1，且 L=1→L=3→full 有基本剂量方向。通过后把三组控制扩到 seeds 3--4；未通过则 C2 只能表述为有效的 block-wise objective，不能声称提升来自正确 temporal correspondence。

## 5. 暂缓实验

- `T=4/6/8` 长度泛化：仅在第 2、3 节通过后运行；
- MPRT 训练：仅在第 1 节正式 gate 通过、且 same-candidate replay 优于 encoder-RC 后运行；
- 新 reward panel、rank weighting、GSPO/REAL/GSPO 等已判负分支不得重启。

## 6. 完成标志

- local target gate：`REACHABLE_PROJECTION_OK`；
- MRRT cache / analysis：`MRRT_CACHE_OK` / `MRRT_ANALYSIS_OK`；
- training sweep：`GRPO_MSP_OK`；
- factorial：`MSP_FACTORIAL_OK`；
- temporal controls：`TEMPORAL_CONTROLS_OK`。
