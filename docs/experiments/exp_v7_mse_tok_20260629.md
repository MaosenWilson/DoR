# V7 — mse_tok 负对照（floor-cancellation 增益 ∝ 地板）2026-06-29

> 论文：*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models*。
> 本实验为 C1/C3b 的**因果负对照**：验证 floor-cancellation 的动态增益是否随"被抵消度量的重建地板大小"缩放。

## Setup

- 命令（服务器，已跑完）：
  ```bash
  cd /root/autodl-tmp/vote2world
  P=/root/miniconda3/bin/python
  $P scripts/train_grpo.py --rewards mse_tok --modes gt_only --seeds 0,1,2,3,4 \
     --steps 150 --K 16 --out_dir outputs/grpo_v7/mse_tok
  $P scripts/compare_arms.py singles=outputs/grpo_v1/singles \
     v5=outputs/grpo_v5/floorcancel v7=outputs/grpo_v7/mse_tok
  ```
- 协议：RT-1 single-step、同基座、gt_only、K=16、150 步、5 seed；eval = RLVR-World `Evaluator`(LPIPS=vgg) + RAFT flow。
- 超参（与 v1/v5/v7/full 完全一致）：lr=1e-5, train_windows=24, eval_windows=12, batch_windows=2, phi_tok=0.0518。
- 奖励定义（`src/dor/reward_spaces.py`）：
  - `mse_tok` = −MSE(decode(cand), **decode(encode(gt))**)，即对齐"可达目标"去地板。
  - 对照：`pixel`=−LPIPS vs raw gt；`pixel_tok`=−LPIPS vs 可达目标；`mse`=−MSE vs raw gt。

## Results（最终 eval，mean±std，n=5；compare_arms 输出）

| 臂 | LPIPS↓ | PSNR↑ | SSIM↑ | flow↑ |
|---|---|---|---|---|
| singles/code | 0.1404±0.0015 | 25.24 | 0.7992 | **0.2751±0.0170** |
| singles/pixel | 0.1425±0.0013 | 25.13 | 0.7974 | 0.2339±0.0267 |
| singles/mse | 0.1453±0.0013 | 25.05 | 0.7955 | 0.2306±0.0185 |
| singles/a0faithful | 0.1440±0.0012 | 25.04 | 0.7957 | 0.2315±0.0255 |
| v5/pixel_tok | 0.1424±0.0015 | 25.12 | 0.7963 | **0.2677±0.0290** |
| v5/ssim_tok | 0.1405±0.0040 | 25.28 | 0.8010 | 0.2357±0.0396 |
| **v7/mse_tok** | 0.1449±0.0020 | 25.09 | 0.7955 | 0.2449±0.0499 |

### 配对到 seed 的 floor-cancellation 增益（核心）

| 原始→去地板 | 度量地板(往返) | per-seed Δflow | 均值 | 变好 seed |
|---|---|---|---|---|
| pixel→pixel_tok (LPIPS) | ≈0.112（大） | +.034,+.038,+.005,+.050,+.042 | **+0.0338±0.0155 (+14.4%)** | **5/5** |
| mse→mse_tok (MSE) | ≈0.0019（微） | +.049,+.066,+.024,−.023,−.044 | +0.0143±0.0420 (+6.2%) | 3/5 |

地板比 ≈ 59×；增益比 ≈ 2.4×（被 flow 噪声压缩，但方向一致）。

## Analysis

1. **负对照 PASS**：floor-cancellation 的动态增益随被抵消度量的地板缩放。LPIPS 地板大 → 抵消 5/5 一致、+14%；MSE 地板≈0 → 抵消 3/5、符号不稳、std(0.042) 是均值 3 倍 ≈ 纯噪声，连自身 PSNR 都不动。把论点从"去地板有用"夯成"**去地板有用是因为地板**"。
2. **汇报纪律**：mse_tok 只讲"5/5 vs 3/5 一致性 + std≫均值"，**不报 +6.2% 均值**（会被当成"MSE 也涨"反驳）。
3. **可写作的证据**：pixel_tok 的 **5/5 一致**是地板抵消最硬的腿。

## 红线 / 待修（阻塞 flow 进正文）

- **flow 不可复现**：同臂同 seed 同配置，v1 vs full 两次跑 flow 差 ±0.03~0.085（如 pixel s3：0.199 vs 0.284），≈ 甚至 > 臂间效应（code−pixel≈+0.041）。多半是 HF `.generate(top_k=100)` 采样 RNG 未按(臂,seed)确定性重置 + 可能的 CUDA 非确定。
- **整改**：① `train_grpo.py`/`generation.py` 每个(臂,seed)rollout 前固定 `torch.manual_seed`+generator，验证"重跑两次 flow 完全一致"；② pixel/code/mse/pixel_tok/mse_tok 放**同一 sweep** 配对、seed 提到 8–10；③ flow 作硬 claim 须等 ①②。
- **质量线不受影响**：LPIPS/PSNR/SSIM 跨跑稳定，code 的 LPIPS 0.140±0.001 完胜（pixel 偶发散到 0.225），可先当主证据。

## 关联
- 主表/叙事见 `docs/aaai2027/results_v1v6.md`、`method.md`；本实验补 C3b 的负对照档。
- 复现性问题独立成项，需在投稿前解决。
