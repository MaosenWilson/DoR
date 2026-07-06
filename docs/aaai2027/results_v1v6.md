# 验证结果（贡献 ↔ 证据，2026-06-29）

> 论文：*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models*。配套 `method.md`。所有数字来自 RT-1 single-step、同基座、K=16、150 步、5 seed、eval 用 RLVR-World `Evaluator`（LPIPS=vgg）+ RAFT flow。

---

## 主对比表（所有臂，最终 eval，mean±std，n=5）

| 臂 | 说明 | LPIPS↓ | PSNR↑ | SSIM↑ | flow↑(动态) |
|---|---|---|---|---|---|
| **code** | 解码前 FSQ RMS（DoR 核心） | **0.1404±0.0015** | 25.24 | 0.7992 | **0.2751±0.017** |
| pixel | −LPIPS vs 原GT | 0.1425±0.0013 | 25.13 | 0.7974 | 0.2339±0.027 |
| mse | −MSE vs 原GT | 0.1453 | 25.05 | 0.7955 | 0.2306 |
| a0faithful | −(MSE+LPIPS)＝**RLVR 真实奖励** | 0.1440 | 25.04 | 0.7957 | 0.2315 |
| **pixel_tok** | −LPIPS vs **可达目标**（去地板） | 0.1424±0.0015 | 25.12 | 0.7963 | **0.2677±0.029** |
| ssim_tok | SSIM vs 可达目标 | 0.1405±0.0040 | 25.28 | 0.8010 | 0.2357 |
| dorw | 地板加权融合(code+mse+lpips) | 0.1408 | 25.25 | 0.7987 | 0.2709 |
| hybrid_tok | z融合(code+pixel_tok) | 0.1411 | 25.24 | 0.7986 | 0.2575 |
| drgrpo/dorw | Dr.GRPO（无过滤） | 0.1407 | 25.28 | 0.7995 | 0.2739 |
| drgrpo/pixel | Dr.GRPO on pixel | 0.1440 | 25.02 | 0.7951 | 0.2568 |
| v3 | Dr.GRPO+地板过滤 | 0.1467 | 24.71 | 0.7902 | 0.2055 |

---

## 贡献 ↔ 证据

### C1 地板存在、量级度量依赖（进展4，eval-only）——**旁证，非主证据**

> 论证已按导师第二轮意见修正（见 `clarify_token_is_feature.md` §6）：**不再用"LPIPS/SSIM 地板 > 信号"作主证据**。地板**有害**的主证据是度量无关的 C2 秩翻转率。

奖励的地板是 **per-sample full-reference** 量，量级随度量变（往返地板 $\phi_{\text{tok}}$ vs 帧间动态信号，同度量）：

| 度量（逐样本） | 往返地板 $\phi_{\text{tok}}$ | 帧间动态信号 | 谁大 |
|---|---|---|---|
| LPIPS | 0.112 | 0.082 | 地板 > 信号 |
| SSIM | 0.841 | 0.902 | 地板 ≳ 信号 |
| MAE | 0.0246 | 0.0226 | 地板 ≈ 信号 |
| MSE | 0.0019 | 0.0034 | 地板 < 信号 |
| PSNR | 27.40 | 27.29 | 地板 < 信号 |

读法（修正后）：
- 地板对每个度量都 >0，但**量级是 (度量, tokenizer) 对的性质**——感知/结构度量（LPIPS/SSIM）大、像素 MSE 小。RLVR-World 的奖励恰用 LPIPS → 最脆。
- **"地板 > 信号"只作度量依赖的旁证**：LPIPS/SSIM 是 full-reference 保真度量，绝对差距会被其对像素偏差的敏感性放大，不足以独立证明地板有害。
- 地板**有害**的硬证据 = **C2 的秩翻转率 $\arccos(\rho)/\pi$（无量纲、跨度量）**，见下。
- 以上均为 **per-sample 量**；分布级特征指标（FVD / FD-DINOv2）用于最终评测、**当不了 per-sample 奖励**。

### C2 机理：翻转率 = arccos(ρ)/π（B2，无参照，5 seed）
- 实测组内翻转率 ≈ **0.185** vs 闭式上界 arccos(ρ)/π ≈ **0.186**，5 seed 差 ~0.002，逐 seed 吻合。
- 弱信号窗口翻转 **~30%**，强信号窗口 ~10%；`corr(翻转率, 信号)=−0.79`。

### C3a code 是单分量最优（主表）
- code 在 LPIPS/PSNR/SSIM/flow 全第一档、**flow 0.275 最高**、std 最小、5 seed 不发散。
- **击败 RLVR 真实奖励 a0faithful**（0.140 vs 0.144；flow 0.275 vs 0.232）。

### C3b 地板抵消（pixel_tok）成立——图像侧贡献
- **pixel_tok vs pixel：flow 0.2677 vs 0.2339（+14%）**，LPIPS/PSNR 持平（gap > std）。
- 即"对齐可达目标去地板"让被污染的 LPIPS 奖励**重新学到动态**；pixel_tok 的动态逼近 code。
- → floor-cancellation 原理在**解码后**也成立（与 code 在解码前互为实例）。

---

## 负结果（写成消融，反向支持核心论点）

| 尝试 | 结果 | 说明 |
|---|---|---|
| 多分量融合 dorw | ≈ code | 地板权重把 LPIPS(0.018)/MSE(0.39) 压向 0，塌成 code |
| 干净分量融合 hybrid_tok | flow 0.258 < code 0.275 **且 < pixel_tok 0.268** | 两排序部分冲突 → 折中更差 |
| Dr.GRPO（无过滤） | 中性（≈标准 GRPO） | 干净 reward 无排序噪声可修 |
| Dr.GRPO + 地板过滤(v3) | 全面变差 | 硬过滤删数据 + 破坏 std 对地板组的天然降权 |

**统一结论：增益来自"奖励在哪个空间 / 对齐哪个目标"，不是融合或改 GRPO。** 负结果强化主张。

---

## 一句话总结
**问题（地板，跨度量）→ 机理（翻转率 bound，已验）→ 方法（floor-cancellation：code 解码前 + pixel_tok 解码后，均去地板、均胜 raw 版的动态）**；融合与 GRPO 花招经实验证伪，作诚实消融。

---

## 论文图表清单（每张对应哪个贡献 / 用哪份数据）

| 图/表 | 内容 | 支撑贡献 | 数据来源 |
|---|---|---|---|
| **Fig 1** | 地板跨度量柱状：各度量 地板 vs 帧间信号（LPIPS/SSIM 地板≥信号） | C1 | 进展4 `outputs/analysis/floor_metrics.json`（`probe_floor_metrics.py`） |
| **Fig 2** | 翻转率 vs 闭式上界 $\arccos(\rho)/\pi$（散点/曲线，5 seed）+ 翻转率随 σ⋆ 分层 | C2 | `analyze_rank_flip.py` on `reward_spaces_s*.npz` |
| **Table 1** | 主表：各臂 × {LPIPS,PSNR,SSIM,flow}，5 seed mean±std，标注解码前/后 | C3a/C3b | `compare_arms.py`（grpo_v1/singles + v5 + dorw…） |
| **Fig 3** | pixel_tok vs pixel 的 flow 提升（+14%）+ 训练曲线 | C3b（地板抵消） | grpo_v1/singles, grpo_v5/floorcancel |
| **Fig 4** | 消融:融合(dorw/hybrid_tok)≈/<code、GRPO(drgrpo/v3)无益——柱状 | 负结果（强化主张） | grpo_v1/dorw, v3, v4, v6 |
| **Fig 5（主观，可选）** | 随机 + 运动 case:pixel(静止/糊) vs code/pixel_tok(抓住运动) | 佐证 | ckpt 导出 |

> 出图脚本待写（matplotlib，从上述 json/npz 读）；数据全部已在服务器 `outputs/`，纯离线、无需再训练。

## 待办（到投稿）
1. 写图脚本（Fig1–4，数据现成）。
2. Method/Experiments 英文初稿（`method.md` + 本文件为骨架）。
3. （可选/导师定）多步 rollout 强化动态；FD-DINOv2/CMMD 评测补充。
