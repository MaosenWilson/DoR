# Experiments：证据账本与新增验证合同

> 只记录可复核事实、固定统计协议和待完成实验。历史搜索过程移入 `_archive/`，不得用历史最佳 seed 替代正式汇总。

## 1. Experimental Setup

### 1.1 System

- Dataset：RT-1 `fractal20220817`；
- Tokenizers：single-step CNNFSQ；multi-step compressive FSQ；
- World model：Llama-style autoregressive video world model；
- Candidate sampling：HF `.generate`，temperature 1.0，`top_k=100`，$G=16$；
- Reward：raw 或 reachable-target `-(MSE+LPIPS)`；
- Optimizer：lean on-policy group-normalized policy gradient；multi-step KL coefficient 0.001。正式复核使用 RLVR-World/VERL 的 `low_var_kl`；2026-07-13 前多步结果使用线性 sampled log-ratio，标为 provisional；
- Multi-step protocol：$T=8$，30 steps，batch windows 2，24 train + 8 held-out windows。

### 1.2 Evaluation

Headline metrics 均相对 raw real frames：LPIPS-vgg、MSE、PSNR、SSIM。LPIPS-last 单独测 rollout tail。dmotion/flow 为动态 readout；DINO-KID/FD 为 set-level distributional readout，绝不进入训练 reward。

### 1.3 Statistical rules

1. matched comparison 使用相同 seed、split、sampling protocol；
2. seed 是训练重复单位，horizon cells 不是独立 seeds；
3. 报告 mean $\pm$ sample SD、paired deltas、wins 和双侧 paired test；
4. calibration/replay 对同 episode 多窗口使用 episode-cluster bootstrap；
5. 固定 $n=10$ 扩种后停止，不继续补 seed 追显著性；
6. 方法选择不展示“最好 seed”。

## 2. C1：Rank-Corruption Diagnosis

### 2.1 Same-candidate closure

184 个 held-out contexts、5 次 fresh generation，共 920 candidate groups，$G=16$。五次 generation 共享 contexts，因此只作为 five-repetition diagnostic，不把 920 当独立 trials。

| reward space | $\rho_{raw}\rightarrow\rho_{RC}$ | flip$_{raw}\rightarrow$flip$_{RC}$ | theory raw/RC |
|---|---:|---:|---:|
| LPIPS | 0.838→0.863 | 0.162→0.151 | 0.158/0.148 |
| MSE | 0.801→0.824 | 0.181→0.168 | 0.184/0.173 |
| MSE+LPIPS | 0.841→0.867 | 0.160→0.149 | 0.156/0.146 |

三个空间均在 5/5 generation repetitions 中降低 flip；$\arccos(\rho)/\pi$ 的预测误差小于 0.005。该实验支持“candidate-dependent rank perturbation 可测且 RC 可降低”，不证明 pre-decode reference 是唯一正确排序。

### 2.2 Cross-codec / cross-horizon audit with episode-cluster bootstrap（2026-07-12，`audit_rank_calibration.py`）

零训练统一审计，聚类单位 = episode（20 episodes；同 episode 的 groups 不作独立样本），bootstrap 2000：

| stratum | groups | $\rho_{raw}\rightarrow\rho_{RC}$ | $\Delta\rho$ [95% CI] | boot-$p$ | $\Delta$flip | boot-$p$ |
|---|---:|---|---|---:|---:|---:|
| single-step CNN-FSQ | 148 | 0.850→0.866 | +0.0168 [+0.0035, +0.0289] | 0.0075 | −0.0082 | 0.030 |
| multi-step compressive FSQ（3 gen × 64 win × h2–7） | 1152 | 0.726→0.740 | +0.0145 [+0.0105, +0.0176] | <5e-4 | −0.0087 | <5e-4 |

- 逐 horizon h=2..7 全部显著（$\Delta\rho$ +0.0117~+0.0182，$\Delta$flip −0.0077~−0.0102，boot-$p\le0.001$）；$\arccos(\rho)/\pi$ 律在两种 codec、校准前后均成立（平均偏差 ≈0.03）。
- **RIR 剂量响应（预注册趋势）**：multi 四分位 $\Delta\rho$ = +0.0083/+0.0120/+0.0173/+0.0203 单调；cluster-boot Spearman(RIR, $\Delta\rho$) = +0.120 [+0.027, +0.202]，$p$=0.007。single-step 趋势 +0.151（单侧 $p$=0.04，最高分位驱动）——RIR 只作 severity stratification，不作在线 gate（method.md §3.2 边界）。
- 意义：C1 的 rank repair 不再依赖单一 tokenizer，迁移到更强压缩的 multi-step FSQ 与全部预测 horizon。红线：不做 partial-RC 系数、不把 RIR 变 context 权重、不人为构造压缩等级。
- 产物：`outputs/analysis/rank_calibration_audit.json`；底料 = `calibration_spatial.npz` + `temporal_reliability_cache.npz`（复用）。

## 3. C1：RC Training Transfer

### 3.1 Multi-step sequence GRPO, fixed seeds 0--9

| comparison | metric | mean delta | wins | paired $t$ | two-sided $p$ | status |
|---|---:|---:|---:|---:|---:|---|
| seq-RC − seq-raw | LPIPS $\downarrow$ | -0.00263 | 6/10 | -0.99 | 0.346 | not confirmed |
| seq-RC − seq-raw | LPIPS-last $\downarrow$ | -0.00454 | 5/10 | -0.84 | 0.420 | not confirmed |
| seq-RC − seq-raw | MSE $\downarrow$ | -0.000412 | 6/10 | -1.26 | 0.239 | not confirmed |

旧 seeds 0--4 的 5/5 方向没有在新增 seeds 5--9 复现（LPIPS 仅 1/5）。因此不能声称 RC 在 multi-step sequence GRPO 中独立改善训练；C1 的有效性必须由 local reachability gate、single-step transfer 和完整 factorial interaction 重新界定。

## 4. C2：Temporal-Return GRPO

### 4.1 Main multi-step results, seeds 0--4

| method | seeds | LPIPS $\downarrow$ | LPIPS-last $\downarrow$ | MSE $\downarrow$ |
|---|---:|---:|---:|---:|
| Base world model | eval-only | 0.2157 | 0.2260 | 0.01484 |
| Official RLVR checkpoint | eval-only | 0.2115 | 0.2155 | 0.01378 |
| Seq-GRPO, raw verifier | 0--4 | 0.2082 $\pm$ 0.0064 | 0.2226 $\pm$ 0.0181 | 0.01405 $\pm$ 0.00054 |
| Seq-GRPO, RC verifier | 0--4 | 0.2009 $\pm$ 0.0042 | 0.2096 $\pm$ 0.0046 | 0.01307 $\pm$ 0.00073 |
| Temporal-return GRPO, RC | 0--4 | **0.1980 $\pm$ 0.0034** | **0.2055 $\pm$ 0.0026** | **0.01276 $\pm$ 0.00069** |
| Gain-return GRPO, RC | 0--4 | 0.1989 $\pm$ 0.0070 | 0.2072 $\pm$ 0.0068 | 0.01271 $\pm$ 0.00101 |
| Frame-only GRPO, RC | 0--4 | 0.2086 $\pm$ 0.0035 | 0.2164 $\pm$ 0.0051 | 0.01431 $\pm$ 0.00030 |

相对 official checkpoint，当前 full method 的 LPIPS/LPIPS-last/MSE 相对改善为 6.4%/4.6%/7.4%。该比较只适用于相同 held-out protocol，不声称复现并击败官方完整训练设置。

### 4.2 Direct paired comparison, fixed $n=10$

`return-RC - seq-RC`：

| metric | mean delta | wins | paired $t$ | two-sided $p$ |
|---|---:|---:|---:|---:|
| LPIPS | -0.002803 | 9/10 | -2.90 | 0.0176 |
| LPIPS-last | -0.004799 | 9/10 | -4.14 | 0.0025 |
| MSE | -0.000272 | 7/10 | -1.68 | 0.1270 |

预注册 primary LPIPS 通过，Holm-adjusted $p=0.0353$；LPIPS-last Holm-adjusted $p=0.0076$；MSE 仅方向支持。新增 seeds 5--9 上 LPIPS 5/5 改善，说明 C2 不是旧 seeds 驱动。

### 4.3 Structural ablation

| comparison | LPIPS delta | LPIPS-last delta | result |
|---|---:|---:|---|
| frame-only − temporal return | +0.0105 | +0.0109 | 0/5 better；$t=+6.2/+5.1$ |
| frame-only − sequence RC | +0.0077 | +0.0068 | 0/5 better |
| gain-return − temporal return | +0.0009 | +0.0017 | no gain |
| horizon-KL − temporal return | +0.0008 | +0.0013 | no gain, 3 seeds |

frame-only 结果说明收益来自 future-return structure，而非仅把 sequence reward 切成逐帧优势。

### 4.4 Horizon signature

5-seed `return-RC - seq-RC`：

| horizon | LPIPS $\downarrow$ | MSE $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ |
|---:|---:|---:|---:|---:|
| 2 | -0.001600 | +0.000007 | +0.043961 | +0.001196 |
| 3 | -0.002211 | -0.000180 | +0.111320 | +0.002836 |
| 4 | -0.002405 | -0.000255 | +0.157662 | +0.003374 |
| 5 | -0.003407 | -0.000396 | +0.207446 | +0.005203 |
| 6 | -0.003541 | -0.000461 | +0.239727 | +0.004205 |
| 7 | -0.004083 | -0.000574 | +0.234541 | +0.005021 |

该趋势与延迟误差传播一致，但 horizon cells 在 seed 内相关。正式统计应使用 seed-clustered slope 或 paired seed summary，不能把 30 cells 当独立 observations。

## 5. Boundary Readouts

### 5.1 Motion

Temporal Return 的 dmotion 不支持稳定改善；论文不声称更好的 dynamics。后续实验把 dmotion 作为 non-inferiority boundary，而不是从主表删除。

### 5.2 Distributional realism

单 seed eval-only KID-DINOv2：base 5.08 < official RLVR 6.58 < return-RC 9.82 < seq-RC 12.42 < seq-raw 13.16。所有 post-trained checkpoints（包括官方）相对 base 的分布指标变差；该结果显示 full-reference fidelity 优化的分布代价，不支持“整体视频生成质量全面提升”。

## 6. Compact Negative Ledger

| intervention | evidence | paper role |
|---|---|---|
| Additional MSE/SSIM/code/gradient fusion | learned weights collapse or training no gain | minimal-verifier boundary |
| Extended IQA panel | unstable across confidence margins | appendix |
| Adaptive Rank-Guard | 3-seed training six metrics worsen | appendix |
| Local-floor spatial weighting | replay primary RED；episode-cluster audit | appendix |
| Dr.GRPO / REAL / GSPO / segmental | neutral or harmful | one-sentence boundary |
| Gain shaping / horizon-aware KL | no gain over plain return | main ablation |

这些结果不能写成“穷尽 reward/GRPO 设计空间”，也不计为独立贡献。

## 7. Missing Decisive Experiments

### E0. Official-objective implementation audit

- 官方视频脚本：`kl_loss_type=low_var_kl`、`kl_loss_coef=0.001`、`ppo_epochs=1`；
- 本地旧实现：线性 $\log\pi-\log\pi_{ref}$，与官方不一致；
- 2026-07-13 已实现 $\exp(k)-k-1$（$k=\log\pi_{ref}-\log\pi$）及公式/梯度方向单测；
- 必须完成服务器 smoke 后，用相同 seeds 重跑关键 `seq-RC` / `return-RC`，旧显著性不得直接进入 headline。

### E1. Fixed $n=10$ confirmation

- 旧线性 KL 协议已完成：C2 primary 通过，C1 multi-step independent effect 未通过；正式 `low_var_kl` 复核待完成；
- primary：final full-rollout LPIPS；
- success：mean delta <0、至少 7/10 seeds、two-sided paired $p<0.05$；
- 不因结果临界继续扩 seed。

### E2. Complete $2\times2$ causal matrix

新增 `raw verifier + temporal return` seeds 0--9。计算 verifier main effect、credit main effect 和 interaction。只有该矩阵完成后才可以说 RC 与 Temporal Return 互补。

### E3. Local Reachability / Metric Projection Gate

1. 检验 encode-decode target 是否为 FSQ 合法邻域内的局部 metric optimum；64-window gate 已 64/64 找到更优 code；
2. 数值验证 raw/RC pair-gap 的精确 residual decomposition；
3. constant-floor subtraction 必须与 raw ranking/gradient 完全相同；
4. 与等失真平滑 target、residual-direction control 比较，排除一般预处理解释；
5. 邻域搜索稳定找到更近 raw GT 的合法 code，因此 encoder-RC 降级为 baseline，方法升级为 fixed-budget MRRT 后做训练 gate。

其中第 3 项已通过数值单测：context-constant floor subtraction 的排序和组归一化 advantage 在
浮点精度内完全相同。它是“不可能有效”的分析对照，不占用训练臂。

64-window 正式结果（固定 8 positions、2 rounds）：64/64 改善；relative gain mean/median/q05
=0.834%/0.770%/0.382%；absolute LPIPS+MSE gain $0.000961\pm0.000450$；LPIPS 64/64
下降（mean $-0.000953$），MSE 39/64 下降（mean $-8.18\times10^{-6}$）；平均 token-cell
Hamming fraction 0.00601。全部窗口接受 2/2 moves，故不得称搜索已收敛。

下一判据：MRRT 在 held-out raw-GT LPIPS/MSE/SSIM 上超过 encoder-RC，且 matched-random legal
target 不出现同等收益，才可成为 headline 方法；否则只保留 local-optimality diagnosis。

### E4. Temporal Correspondence Controls

固定同一 RC/raw verifier，比较 frame-only、截断 $L=1/3$、full return 和 candidate-shuffled return。后者逐 horizon 保留 reward multiset，仅破坏候选身份的跨时间对应。先 paired 3 seeds；只有 aligned/full 相对所有 control 同向才扩 5 seeds。

### E5. Horizon-length generalization

仅在 E1 支持 C2 后运行 $T\in\{4,6,8\}$ paired 3 seeds，检验 return advantage 是否随 horizon 增长。不得选择性只报告有利长度。

## 8. Claim-Evidence Contract

| proposed claim | minimum evidence | current status |
|---|---|---|
| RC reduces rank corruption | same-candidate closure across metrics/repetitions | met |
| RC improves multi-step sequence training | fixed $n=10$ raw-vs-RC | not supported |
| Temporal Return improves multi-step fidelity | fixed $n=10$ paired primary | met |
| RC and Temporal Return are complementary | complete $2\times2$ matrix | missing |
| Encoder reconstruction is a locally justified reachable target | 64/64 存在更优合法邻域 code | rejected；升级为 MRRT |
| Benefit grows with rollout length | $T=4/6/8$ paired experiment | missing |
