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

### 3.A RCAV Gate A：latent action verifier（开发性解剖，未训练 world model）

目标是在把 learned action reward 接入 GRPO 之前，先证明它在真实 transition 和冻结 base-policy candidates 上都有独立预测力。

**D0 data audit.** 服务器已审计 20 episodes / 958 adjacent transitions。数据按 episode 划分 12/4/4 train/calibration/test，不允许相邻帧跨 split。主分析只使用与官方 action ranges 一致的 arm-motion 维度 `0,1,2,4,5,6`；gripper 维度 3 只作附加消融，常量 terminate/base 维度不进入拟合。

**D1 real-transition gate.** 固定 $4\times5$ pooled current/delta/absolute-delta FSQ features，calibration 只选 ridge $\alpha\in\{10^{-4},10^{-3},\ldots,10^2\}$，test 只读一次。Primary 同时要求：

1. test macro mean $R^2>0$，且 6 个 motion dimensions 中至少 4 个 $R^2>0$；
2. 相对 200 次 episode 内 action-label 循环错位 null 的单侧 $p<0.01$；该 null 保留每个 episode 的动作边际分布与时间自相关，只破坏帧转移与动作的对齐；
3. episode-macro normalized RMSE 优于 train-mean predictor。

**D2 generated-candidate gate.** 只从 D1 的 test episodes 按 episode 等额采样 context，冻结 base policy，$K=16$，候选按同一 payload 评分。保存所有 candidate-level action/state/dmotion/flow scores，不只保存汇总结论。Primary 要求：

1. 至少 80% groups 的 $\operatorname{Std}_iR_i^{\mathrm{act}}>10^{-4}$；
2. context-level Spearman$(R^{\mathrm{act}},\mathrm{dmotion})$ 的 90% episode-cluster bootstrap lower bound $>0$；
3. 与 RC-state reward 的 median absolute within-group Spearman $<0.9$，排除同质复制。

RAFT flow 为 confirmatory readout，不因单次噪声设硬门；但若 flow 显著反向，即使 dmotion 过门也必须先做 case audit，不得直接进入 Gate B。

**D1-v1 正式结果.** 20 episodes / 958 adjacent transitions；12/4/4 episode split。$4\times5$ pooled one-step ridge 选择到搜索边界 $\alpha=100$，test mean $R^2=-0.739$，六维均为负，mean NRMSE $1.246>0.959$ mean-action baseline，within-episode circular permutation $p=0.557$。判决：当前 verifier 无资格进入 candidate Gate B 或 GRPO；不判决 action axis 本身。

**D1-v2 预注册矩阵.** 固定 $h=1\ldots4$、offset $\{-1,0,+1\}$、first/mean action target、$4\times5$/$8\times10$ FSQ spatial features。配置和 ridge $\alpha$ 只按 calibration mean $R^2$ 选择。三种 split 的角色严格区分：transition-random 是可记忆上界，episode-blocked 检查同域时间外推，episode-disjoint 是唯一正式泛化判决。Primary gate 要求 episode-disjoint test mean $R^2>0$、circular-shift permutation $p<0.05$、episode-matched retrieval 的 cluster-bootstrap 90% lower bound $>0.5$，并要求完整 transition feature 相对 matched current-state-only predictor 的 normalized-error gain 90% lower bound $>0$；方向 balanced accuracy 仅作 secondary。

**D1-v2 FSQ result: RED.** random winner 为 `h4/8x10/mean/offset−1`，test $R^2=+0.404$；blocked winner $R^2=-0.333$；episode winner `h1/8x10/first/offset0` test $R^2=-0.066$（1/6 positive dimensions），direction BA $0.470$，retrieval $0.500$ [0.481, 0.526]，permutation $p=0.945$。transition-over-state gain $-0.0351$ [$-0.0582,-0.0124$]，4/4 episode 为负。结论只否定 pooled linear FSQ action verifier，并将 random 正结果归因于同轨迹相关/重叠捷径。

**A2.2 decoded-motion oracle.** 冻结 RAFT-small 在真实 frame pairs 上提取 $(u,v,|u|,|v|,\|F\|)$ 空间流场，并与 pooled current RGB 组成 full feature；matched RGB-only 是 shortcut control。严格复用 A2.1 的 grid、split、calibration selection、permutation 和四项 gate，不增加神经网络训练。RAFT 通过而 FSQ 失败才解锁 spatial latent verifier；RAFT 也失败则停止 action-verification 分支。

**A2.2 result: statistical GREEN, semantic hold.** episode winner 为 `h1/8x10/first/offset−1`：$R^2=+0.0055$（3/6 positive dimensions），direction BA $0.541$，retrieval $0.527$ [0.503, 0.554]，permutation $p=0.00498$，transition-over-state gain $+0.0335$ [0.0162, 0.0527]，4/4 episode 为正。该证据证明 decoded flow 包含跨 episode 动作信息，并将失败定位到“pooled linear FSQ interface 或其表征”而非视觉不可观测性；它不能单独证明 tokenizer 已不可逆丢失动作信息。`offset−1` 只识别过去动作，与 prompt 的当前 command 不一致，不能解锁 reward training。

**A2.3 command-aligned gate.** 固定 `offset=0,target=mean`，仅在 calibration 中选择 $h\in\{1,2,3,4\}$ 与 pool。沿用 A2.2 四项门槛。该 gate 重用同一 episode split，定位为开发性安全检查；若 RED，action-verification 分支停止，不得回退使用更好看的 `offset−1`。

**A2.3 result: RED near-miss.** episode winner `h1/8x10/mean/offset0`：$R^2=+0.012$（3/6 dimensions），permutation $p=0.00498$，transition-over-state gain $+0.0381$ [0.0204, 0.0581]；retrieval $0.524$ [0.497, 0.559] 是唯一未过项。门槛不变，结果不得写作 GREEN。

**A2.4 nested cross-fit（预注册补功效）.** 5 个 outer folds 各保留 4 个完整 episode；每个 outer-train 的 16 episodes 再固定分成 12 train / 4 calibration，仅内层选择 horizon、pool、ridge alpha；outer episode 只预测一次。汇总 20 episode OOF prediction，并以同样的 $R^2>0$、permutation $p<0.05$、retrieval cluster-bootstrap q05 $>0.5$、transition-over-state q05 $>0$ 判决。该分析减少单次 4-episode split 方差，但不是新数据 replication；仍需 candidate gate 承接。

**A2.4 result: GREEN.** OOF $R^2=+0.0258$（per-dim +0.110/+0.143/+0.077/−0.058/−0.087/−0.029），direction BA $0.543$，retrieval $0.524$ [0.512, 0.538]，transition-over-state gain $+0.0409$ [0.0222, 0.0611]，permutation $p=0.00498$。5 folds 选择 horizon 3/1/1/2/1，pool `8x10` 为 4/5。该结果建立 command-aligned decoded-motion observability，但尚未证明生成候选上的 reward utility。

**Gate B generated candidates（预注册）.** 冻结 `h1/8x10/offset0`，per-dim weights 为 OOF $[R_d^2]_+$ 归一化；每个 episode 使用对应 outer-fold payload。冻结 base policy，episode-balanced 80 windows，$K=16$。Primary 同时要求：(i) 至少 80% groups 的 reward std $>10^{-4}$；(ii) Spearman$(R^{act},\mathrm{dmotion}_{raw})$ 的 episode-bootstrap q05 $>0$；(iii) matched-command minus cross-episode-shuffled-command Spearman 的 q05 $>0$；(iv) median $|\rho(R^{act},R^{RC})|<0.9$。Raw-GT LPIPS/MSE、RAFT-flow fidelity 和 top-bottom deltas 全量保存为 secondary；RAFT readout 不作 primary，避免与 verifier extractor 循环验证。

**Gate B1 result: RED, positive-direction near-miss.** nondegenerate 1.000，median $|\rho(action,RC)|=0.301$；dmotion rho $+0.064$ [−0.020,+0.141]，matched−shuffled $+0.063$ [−0.037,+0.162]，top−bottom dmotion $+0.018$ [−0.010,+0.047]。Secondary：mean raw-fidelity rho +0.137，RAFT rho +0.179，top−bottom LPIPS +0.00333、MSE +0.000324。B1 不得写作有效。

**Gate B2 stochastic replication（一次性）.** contexts/window seed 固定为 7401，只把 generation seed 改为 17401，仍为 $K=16$。B1/B2 对齐 episode+start 后，先平均每个 context 的两次 rho/delta，再进行 episode bootstrap；要求两次 replication 的 dmotion rho 与 matched−shuffled mean 均为正，且 combined 两项 q05 $>0$。其余 nondegeneracy/RC redundancy 门槛不变。combined RED 时停止 RCAV，不运行第三个 generation seed。

**Gate B2 final result: RED.** Rep2：dmotion rho $+0.057$ [−0.026,+0.135]，matched−shuffled $+0.062$ [−0.040,+0.165]，top−bottom dmotion $+0.007$ [−0.026,+0.039]。两次 replication 均值方向一致；context-average combined 为 dmotion rho $+0.060$ [−0.012,+0.127]、matched−shuffled $+0.063$ [−0.035,+0.153]，median $|\rho(action,RC)|=0.310$。按停止规则终止 RCAV，不进入 GRPO。可报告边界：decoded real transitions 存在可泛化 command signal（A2.4 GREEN），但其在当前 base-policy candidate support 上不足以形成统计可靠的 command-specific ranking。

**后续分支.** 若 FSQ split 均失败，运行 decoded RAFT motion oracle；oracle 正而 FSQ 负才允许升级 spatial CNN/flow-conditioned latent verifier。若信号集中于 $h>1$，实现 action-segment verifier。只有 D1-v2 与候选独立 readout D2 同时绿灯，才在同一 paired sweep 比较 `raw / RC-state / action-only / scalar-sum / RCAV-projection`；不允许用历史 `pixel_tok` 跨 sweep 数字作新方法基线。

### 3.0 C1 decision-level counterfactual validation（预注册，待运行）

排序闭环只能说明 raw 与 RC 的组内偏好不同，不能说明 RC 的偏好更有用。为区分
``decoder-matched verification'' 与任意 target smoothing，在冻结的 multi-step base policy 上进行如下
**不训练**反事实实验：

1. 从与 24 个训练窗口、8 个 held-out evaluation 窗口均不重叠的 episode windows 中采样 $K=16$ group；在每个预先指定的可验证 pivot $h\in\{2,3,4,5\}$ 计算 raw、同 codec RC 与 blur-sham 三种排序。先前 $h=2$、16-window yield pilot 只用于确认冲突覆盖率，未用于任何 outcome 判读或阈值选择。
2. 对每个 pivot 只保留 $j_{\mathrm{raw}}\ne j_{\mathrm{RC}}$ 的 context-horizon。两种 verifier 的 pivot reward 仅用于选择各自的 candidate，永不进入 outcome。
3. 固定各自已选 prefix、后续动作和采样随机数，分别从两个 prefix 续推至 $h+1\ldots7$；对未来真实帧计算 raw-GT LPIPS（primary）、MSE、PSNR、SSIM 和 LPIPS-last。RC 与 raw 用同一组 continuation random seeds。
4. blur-sham 的强度仅在独立 calibration windows 上选择：其 target-to-GT 的 $\mathrm{LPIPS}+\mathrm{MSE}$ 均值匹配 RC 的 codec reconstruction floor，之后冻结。它控制“任意模糊/平滑 ground truth”解释。

统计单位是 context-horizon；每个 episode 内先平均其窗口、重复和 pivot，之后按 episode cluster bootstrap。它不是 $K(K-1)/2$ candidate pairs，也不把 generation repetitions 视为独立样本。预注册 primary 为冲突 contexts 上
$\Delta_{\mathrm{RC-raw}}\mathrm{LPIPS}_{h\ge3}<0$，并要求其 95% cluster-bootstrap CI 不跨零；
secondary 要求 RC 同时优于 blur-sham，且 MSE/SSIM 方向一致。只有 primary 和 codec-specific control 均通过，才进行 5-seed single-step `raw / RC / blur-sham` training gate；否则 C1 限定为诊断结果，不再作为 correction 主张。

这个 protocol 的因果边界是“在世界模型内部，选择一个 candidate prefix 会如何影响同模型的后续 rollout”，并非真实环境干预。它比同帧 raw-GT readout 更独立，但不能替代真实机器人闭环评测。

### 3.1 Multi-step sequence GRPO, fixed seeds 0--9

| comparison | metric | mean delta | wins | paired $t$ | two-sided $p$ | status |
|---|---:|---:|---:|---:|---:|---|
| seq-RC − seq-raw | LPIPS $\downarrow$ | -0.00263 | 6/10 | -0.99 | 0.346 | not confirmed |
| seq-RC − seq-raw | LPIPS-last $\downarrow$ | -0.00454 | 5/10 | -0.84 | 0.420 | not confirmed |
| seq-RC − seq-raw | MSE $\downarrow$ | -0.000412 | 6/10 | -1.26 | 0.239 | not confirmed |

旧 seeds 0--4 的 5/5 方向没有在新增 seeds 5--9 复现（LPIPS 仅 1/5）。因此不能声称 RC 在 multi-step sequence GRPO 中独立改善训练；C1 的有效性必须由 single-step transfer、排序闭环与独立的 decision-level validation 重新界定。

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

## 7. RC-Energy 预注册实验合同

**状态：正式门控 RED，不进入训练。** RC-Energy 不与旧 reward panel 共用准入逻辑。它检验的是 pointwise scoring 与 conditional ensemble scoring 的差异，而不是新增一个图像指标。

### 7.0 Implementation audit（不计作实验结果）

2026-07-13 在 RTX 5090 服务器完成实现审计：RC-Energy 专项 6 tests 与全仓 76 tests 全部通过；真实 CNN-FSQ/Llama/LPIPS-VGG smoke 返回 RGB + 5 个 VGG blocks，四个同几何 reward arms 均为有限且非退化，并与 `dor.reward_spaces.gt_reward` 接线路径逐项一致。`rc_energy_point` 与 `rc_energy` 各完成 1-step backward、checkpoint 保存与 `GRPO_OK`。smoke 的单步指标不作科学结论；正式状态仍由 7.1 门控决定。

### 7.1 双组零训练门控

固定 148 个 calibration windows、$K=16$、base policy 与两个独立 generation seeds。先按 episode 划出 20% 仅估计 RGB/VGG feature-block scales；这些 episode 不进入 gate 统计。其余 episode 对两个方向分别计算：

$$
\Delta\rho^{A\to B}
=\rho(R_{\mathrm{RCE}}^A,U^{A\to B})
-\rho(R_{\mathrm{RC}}^A,U^{A\to B}).
$$

Primary 为 A→B 与 B→A 的 Pearson $\Delta\rho$ episode-bootstrap 90% CI 下界均大于零。Spearman gain 必须方向为正，因为 GRPO 主要消费排序。两个结构负对照为 reversed pairwise sign 和跨 context shuffled pairwise contribution；RC-Energy 对两者的 Pearson gain CI 下界均须大于零。

Non-inferiority boundary：RC-Energy top candidate 相对 pointwise RC 的 raw-GT LPIPS bootstrap q95 不得高于 $+0.002$；MSE relative delta q95 不得高于 $+2\%$。所有条件同时满足才为 GREEN；RED 后不修改 feature blocks、Energy exponent 或 margin，也不进入训练。

### 7.2 GREEN 后的单步 $2\times2$

| target | pointwise（同一 $\Delta$） | Energy（同一 $\Delta$） |
|---|---|---|
| raw GT | `raw_energy_point` | `raw_energy` |
| reachable GT | `rc_energy_point` | `rc_energy` |

`a0faithful` 与 `a0faithful_tok` 同协议重跑，作为原 RLVR 与现有 RC 的外部基线。第一阶段固定 paired seeds 0--2；完整方法必须在 raw-GT LPIPS/MSE 至少 2/3 seeds 不差于 RC，并在 conditional Energy readout 与至少一项 DINO/Inception distributional readout 改善，才扩到 seeds 3--4。否则 RC-Energy 作为被否定的 distributional extension 记录，C1 维持现有 RC。

### 7.3 Formal gate result

正式缓存为 148 windows、两组独立 generation seeds、$K=16$；31 windows 所属 episode 仅估计 block scales，其余 117 windows（16 episodes）进入 episode-cluster bootstrap。RC-Energy 对独立组 raw-target distributional utility 的 Pearson correlation 在 A→B 从 0.272 升至 0.745，在 B→A 从 0.237 升至 0.717；配对增益分别为 +0.4759 [0.3986, 0.5616] 与 +0.4758 [0.3964, 0.5531]。Spearman 增益同样约为 +0.392，且 reversed/shuffled controls 的增益区间均严格为正，说明 group contribution 是可复现的分布信号，不是同组机械相关。

但该信号改变了约一半 top candidate，并稳定损害 raw-GT fidelity：LPIPS mean delta 为 +0.00396/+0.00376，q95 为 +0.00701/+0.00511；MSE relative mean delta 为 +6.26%/+5.96%，q95 为 +10.25%/+8.39%。两方向均违反预注册的 LPIPS $+0.002$ 与 MSE $+2\%$ non-inferiority boundary，因此总体 verdict 为 RED，禁止进入 GRPO。

事后剂量诊断显示缩小 pairwise coefficient 可在保真边界内保留部分相关增益，但该系数不再是 proper Energy Score 推导出的固定 influence，不能作为本门控的补救或正式证据。使用 scale-split median bandwidth 的有界 RBF kernel score 仍产生约 +0.52 Pearson gain，同时复现约 +0.0037 LPIPS 与 +6% MSE 退化。这表明冲突来自 conditional distribution coverage 与单观测 full-reference fidelity 的目标差异，而非 Energy distance 是否有界。该分支作为 negative auxiliary result 保留；后续若继续，只允许预注册的 fidelity-constrained formulation，并须用独立数据重新门控。

## 8. Missing Decisive Experiments

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

#### E2-a. Calibration--Credit Coupling mechanism audit（零训练）

复用冻结 base policy 的 multi-step temporal cache，在相同 candidate rollouts 上分别累积 raw、RC 与 pre-decode diagnostic returns。对每个 return start $t$ 计算：(i) raw/RC return 与 diagnostic return 的组内 rank correlation；(ii) pairwise disagreement；(iii) 累积 residual interaction $\Xi_{i,t}$ 的组内标准差。统计按 episode cluster bootstrap，generation repetitions 与 horizons 不作为独立样本。

机制门要求 aggregate RC-minus-raw $\Delta\rho$ 的 95% CI 下界 $>0$、$\Delta\mathrm{flip}$ 的 95% CI 上界 $<0$，且 earliest-minus-latest residual dispersion 的 95% CI 下界 $>0$。该审计只验证耦合机制，不替代训练结果。

**E2-a result: GREEN.** 冻结 cache 含 3 个 generation repetitions、64 windows、6 个 horizons、$K=16$。从 latest start（1 个 reward）到 earliest start（6 个 rewards），residual dispersion 为 0.00425/0.00777/0.01078/0.01334/0.01545/0.01698，呈单调累积。Aggregate $\Delta\rho=+0.0132$，episode-cluster bootstrap 95% CI $[+0.00847,+0.01756]$；$\Delta\mathrm{flip}=-0.0082$，95% CI $[-0.01044,-0.00581]$；earliest-minus-latest dispersion $+0.01272$，95% CI $[+0.01102,+0.01524]$。三项均通过。该门只建立 calibration--credit coupling 的机制前提，C3 是否成立仍由 E2-b 固定 $n=10$ 训练交互决定。

#### E2-b. Super-additive training interaction

对每个 paired seed 和 lower-is-better metric 定义

$$
I_s=(Y_{\mathrm{return,RC},s}-Y_{\mathrm{seq,RC},s})
-(Y_{\mathrm{return,raw},s}-Y_{\mathrm{seq,raw},s}).
$$

Seeds 0--4 仅为 provisional gate：primary LPIPS 要求 mean $I<0$、至少 4/5 同向、`return-RC` 为四臂最低，才扩到预注册 seeds 0--9。正式 C3 要求 $n=10$ 至少 7/10 的 $I_s<0$、单侧 exact sign-flip $p<0.05$、paired bootstrap 95% CI 上界 $<0$。LPIPS-last/MSE 为支持性指标。机制门或训练门任一失败，C3 不成立；不得将四臂最低均值包装为超加和贡献。

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

## 9. Claim-Evidence Contract

| proposed claim | minimum evidence | current status |
|---|---|---|
| RC reduces rank corruption | same-candidate closure across metrics/repetitions | met |
| RC improves multi-step sequence training | fixed $n=10$ raw-vs-RC | not supported |
| Temporal Return improves multi-step fidelity | fixed $n=10$ paired primary | met |
| RC and Temporal Return are complementary | complete $2\times2$ matrix | missing |
| RC and Temporal Return have a super-additive coupling | mechanism audit + fixed $n=10$ interaction | mechanism met; training interaction missing |
| Encoder reconstruction is a locally justified reachable target | 64/64 存在更优合法邻域 code | rejected；升级为 MRRT |
| Benefit grows with rollout length | $T=4/6/8$ paired experiment | missing |

## 10. Cross-Platform Validation Contract

本节在外部结果产生前冻结。目标是反驳或支持 C1/C2 是否超出 RT-1 的 CNN-FSQ/Llama
实现，而不是补一张泛化表。外部数据不得用于重新选择 RC 的双指标、Temporal Return 形式、
$\gamma$、学习率、训练步数或统计阈值。

### 10.1 VP2-RoboSuite / iVideoGPT：主外部验证

**固定资产。** iVideoGPT commit `d601d5cac9e96c6aa0c17cb37ed6a7c7ca1fb210`（MIT），
动作条件 checkpoint `thuml/ivideogpt-vp2-robosuite-64-act-cond`，以及 VP2 RoboSuite
`5k_slice_rendered_256.hdf5`。服务器资产位于 `/root/autodl-tmp/external_wm/`。P0 必须写出
checkpoint SHA-256、HDF5 schema、episode ID 和 train/calibration/test manifest；test episode
不得参与 tokenizer/metric scale 选择或训练。

实际使用的 transformer `model.safetensors` SHA-256 为
`b46283e08cec6adb88a2d8c9496b39b3501e55262a2401618d6fcd9c11115261`。

**语义保持。** 给定同一 context-action 条件，冻结上游 tokenizer 编码真实 future frame，候选
均由上游 action-conditioned categorical sampler 产生。raw verifier 为 decoded candidate 对 raw
GT 的 $-(\mathrm{MSE}+\mathrm{LPIPS})$；RC 仅把 target 换为 $D(E(s'))$。所有报告指标仍对
raw GT 计算 LPIPS（primary）、MSE、PSNR、SSIM 与 LPIPS-last。Temporal Return 对 future-frame
block 计算；每个 sampled token 的 policy log-prob 由 iVideoGPT action model 的 teacher-forced
logits 求得，禁止引入 decode 后像素梯度或额外 reward head。

**P0，接线审计，不训练。** 固定 8 个 training 与 8 个 held-out contexts、$K=4$，验证：

1. context/action/frame 的方向、范围、分辨率、token layout 和 decode shape；
2. 在 $D(E(s'))$ 与 raw GT 人为设为相同的合成输入上，raw/RC 排序严格一致；
3. sampled token 的 teacher-forced log-prob 与上游 `generate` vocabulary 一致，且梯度只穿过
   iVideoGPT Transformer；
4. 保存一条 rollout 的 frame、tokens、rewards 和 log-prob trace。

**P1，C1 零训练闭环。** 固定 held-out episode contexts、$K=16$、两个 generation seeds，
比较 raw/RC 对**同一候选**的 group rank correlation、pairwise flip 与可达残差强度。统计单位
是 episode，使用 episode-cluster bootstrap；candidate pairs、horizons 和 generation repetitions
均不作为独立样本。仅当 RC-minus-raw 的 $\Delta\rho$ 95% CI 下界 $>0$，且
$\Delta\mathrm{flip}$ 95% CI 上界 $<0$，才进入 P2。

**P2，C1/C2 四臂训练。** 固定 `raw-sequence`、`RC-sequence`、`raw-return`、`RC-return`。
先跑 paired seeds 0--2；primary 是 held-out raw-GT full-rollout LPIPS，MSE/PSNR/SSIM 与
LPIPS-last 均全量报告。只有 `RC-return` 在 primary 上至少 2/3 不劣于每个其余臂、且 secondary
未出现系统性反向趋势，才用**同一超参**扩至 seeds 0--4。最终统计只基于全部完成 seeds 的配对
效应和 CI；禁止挑选最优 seed。

P2 分别报告 RC 对 sequence credit 的 C1 transfer、return 对 raw verifier 的 C2 transfer，以及
预先定义的 calibration--credit interaction。完整臂的最低均值不等同于超加和；交互 CI 不支持时
只能报告完整组合。

#### 10.1.1 Formal VP2 result

P0 接线、teacher-forced log-prob、raw/RC identity anchor、固定 context schedule 与 train/eval episode
disjointness 均通过。初版 P1 使用 token Hamming 作为 decoder-free reference；code review 后判定
8192-way VQ token ID 为类别编号，Hamming 将所有 code substitution 当成等距，不能作为主几何。
正式复算改用与公开 `detokenize` 完全一致的 dynamics codebook lookup + `post_quant_linear` 输出，
即 conditional decoder 实际消费的连续 latent；Hamming 与未投影 codebook RMS 全量保留为敏感性。

冻结候选 cache 的 10,000 次 episode-cluster bootstrap（95% CI）结果如下：

| VP2 protocol | $\Delta\rho_s$ ↑ | $\Delta$ flip ↓ | RC-top latent gain ↑ | verdict |
|---|---:|---:|---:|---|
| H2, 64 contexts, $K=16$, 2 draws | +0.0932 [0.0674, 0.1217] | -0.0409 [-0.0524, -0.0306] | +0.00203 [0.00115, 0.00300] | GREEN |
| H8, 8 contexts, $K=16$, 2 draws | +0.0753 [0.0408, 0.1104] | -0.0346 [-0.0488, -0.0207] | +0.00142 [0.00013, 0.00238] | GREEN |

P2 在冻结 16-episode raw-GT evaluator、固定 context schedule、paired seeds 0--2、20 steps 下完成。
最终均值中 raw-sequence LPIPS 为 0.03031，RC-sequence 为 0.03052，raw-return 为 0.03059，
RC-return 为 0.03066；raw-sequence 最低。RC-sequence 相对 raw-sequence 的 LPIPS delta 为
+0.00022，仅 2/3 seeds 改善；MSE/LPIPS-last 同样不形成稳定外部收益。补测的 token readout 在
step 10 显示 RC 使 exact token mismatch 3/3 seeds 改善（mean -0.00285，paired $t=-4.63$），但
MSE/SSIM 3/3 反向，step 20 后 token 优势也不稳定。结论是 C1 的 rank repair 与短程 token update
可迁移，但不等价于 VP2 raw-pixel evaluator 改善。

针对失败进行的两个适配只作诊断，不升级为方法：(i) H8 raw/RC reward 的 horizon std span 为
7.41/6.85，先按 horizon 等化尺度可离线提高 latent-return rank agreement，但 1-seed paired
10-step 训练中 LPIPS/MSE/LPIPS-last 均差于 plain return；(ii) raw-top regret 约束选择的
$R_{0.25}=0.75R_{raw}+0.25R_{RC}$ 亦未胜过 raw-sequence。两者均按 admission rule 终止，未扩种子。

### 10.2 IRIS Atari：隔离可迁移性探针

固定 IRIS commit `24326aaaa283c527f42b89b44cfdecf2665a7a16`（GPL-3.0）与 Breakout/Pong
checkpoint。先做 replay audit，登记合法 Atari ROM provenance、Gym/ALE 版本、action repeat、
frame preprocessing、episode seeds、checkpoint SHA-256，并验证同一 action sequence 可以重放
一致的 GT trajectory。audit 失败立即终止。

Breakout/Pong checkpoint SHA-256 分别为
`169e3669b7d1901990b4ebce35ff8f3355610fae393e6a587fd62c40faf790f4` 与
`b180b9145b086fb00a7bcb9918da9ee373bde7dd2e0d04d6a7054aa1d10284be`。

IRIS 上游源码与 checkpoint 保持在 ignored 的 `third_party/iris/` / 外部资产目录；tracked adapter
不复制上游实现，只在独立实验入口动态加载并通过保存的 `npz` rollout 与 DoR 交换。P0 后只运行
与 VP2 P1 同构的 single-step raw/RC rank closure。
通过后才允许 10.2.2 中冻结的 paired 5-seed training conversion；IRIS 不承担 C2 的正式显著性
主张，除非 temporal token-block objective 完成独立 code review 并重新登记。

#### 10.2.1 Formal IRIS result

环境固定为 ALE 0.9.0，`repeat_action_probability=0`，action repeat 4；每个原始帧先 resize 到
64x64，再对最后两帧 max-pool，与 IRIS wrapper 顺序一致。由于单帧无法确定 Atari 物体速度，
每个预测保留 4-frame token/action history，prompt 为
$[z(s_{t-3}),a_{t-3},\ldots,z(s_t),a_t]$。Breakout/Pong 各使用 16 个 episode clusters、
128 个 real-transition windows、$K=16$、两个 generation draws。主参照是 tokenizer embedding
经 `post_quant_conv` 后的 decoder-input latent RMS；token Hamming 仅作敏感性。

| IRIS game | $\Delta\rho_s$ ↑ | $\Delta$ flip ↓ | RC-top latent gain ↑ | verdict |
|---|---:|---:|---:|---|
| Breakout | +0.2952 [0.2188, 0.3751] | -0.1336 [-0.1697, -0.0995] | +0.00438 [0.00307, 0.00574] | GREEN |
| Pong | +0.4275 [0.2899, 0.5676] | -0.2182 [-0.2899, -0.1477] | +0.00142 [0.00109, 0.00173] | GREEN |

区间为冻结 cache 上 10,000 次 episode-cluster bootstrap 95% CI。两个游戏的 mechanism 与 selection
判据均通过，支持 C1 超出机器人域、FSQ tokenizer 与 Llama/iVideoGPT 架构。IRIS 不用于选择
RC 指标或 C2 参数，也不承担训练收益和 Temporal Return 主张。

**窗口重叠敏感性。** 主协议冻结后，保持 episode 数、每游戏 128 windows、$K$ 与 generation draws
不变，仅将同一 episode 的 window stride 从 1 增至 4。Breakout 的 $\Delta\rho_s=+0.2230$
[0.1543, 0.3005]、$\Delta$flip$=-0.0938$ [-0.1247, -0.0650]、RC-top latent gain
$=+0.00309$ [0.00201, 0.00438]；Pong 分别为 $+0.3989$ [0.2914, 0.5050]、
$-0.1756$ [-0.2227, -0.1287]、$+0.00141$ [0.00118, 0.00163]。区间仍为 10,000 次
episode-cluster bootstrap 95% CI。该分析排除结果仅由连续窗口高度重叠驱动，但属于主分析后的
稳健性复核，不改变预注册 primary。

#### 10.2.2 IRIS C1 training-conversion protocol（冻结，待运行）

目的不是在 Atari 上测试 C2，而是检验 10.2.1 的强 rank-calibration signal 能否转化为一次真实
world-model policy update 后的 held-out raw-GT fidelity。Breakout 与 Pong 分别使用 stride-4 cache，
固定 16 个 episode 中 12 个训练、4 个评测；划分由固定 `split_seed` 产生并写入每个 JSON。两个 arm
仅改变 verifier target：

$$
R_i^{\mathrm{raw}}=-[\mathrm{LPIPS}(x_i,s')+\mathrm{MSE}(x_i,s')],\qquad
R_i^{\mathrm{RC}}=-[\mathrm{LPIPS}(x_i,D(E(s')))+\mathrm{MSE}(x_i,D(E(s')))].
$$

其余保持一致：同一公开 checkpoint、冻结 tokenizer、$K=16$、20 steps、batch windows 2、
AdamW（lr $3\times10^{-6}$，weight decay 0.01）、gradient clip 1.0、`low_var_kl` coefficient
0.001、固定 context schedule、paired policy seeds 0--4。IRIS checkpoint 含
0.1 dropout，因此 generation 与 teacher-forced candidate log-prob 均固定在 `eval()` mode；
`eval()` 不关闭梯度。评测始终对 raw GT，固定 eval candidate seeds，并同时记录 LPIPS、MSE、
PSNR、SSIM、exact-token mismatch 与 post-quant latent RMS。

Primary 为 final step 20 的 paired RC-minus-raw LPIPS。单游戏通过要求 mean $<0$ 且至少 4/5 seeds
改善；跨游戏汇总额外报告 game 内配对效应与 game×seed cluster bootstrap 95% CI。MSE/SSIM、
token mismatch 与 latent RMS 是支持性 readout，不能替代 primary。禁止 best-step、best-seed、
按游戏改学习率或根据测试集重选超参。若两个游戏均不通过，IRIS 仍只承担 10.2.1 的机制复现，
不得写成训练收益。

**正式结果：primary RED，token/latent conversion GREEN。** 两游戏均完成 raw/RC × seeds 0--4
共 20 runs；每个 run 的 step 为 0/10/20，raw/RC step-0 全指标逐位一致，context schedule hash
一致，数值均有限。KV-cache generation 与 teacher-forced log-prob 的最大误差为
$8.6\times10^{-6}<10^{-4}$。

| game | final raw LPIPS | final RC LPIPS | RC−raw | LPIPS wins | token mismatch RC−raw | latent RMS RC−raw |
|---|---:|---:|---:|---:|---:|---:|
| Breakout | 0.00245281 | 0.00244466 | −0.00000815 | 3/5 | −0.001025 (5/5) | −0.0000692 (5/5) |
| Pong | 0.00007035 | 0.00007084 | +0.00000049 | 1/5 | −0.002759 (5/5) | −0.0001403 (5/5) |

Breakout 的 LPIPS paired $t=-0.80$；Pong 为 $t=+0.83$，均未通过预注册的 4/5 primary。
Breakout MSE/PSNR/SSIM 为 1/5、2/5、1/5 改善；Pong 为 1/5、0/5、1/5，不能用
token readout 替代 raw-GT fidelity。两个游戏中 raw 与 RC arm 相对 step-0 checkpoint 大多都退化，
因此结果不是“raw verifier 有效、RC verifier 无效”，而是当前训练分布没有建立 raw-pixel learning
gain。Breakout step 10 曾有 LPIPS 4/5 的 RC−raw 改善，但 final-step 合同固定为 step 20，不能事后
改用 early checkpoint。

该 cache 的 collection policy 是 FIRE 后均匀随机动作；公开 IRIS checkpoint 同时包含 actor-critic，
上游测试协议使用其 policy sampling（temperature 0.5）。因此当前 P2 被重新解释为冻结且完整报告的
**random-policy stress test**：它证明 RC update 稳定改善 token/decoder-latent readout，却未转化为
OOD random-action raw pixels。允许的下一步仅为新增、独立登记的 actor-policy-matched replication；
不得覆盖本结果、沿用 test episodes 调参或把 matched replication 当作同一实验的补跑。

#### 10.2.3 Actor-policy-matched independent replication（冻结，待运行）

该实验只修复 10.2.2 已识别的 state-action distribution mismatch，不改变 RC、GRPO、优化器、
训练步数、指标或判据。直接从同一公开 checkpoint 加载 actor-critic；actor 输入遵循上游
`Agent.act`，先经冻结 tokenizer encode-decode，LSTM state 在真实 ALE episode 内连续维护，动作按
temperature 0.5 采样。环境保持 ALE 0.9.0、`repeat_action_probability=0`、action repeat 4、
resize-before-max-pool；reset 使用上游 test 的单次 no-op，不再手工 FIRE 或均匀随机动作。

新建独立 16-episode、8-windows/episode、stride-4 manifest，文件与 episode seed 不复用 10.2.2。
训练前必须依次通过：

1. **provenance/actor audit**：checkpoint actor head 与环境 action vocabulary 一致；保存 policy
   entropy、action histogram、episode return、frame-delta MSE 与 manifest hash；
2. **headroom audit**：actor-policy windows 的 frame-delta 与 base raw-GT LPIPS 均非退化，并与
   random-policy cache 并列报告，不要求事后设定有利阈值；
3. **rank gate**：完全复用 10.2.1 的 $K=16$、2 draws、post-quant reference 和 10,000 次
   episode-cluster bootstrap；$\Delta\rho$ 95% CI 下界 $>0$、$\Delta$flip 上界 $<0$、RC-top
   latent gain 下界 $>0$ 才允许训练。

rank gate GREEN 后，原样复用 10.2.2 的 raw/RC × seeds 0--4、20 steps、lr $3\times10^{-6}$、
KL 0.001 与 final-step primary；split/data/eval seeds 保持 9413/9414/9415。成功仍要求单游戏
LPIPS mean RC−raw $<0$ 且至少 4/5 seeds 改善。该 replication 与 random-policy stress test 并列表，
不得合并挑 seed；若仍 RED，则 C1 的跨平台主张永久收缩为 rank/representation calibration，
训练收益只在 RT-1 成立。

### 10.3 RA-RC 跨平台训练准入（预注册）

**目标。** 检验同一个 raw-anchored constrained update 能否把三平台都已观察到的 reachable-space rank repair 转化为 raw-GT fidelity，而不是为每个平台事后选择不同 reward 或指标。方法公式固定为 `method.md` §3.4；RT-1、VP2 与 IRIS 不得使用不同投影阈值或融合系数。

**共同 arms。** 第一阶段固定为 `raw / RC / RA-RC`，candidate sampling、context schedule、optimizer、KL、gradient clipping、训练步数与 evaluator 完全配对。`raw` 与 `RC` 是必要端点；没有这两个端点，无法判断 RA-RC 是保留 raw 还是仅复现 pure RC。多步第二阶段才加入 `RA-RC + Temporal Return`，不得用 C2 掩盖 C1 conversion 失败。

**共同 readouts。** 三平台 primary 均为 held-out raw-GT LPIPS，supporting fidelity 为 raw-GT MSE、PSNR 与 SSIM；共同 codec readout 为 decoder-input post-quant latent RMS；Spearman rank agreement 与 pairwise flip 只作 mechanism readout。token Hamming、flow、LPIPS-last 和分布指标属于平台特定 secondary，不能替代 primary。禁止在 RT-1 报 LPIPS、VP2 报 latent RMS、IRIS 报 token mismatch 后合并声称 universal improvement。

**Stage A：接线与梯度几何冒烟。** 每个平台至少两个 candidate groups，检查：(i) raw/RC 使用同一 sampled candidates 与 token log-prob；(ii) $\lambda^*$ 有限且非负；(iii) 投影后 $\langle g^*,g_{\mathrm{raw}}\rangle/\|g_{\mathrm{raw}}\|^2\geq1-10^{-6}$；(iv) KL 只加入一次；(v) `raw`/`RC` 旧路径数值不变；(vi) progress 日志包含约束触发率、gradient cosine、raw-progress ratio 与 ETA。任一失败均停止。

**Stage B：三 seed 小代价 pilot。** RT-1 single-step、VP2 H8 sequence credit、IRIS actor-policy-matched Breakout/Pong 分别运行 `raw / RC / RA-RC` × seeds 0--2。准入 full run 要求每个平台的 RA-RC-minus-raw final LPIPS mean $<0$ 且至少 2/3 seeds 改善，post-quant latent RMS mean 同向改善；MSE 相对均值退化不得超过 5%，SSIM 绝对均值退化不得超过 0.01；同时 RA-RC 的 raw LPIPS 不得差于 pure RC。pilot 仅作停跑门，不进入 headline table。

**RT-1 Stage-B 结果：provisional GREEN（2026-07-16）。** 固定 final step、seeds 0--2 的完整
`raw / RC / RA-RC` 九臂已完成。RA-RC 相对 raw 的 LPIPS 为
$0.14134$ vs. $0.14250$（$\Delta=-0.00117$，相对 $-0.82\%$，3/3 seeds），MSE
$\Delta=-6.00\times10^{-5}$（$-1.65\%$，3/3），PSNR $\Delta=+0.0599$（3/3），
SSIM $\Delta=+0.00073$（2/3），post-quant latent RMS $\Delta=-0.00739$（$-2.25\%$，
3/3）。相对 pure RC，RA-RC 的 LPIPS 均值再降低 $0.00060$（2/3），但三 seed bootstrap
区间跨零，不能声明优于 RC。约束并非退化分支：三 seed 的 mean constraint-active rate 为
$0.917$--$0.950$；投影前 RC 方向的 raw-progress ratio 为 $0.764$--$0.818$，投影后为
$1.001$--$1.002$。该结果只通过 RT-1 pilot 停跑门，不进入 headline，不触发 RT-1 单独扩种；
下一判决为 VP2 与 actor-policy-matched IRIS 的同协议 Stage B。

**Stage C：正式配对实验。** 通过 Stage B 后，固定协议扩至 10 seeds。universal claim 要求每个平台分别满足：RA-RC-minus-raw LPIPS mean $<0$、至少 8/10 seeds 同向、episode/seed-cluster paired bootstrap 95% CI 上界 $<0$；post-quant latent RMS 同向且 95% CI 上界 $<0$；MSE/SSIM 不得出现显著退化。不得先合并三个平台再检验，因为 pooled mean 可掩盖单平台失败。正式统计同时报告 RA-RC vs raw、RA-RC vs pure RC；headline 检验固定为 RA-RC vs raw，后者用于定位约束相对 pure RC 的代价，不以两者择优。

**Stage D：长序列扩展。** 只有 C1 的 Stage C 至少在 RT-1 与 VP2 通过后，才在 multi-step RT-1/VP2 使用相同 RA-RC 投影比较 sequence 与 Temporal Return。此时 raw/RC surrogate 均先按同一 temporal-credit rule 构造，再进行每组参数梯度投影；不允许 raw 使用 sequence、RC 使用 return。IRIS single-step 只承担 C1，不强行承担 C2。

**判决边界。** 若 Stage B/C 失败，论文仍可报告跨 codec 的 rank-disagreement diagnosis，但不得把 RA-RC 写成普适训练方法，也不得通过 best step、best seed、平台特定 primary 或重新定义 composite score 补救。

### 10.4 失败解释与报告纪律

P0 失败表示接线或环境不可复核，不报告效能；P1 失败表示该 codec/task 上没有可测
rank-calibration signal，C1 的适用范围必须收缩；P2 失败表示排序修复未转化为该平台的训练收益，
禁止用视觉案例、best seed 或不匹配 evaluator 替代 primary。任何外部表都列出全部预注册 arms、
全部完成 seeds、raw evaluator 和上游 checkpoint。
