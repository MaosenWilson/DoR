# RC-GRPO：AAAI-2027 统一故事与证据合同

> 本文档是论文叙事、贡献、术语和证据状态的唯一事实源。Method 只写这里定义的方法，Experiments 只验证这里登记的主张。历史方案和负结果不得反向进入主线。

## 1. 一句话论点

在 tokenized 视频世界模型的 RLVR 后训练中，verifiable reward 并不天然等于可靠学习信号：冻结 tokenizer-decoder 使 raw ground truth 超出策略可达输出空间，多步 rollout 又把一个 sequence-level score 广播给承担不同未来影响的 frame-token blocks。我们先定位 raw/reconstructed target 之间改变候选排序的残差交互，再检验 encoder reconstruction 是否为局部合理的可达目标，最后用 future-frame return 分配 block-level credit。

英文工作句：

> Verifiable rewards are not automatically reliable learning signals for tokenized video world models. We align verification with the decoder-reachable output space and assign future-frame credit according to the reliability of group-relative rankings across rollout horizons.

## 2. 为什么这项工作有意义

RLVR-World 证明了视频世界模型可以把 decoded prediction 与真实帧之间的 MSE/LPIPS 直接作为 verifiable reward，并使用 GRPO 后训练。但视频分支与语言 RLVR 有两个结构差异：

1. 策略并不直接输出像素，而是输出冻结 tokenizer 的离散 token，最终预测被限制在 decoder 的可达集合中；
2. 多步世界模型不是生成一个不可分的答案，而是依次生成会互相影响的 future-frame token blocks。

因此，视频 RLVR 的核心问题不是“还能加几个图像指标”，而是：

> GRPO 用来比较候选并分配梯度的排序，在策略可达空间中是否正确，在不同时间位置上是否同样可信？

如果这两个问题不解决，reward 数值可计算、可复现，仍可能把错误的相对偏好广播给大量 token。该问题直接影响 tokenized video prediction、world-model post-training，以及任何通过冻结 codec 后再计算 full-reference reward 的生成模型。

## 3. 现有管线与两个缺口

给定 context-action 条件 $q=(s_{1:C},a_{1:H})$，策略采样 $G$ 条 future-token trajectories $o_i=(o_{i,1},\ldots,o_{i,H})$，decoder 给出 $\hat s_{i,h}=D(o_{i,h})$。RLVR-World 风格的逐帧 reward 为

$$
R^{\mathrm{raw}}_{i,h}
=-
\left[
\operatorname{MSE}(\hat s_{i,h},s_h')
+\operatorname{LPIPS}(\hat s_{i,h},s_h')
\right].
$$

### 3.1 输出空间缺口：raw target 不可达

真实帧 $s_h'$ 通常不严格属于冻结 decoder 的像素输出集合。即使策略预测了正确 token $E(s_h')$，它也只能得到

$$
\tilde s_h'=D(E(s_h'))\neq s_h'.
$$

raw target 与 reachable target 的残差包含 group-constant 部分和 candidate-dependent interaction。前者会被 GRPO 去均值消除，后者可能改变候选顺序：

$$
R^{\mathrm{raw}}_{i,h}=R^*_{i,h}+\eta_{i,h},
\qquad
\eta_{i,h}-\eta_{j,h}\neq 0.
$$

GRPO 只消费组内相对优势，因此真正的故障不是 floor 均值，而是 pairwise rank corruption。

### 3.2 时间信用缺口：sequence scalar 被广播给所有 token

sequence-level GRPO 先把整段 rollout 的逐帧 reward 压成一个标量，再给所有 future tokens 同一个优势。这无法区分某个早期 frame block 对后续误差传播的责任，也无法表达不同 horizon reward 的可靠性差异。

普通逐帧归一化也不够：它只奖励当前帧，切断了早期 token 对后续帧的延迟影响。现有 5-seed frame-only 对照显著差于 temporal return，说明有效结构不是“把 reward 切细”，而是“把未来质量回传给能够影响它的前序 block”。

## 4. 统一原则：可靠排序驱动的时空校准

论文不再组织成一个 reward trick 加一个 GRPO trick，而是沿一个原则展开：

$$
\boxed{
\text{measure rank reliability}
\rightarrow
\text{align the verifier target}
\rightarrow
\text{assign temporally reliable credit}
}
$$

## 5. 方法

### 5.1 Reconstruction-Calibrated Verifier

将 raw GT 投影到与策略输出相同的冻结 codec 可达空间：

$$
\tilde s_h'=D(E(s_h')).
$$

正式 verifier 保持最小双指标形式：

$$
R^{\mathrm{RC}}_{i,h}
=-
\left[
\operatorname{MSE}(\hat s_{i,h},\tilde s_h')
+\operatorname{LPIPS}(\hat s_{i,h},\tilde s_h')
\right].
$$

RC 不改变候选、policy、decoder 或最终 evaluator；所有 headline metrics 仍对 raw real frame $s_h'$ 计算。因此它不是把任务换简单，而是只校准训练 verifier 的比较空间。

### 5.2 Temporal-Return GRPO

对 frame block $t$，累计所有可能被它影响的 future-frame rewards：

$$
G_{i,t}^{\mathrm{TR}}
=
\sum_{h=t}^{H}
\gamma^{h-t}R^{\mathrm{RC}}_{i,h}.
$$

每个时间位置在同一 candidate group 内标准化：

$$
A_{i,t}^{\mathrm{TR}}
=
\frac{G_{i,t}^{\mathrm{TR}}-mu_t}
{\sigma_t+\epsilon}.
$$

该优势只广播给第 $t$ 个 future-frame token block。KL anchor、采样器、world model 与 tokenizer 均保持原协议。

### 5.3 已否定扩展：Rank-Reliable Temporal Return

**状态：正式离线门控 RED，不进入训练和论文方法。** horizon reliability 虽非平坦，但正确权重相对 plain return 的 episode-bootstrap CI 跨零且 evaluator correlation 没有改善；reversed/shuffled 为负只证明门控有方向判别力，不能把 primary 的不显著正数包装成方法收益。

Temporal Return 默认所有 horizon reward 同样可靠。为把 C1 的 rank diagnosis 与 C2 的 credit assignment 真正统一，在 calibration candidates 上使用 pre-decode token distance 作为诊断参照，估计每个 horizon 的相关性 $\rho_h$ 与翻转率：

$$
\hat p_{\mathrm{flip},h}
=
\frac{\arccos(\hat\rho_h)}{\pi}.
$$

定义 horizon reliability：

$$
w_h
=
\operatorname{clip}
\left(1-2\hat p_{\mathrm{flip},h},0,1\right),
\qquad
\bar w_h=\frac{w_h}{H^{-1}\sum_u w_u}.
$$

候选的 reliability-shaped return 为

$$
G_{i,t}^{\mathrm{RR}}
=
\sum_{h=t}^{H}
\gamma^{h-t}\bar w_h R^{\mathrm{RC}}_{i,h}.
$$

$w_h$ 只由独立 calibration split 冻结，不由训练结果选择，也不对当前候选自适应。该形式有清楚的 pairwise sign-reliability 解释，但不是无偏策略梯度校正定理；它改变了 temporal objective，必须通过重放门控和 paired training 验证。

若 $w_h$ 近似常数，GRPO 标准化会使该方法近似退化为 plain Temporal Return，分支应立即终止。

## 6. 贡献结构

### 6.1 当前已成立的两项贡献

**C1. Codec-Conditioned Rank Calibration.** 发现 decoded verifier 的可达性失配，并建立三段不可拆的校准框架：
(a) **精确残差分解**（method.md §2.2）：对加权平方特征距离有恒等式 $R^{\mathrm{raw}}_i=R^{\mathrm{RC}}_i+2\langle u_i-\tilde v,e\rangle_W-\|e\|_W^2$——常数项被 GRPO 归一化精确消除，改变排序的只有候选相关交互项 $b_{ij}=2\langle u_i-u_j,e\rangle_W$。该恒等式定位差异来源，但不预设 $b_{ij}$ 全是噪声；local metric-projection gate 决定 encoder reconstruction 能否作为正式校准目标。
(b) **训练前 Reachability Audit**（§3.2）：RIR 度量交互严重度，回答"当前 codec 是否需要校准"；诚实边界：只作 severity stratification，不作在线 gate 或 context 权重。
(c) **跨 codec、跨 horizon 闭环验证**（§3.3，episode-cluster bootstrap）：rank disagreement 在 single-step CNN-FSQ（$\Delta\rho$ +0.0168，p=0.0075）与 multi-step compressive FSQ（$\Delta\rho$ +0.0145，p<5e-4；h2–7 逐层显著）同时降低；$\arccos(\rho)/\pi$ 仅作为已有统计关系的预测校验，不作为新理论；修复量随 RIR 四分位单调增强（Spearman +0.120 [+0.027,+0.202]，p=0.007）。downstream 由 paired training 承接（single-step 5-seed；multi-step 独立 RC 增益尚未成立）。
(d) **明确的 target-refinement 边界**：固定 8 个高残差 latent cells、2 轮合法 FSQ 相邻移动的 64-window gate 中，MRRT 在 64/64 窗口改善其优化目标（mean relative gain 0.834%），但在 `raw / encoder-RC / MRRT / matched-random legal move` 的三 seed 下游训练中未胜过 encoder-RC，也未胜过 matched-random control。因此 MRRT 不构成方法或贡献；它只否定“encoder reconstruction 是度量最优投影”的过强说法。配套负空间（进边界不进贡献）：围绕校准 verifier 的 reward 构造五层系统证伪（等权融合/学习权重/面板扩展/在线约束/空间池化）——最小校准双指标 verifier 在此接口信息饱和。

**C2. Frame-block temporal-return credit assignment.** 将 multi-step video GRPO 的信用单位从整条 sequence 改为 future-frame token blocks，通过 reward-to-go 把后续预测质量分配给能够影响它的前序 block；截断 return 与 candidate-shuffled return 在保持同一 reward 边际分布时破坏 future-credit correspondence，用于区分真正的时间信用与一般重标度。正式训练必须使用与 RLVR-World/VERL 一致的 `low_var_kl`，旧线性 KL 结果只作探索性证据。

### 6.2 当前强化方向

不再追求第三个 reward/GRPO 小变体或更强 target 搜索。MRRT 的 downstream gate 已判负；C1 的唯一有价值强化是检验 raw/RC 排序冲突时，RC 所选首帧是否带来更好的、未参与排序的未来续推保真。C2 的强化门是 temporal alignment：用官方 `low_var_kl`、完整 $2\times2$、截断 return 与 candidate-shuffled control 证明收益来自正确的 future-credit correspondence，而非 reward rescaling。

## 7. 当前证据

| Claim | Evidence | Status |
|---|---|---|
| raw verifier 存在 candidate-dependent rank corruption | 5 generation repetitions、LPIPS/MSE/联合空间的 $\rho$/flip 闭环，理论误差 <0.005 | supported |
| RC 降低同候选排序分歧 | 三个 reward spaces 均 5/5 repetitions 降低 flip | supported |
| RC 改善多步 sequence GRPO | 固定 $n=10$ 仅 LPIPS 6/10，$p=0.346$；新增 seeds 仅 1/5 | not supported as an independent multi-step gain |
| Temporal Return 优于 seq-RC | 旧线性 KL 下固定 $n=10$ LPIPS/LPIPS-last 均 9/10；需以官方 `low_var_kl` 同协议复核 | provisional |
| return 结构必要 | frame-only 比 full return 差 LPIPS +0.0105，0/5，$t=+6.2$ | supported |
| Temporal Return 改善运动/分布真实性 | dmotion 不支持，DINO-KID 变差 | rejected；不得主张 |
| Rank-Reliable Return 有额外收益 | replay CI 跨零、$\Delta\rho\le0$ | rejected |

## 8. 必须补齐的实验

### P0：C1 decision-level validation

MRRT 已完成并判负，不扩种子、不增加搜索预算。下一实验只在 raw/RC 首帧排序冲突的候选组中，用相同动作与 common-random-number continuation 续推 $h=2\ldots H$；首帧 raw/RC reward 均不得进入 outcome。以未来 raw-GT LPIPS/MSE/SSIM、以及 candidate-pair win rate 为独立 outcome，检验 RC 选择是否更常导向较好的 future rollout。该实验必须包含 matched-random reachable target 与 episode-cluster bootstrap；若 RC 不胜，则 C1 收缩为诊断性发现，不再主张其作为学习修复。

### P1：完整 $2\times2$ 因子对照

现有矩阵缺少 `raw verifier + temporal return`。正式比较使用 seeds 0--9：

| Verifier | Sequence advantage | Temporal return |
|---|---|---|
| raw | 已有 | **缺失** |
| RC | 已有 | 已有 |

该矩阵回答：Temporal Return 是否独立有效，RC 是否是 temporal credit 的必要 substrate，以及两者是否存在交互。没有它，不能写“两个模块互补”或“统一系统”。

### P2：Temporal correspondence controls

比较 frame-only、$L\in\{1,3,\mathrm{full}\}$ 截断 returns 与 candidate-shuffled return；shuffled control 在每个 horizon 内独立置换候选身份，严格保持该 horizon 的 reward multiset 与归一化方式。只有 aligned/full return 随 credit horizon 增强且胜过 shuffled control，才能把收益归因于时间信用而非重标度。

### P3：长度泛化

若固定 $n=10$ 支持 Temporal Return，再用 paired 3 seeds 比较 $T\in\{4,6,8\}$ 的 seq-RC/return-RC。目标不是挑最佳 $T$，而是检验 temporal credit 的相对收益是否随 rollout horizon 增长。

## 9. 负结果如何进入论文

负结果只服务边界，不作为独立贡献：

- 多指标静态融合、learned fusion、pre-decode code/gradient：未超过最小 RC verifier；
- adaptive Rank-Guard：离线可预测但训练全线变差；
- local-floor spatial pooling：预注册主配置 RED；只否定该权重族，不关闭空间轴；
- Dr.GRPO、REAL-style VPO、GSPO、segmental variants：未超过 vanilla GRPO；
- gain shaping 与 horizon-aware KL：未超过 plain Temporal Return；
- distributional metrics：full-reference RLVR 的 fidelity gain 伴随 DINO distributional cost。

正文只保留与主张直接相关的三类负证：minimal verifier saturation、frame-only temporal ablation、distributional boundary。其余进入 appendix 或不写。

## 10. 安全主张与禁区

### 可以写

- Tokenizer target mismatch changes group-relative rankings through candidate-dependent residual interactions; cross-codec audits identify the subset consistent with corruption.
- Replacing raw GT with the encoder-induced reconstruction aligns both sides of the verifier to decoder outputs, but is not called a metric projection until the local-reachability gate passes.
- Frame-block temporal returns improve held-out full-reference LPIPS under the fixed RT-1 protocol at fixed $n=10$.
- Extra reward components and generic optimizer substitutions did not improve this setting.

### 不能写

- reconstruction floor 的绝对均值直接导致 GRPO 失败；
- RC 对所有 tokenizer、数据集和 metric 普遍更好；
- Temporal Return 是通用或原创的 reward-to-go 理论；
- 方法改善 motion、diversity 或 distributional realism；
- 负结果证明整个 reward/空间/GRPO 设计空间已经关闭；
- Rank-Reliable Return 在门控和训练前已经是贡献。

## 11. 标题策略

当前安全标题：

> **RC-GRPO: Reconstruction-Calibrated Temporal Credit Assignment for Tokenized Video World Models**

只有 C3 验证成功后才考虑：

> **RC-GRPO: Rank-Reliable Temporal Credit for Tokenized Video World Models**

后者更统一，但在 C3 未成立时不可使用。

## 12. 文档规则

- `story.md`：唯一论点与证据状态；
- `method.md`：可复现公式，pending 方法必须显式标记；
- `experiments.md`：结果事实、统计和实验合同；
- `RUN.md`：仅保留当前实验命令；
- `introduction.md / related_work.md / preliminaries.md`：中文正式写作稿；
- `reviewer_audit.md`：投稿前反方审查；
- 过期策略、日期型状态稿和失败方案移入 `_archive/`，不得与 canonical docs 并列。
