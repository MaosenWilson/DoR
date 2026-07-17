# RC-GRPO：AAAI-2027 统一故事与证据合同

> 本文档是论文叙事、贡献、术语和证据状态的唯一事实源。Method 只写这里定义的方法，Experiments 只验证这里登记的主张。历史方案和负结果不得反向进入主线。

## 1. 一句话论点

在 tokenized 视频世界模型的 RLVR 后训练中，verifiable reward 并不天然等于可靠学习信号：冻结 tokenizer-decoder 使 raw ground truth 超出策略可达输出空间，多步 rollout 又把一个 sequence-level score 广播给承担不同未来影响的 frame-token blocks。我们先定位 raw/reconstructed target 之间改变候选排序的残差交互，再检验 encoder reconstruction 是否为局部合理的可达目标，最后用 future-frame return 分配 block-level credit。

英文工作句：

> Verifiable rewards are not automatically reliable learning signals for tokenized video world models. We diagnose codec-induced rank disagreement, constrain reachable-target updates by raw-fidelity progress, and assign future-frame credit to the token blocks that can affect it.

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

## 4. 统一原则：支持集、raw 保真约束与时间信用

现有 decoded verifier 面临一个不能靠“换 target”单独解决的双重约束：raw GT 通常不属于冻结 decoder 的可达集合，因此会把 codec residual interaction 混入候选排序；但 held-out 任务仍要求预测接近 raw GT，完全以 reachable reconstruction 替换训练 target 又可能改变最优策略。RT-1、VP2 与 IRIS 的现有结果正好暴露了这一区别：RC 的 rank/latent mechanism 可跨 codec 复现，而 raw-GT downstream conversion 尚不普适。因此下一候选主线不是继续增加 reward 指标，而是把 raw fidelity 写成 RC policy update 的显式约束：

$$
\boxed{
\text{reachable-target calibration}
\rightarrow
\text{raw-anchored constrained update}
\rightarrow
\text{frame-block temporal credit}
}
$$

RC 是已验证的支持集校准。**Raw-Anchored Reconstruction Calibration (RA-RC)** 是待验证的 policy-update extension：在同一 candidate group 上同时构造 raw 与 RC 的 GRPO surrogate，将 RC 梯度投影到“至少保持 raw-GRPO 一阶进度”的半空间，再叠加一次共享 KL anchor。它不使用静态融合权重、不删除候选，也不改变 evaluator。该方法必须在 RT-1、VP2 与 actor-policy-matched IRIS 上使用同一公式和共同 raw-GT primary 才能升格为贡献；当前仅为预注册候选。

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

### 5.2 Raw-Anchored Reconstruction Calibration（候选，待验证）

在同一 candidate group 上分别由 $R^{\mathrm{raw}}$ 与 $R^{\mathrm{RC}}$ 构造标准化 GRPO surrogate，记其**最小化方向**为 $g_{\mathrm{raw}}$ 与 $g_{\mathrm{RC}}$。RA-RC 求解

$$
g^*=\arg\min_g\frac12\|g-g_{\mathrm{RC}}\|_2^2
\quad\mathrm{s.t.}\quad
\langle g,g_{\mathrm{raw}}\rangle\geq\|g_{\mathrm{raw}}\|_2^2.
$$

其闭式解为

$$
g^*=g_{\mathrm{RC}}+
\frac{\left[\|g_{\mathrm{raw}}\|_2^2-
\langle g_{\mathrm{RC}},g_{\mathrm{raw}}\rangle\right]_+}
{\|g_{\mathrm{raw}}\|_2^2+\epsilon}g_{\mathrm{raw}}.
$$

该投影是相对 RC 方向的最小欧氏修正，并保证同一批候选上的 raw surrogate 一阶下降不低于 raw-GRPO 方向。约束按 candidate group 独立施加，再跨 context 累积；KL 只计算一次并在投影后加入。该保证只针对 sampled surrogate 的一阶局部几何，不等价于 held-out LPIPS 保证，因此必须由 paired training 验证。

### 5.3 Temporal-Return GRPO

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

### 5.4 已否定扩展：RC Energy Verifier

令 $x_i=D(o_i)$、$\tilde y=D(E(s'))$，并用冻结的多尺度特征映射 $\psi$ 定义

$$
\Delta(x,x')=\|\psi(x)-\psi(x')\|_2.
$$

$\psi$ 由 RGB 与 LPIPS-VGG 多层归一化特征构成；各 block scale 仅由独立 calibration split 的 base-policy candidates 估计并冻结。条件预测分布的 Energy objective 为

$$
J_{\mathrm{RCE}}
=-\mathbb E\Delta(X,\tilde y)
+\frac12\mathbb E\Delta(X,X').
$$

其 score-function 梯度对应候选 reward

$$
R_i^{\mathrm{RCE}}
=-\Delta(x_i,\tilde y)
+\frac1{G-1}\sum_{j\ne i}\Delta(x_i,x_j).
$$

该系数由 proper score 推导，不引入可调 diversity weight。第一项校准 reachable fidelity，第二项是该候选对 ensemble score 的分布贡献；两者仍由原始 GRPO 做组内标准化。正式训练前必须通过两个独立 candidate groups 的 cross-group utility 门控，且 raw-GT LPIPS/MSE 不得出现预注册阈值以上的退化。

### 5.5 已否定扩展：Rank-Reliable Temporal Return

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
(c) **跨 codec、跨 horizon、跨架构闭环验证**（§3.3 与 experiments.md §10，episode-cluster bootstrap）：rank disagreement 在 single-step CNN-FSQ（$\Delta\rho$ +0.0168，p=0.0075）与 multi-step compressive FSQ（$\Delta\rho$ +0.0145，p<5e-4；h2–7 逐层显著）同时降低；在外部 iVideoGPT/VP2 dual-VQ 上，使用真实 decoder-input post-quant latent 而非类别 ID Hamming 作为独立参照，H2/H8 的 $\Delta\rho$ 分别为 +0.0932/+0.0753，95% CI 均严格为正；在 IRIS Atari VQ Transformer 上，Breakout/Pong 分别为 +0.2952/+0.4275，并同步降低 pairwise flip。$\arccos(\rho)/\pi$ 仅作为已有统计关系的预测校验，不作为新理论；修复量随 RIR 四分位单调增强（Spearman +0.120 [+0.027,+0.202]，p=0.007）。downstream 由 paired training 承接：single-step RT-1 的 5-seed 结果成立；multi-step RT-1 的独立 RC 增益与 VP2 raw-GT pixel training transfer 均未成立，因此“rank repair”不得写成“所有平台均提升最终像素指标”。
(d) **明确的 target-refinement 边界**：固定 8 个高残差 latent cells、2 轮合法 FSQ 相邻移动的 64-window gate 中，MRRT 在 64/64 窗口改善其优化目标（mean relative gain 0.834%），但在 `raw / encoder-RC / MRRT / matched-random legal move` 的三 seed 下游训练中未胜过 encoder-RC，也未胜过 matched-random control。因此 MRRT 不构成方法或贡献；它只否定“encoder reconstruction 是度量最优投影”的过强说法。配套负空间（进边界不进贡献）：围绕校准 verifier 的 reward 构造五层系统证伪（等权融合/学习权重/面板扩展/在线约束/空间池化）——最小校准双指标 verifier 在此接口信息饱和。

C1 的证据合同明确分成两级，而不是把 rank correlation 与训练收益混为一谈：**Level I** 检验
同候选排序是否更贴近 decoder-input representation，回答问题是否跨 codec 存在；**Level II** 在
与 checkpoint 匹配的 state-action distribution 上检验这种修复能否改善 held-out raw-GT fidelity。
Level I 已跨 RT-1、VP2 与 IRIS 成立；Level II 当前只在 RT-1 single-step 成立。VP2 与 IRIS 的
边界结果约束普适性，不得被 external mechanism table 偷换成 universal training gain。

**C2. Frame-block temporal-return credit assignment.** 将 multi-step video GRPO 的信用单位从整条 sequence 改为 future-frame token blocks，通过 reward-to-go 把后续预测质量分配给能够影响它的前序 block；截断 return 与 candidate-shuffled return 在保持同一 reward 边际分布时破坏 future-credit correspondence，用于区分真正的时间信用与一般重标度。正式训练必须使用与 RLVR-World/VERL 一致的 `low_var_kl`，旧线性 KL 结果只作探索性证据。

### 6.2 条件成立的第三贡献：Calibration--Credit Coupling

**状态：机制门 GREEN，训练交互待验；尚不是当前贡献。** C1 与 C2 同时启用后取得最低均值，只能说明完整系统有效，不能自动构成第三项贡献。C3 只有在“机制累积”和“训练超加和交互”两层同时通过时成立。

逐 horizon 写出 raw 与 RC verifier 的差异

$$
R^{\mathrm{raw}}_{i,h}=R^{\mathrm{RC}}_{i,h}+c_h+\eta_{i,h},
\qquad \frac1G\sum_i\eta_{i,h}=0.
$$

对从 block $t$ 开始的 temporal return，有精确分解

$$
G^{\mathrm{raw}}_{i,t}
=G^{\mathrm{RC}}_{i,t}+C_t+\Xi_{i,t},\qquad
\Xi_{i,t}=\sum_{h=t}^{H}\gamma^{h-t}\eta_{i,h}.
$$

组内中心化会消去 $C_t$，但不会消去候选相关的 $\Xi_{i,t}$。因此更长的 credit path 可能累积 verifier residual interaction，使 temporal credit 比 sequence-level credit 更依赖校准。该论点不预设各 horizon residual 独立；$\Xi_{i,t}$ 的实际离散度和排序影响由冻结候选 cache 直接测量。

训练端采用 verifier（raw/RC）$\times$ credit（sequence/return）的完整 $2\times2$。对 lower-is-better 指标 $Y$，定义每个 seed 的超加和交互

$$
I_s=
\left(Y_{\mathrm{return,RC},s}-Y_{\mathrm{seq,RC},s}\right)
-\left(Y_{\mathrm{return,raw},s}-Y_{\mathrm{seq,raw},s}\right).
$$

$I_s<0$ 表示 RC 使 Temporal Return 获得了超过两模块简单相加的额外收益。正式 C3 要求：固定 $n=10$ 的 primary LPIPS 上 mean $I<0$、至少 7/10 seeds 同向、单侧 exact sign-flip $p<0.05$、paired bootstrap 95% CI 上界小于零，且完整 `return-RC` 臂具有四臂最低 mean。LPIPS-last/MSE 只作支持性 readout，不替代 primary。机制审计还必须显示 RC 相对 raw 提高 return ranking 与 pre-decode return diagnostic 的一致性，并且 early return 中累积的 residual dispersion 高于 late return。任一层失败，C3 降级为“完整方法组合”，不得单列贡献。

2026-07-14 冻结 base-policy cache 的零训练机制门通过。随着 return 包含的 reward 项从 1 增至 6，候选相关 residual dispersion 从 0.00425 单调增至 0.01698；跨所有 starts，RC-minus-raw 的 return-rank correlation 增益为 $+0.0132$，episode-cluster bootstrap 95% CI $[+0.0085,+0.0176]$，pairwise disagreement 变化为 $-0.0082$，95% CI $[-0.0104,-0.0058]$。earliest-minus-latest residual dispersion 为 $+0.01272$，95% CI $[+0.01102,+0.01524]$。该结果支持 residual interaction 沿 return 累积，但不证明训练超加和；C3 仍等待官方 `low_var_kl` 四臂结果。

### 6.3 C1 扩展方向（预注册，尚非已成立贡献）

**Rejected candidate: Codec- and Distribution-Calibrated Verification.** RC 已解决 target support mismatch，但 pointwise expected distance 只刻画单个预测与一个观测之间的误差，没有利用 RLVR 已采样的 $G$ 成员条件 ensemble。RC-Energy 用 proper Energy Score 同时评价 reachable fidelity 与候选分布，并从总体 objective 的 score-function gradient 导出无自由融合系数的 candidate reward。

准入门控固定使用相同 contexts 上两个独立 generation seeds。组 A 内计算的 RC-Energy influence 必须比 pointwise RC 更能预测组 B 定义的 out-of-group raw-target distributional utility，A→B 与 B→A 的 episode-bootstrap 90% CI 下界均须大于零；同时 top-candidate raw LPIPS/MSE 不得超过预注册 non-inferiority margin。shuffled-context 与 reversed pairwise term 是负对照。任一 primary 条件失败即停止，不进入 GRPO。

正式门控结果为 RED：两方向 cross-group Pearson/Spearman 均大幅改善且通过 reversed/shuffled controls，但 top-candidate raw-GT LPIPS 退化约 0.0038--0.0040、MSE 退化约 6%，超过预注册边界。故未运行 $2\times2$ GRPO，C1 保持现有 Codec-Conditioned Rank Calibration，不将 distributional extension 写入贡献。

#### 已否定备选：Reachability-Constrained Action Verification

该分支曾尝试验证一个与视觉保真正交的 action axis。对真实 RT-1 transition 训练冻结的低容量 latent inverse verifier

$$
q_\psi(a_t\mid z_t,z_{t+1}),\qquad z_t=E(s_t),
$$

并对生成候选定义

$$
R_i^{\mathrm{act}}
=\sum_{d\in\mathcal M}w_d\log q_\psi(a_{t,d}\mid z_t,z_i).
$$

Gate A-v1 使用单步相邻 FSQ、$4\times5$ 强池化、线性 ridge 和 episode-disjoint split。正式结果为全维负 $R^2$（mean $-0.739$）、NRMSE 劣于 mean-action baseline、permutation $p=0.557$。这个结果只否定该四项假设的联合实现，不能区分动作/帧延迟、空间信息损失、连续动作不可辨识和跨 episode 域移；因此不把一次负结果错误外推为“动作验证不可行”。

Gate A-v2 改为预注册的 **action observability dissection**。在不生成候选、不训练 world model 的真实 transition 上，固定扫描 $h\in\{1,2,3,4\}$、动作偏移 $\delta\in\{-1,0,+1\}$、首动作/区间均值两类 target、$4\times5$/$8\times10$ 两种空间保真度，并分别报告 transition-random、episode-blocked 与 episode-disjoint split。配置和 ridge 强度只由 train/calibration 选择；episode-disjoint test 对最终选择只读一次。连续 $R^2$ 是 primary，方向 balanced accuracy 与 episode-matched action retrieval 是结构诊断，within-episode circular shift 是 null。

Gate A-v2 的 FSQ 结果为 RED：random split 在 $h=4$ 的 mean-action target 上达到 $R^2=0.404$，但 blocked 为 $-0.333$，episode-disjoint 为 $-0.066$、retrieval $0.500$、permutation $p=0.945$。更关键的是 transition 相对 state-only 的 normalized-error gain 为 $-0.0351$，90% CI $[-0.0582,-0.0124]$，四个 test episodes 全负。这证明 random 正数来自同轨迹相关/重叠捷径，不是可泛化的 action-effect evidence。

只有 episode-disjoint test 同时满足 mean $R^2>0$、permutation $p<0.05$、action retrieval 的 episode-bootstrap 90% 下界 $>0.5$，且完整 transition feature 相对 matched state-only predictor 的 normalized-error gain 90% 下界 $>0$，才认为存在动作可观测性。A2.2 RAFT oracle 达到该门槛：$R^2=+0.0055$、retrieval $0.527$ [0.503, 0.554]、permutation $p=1/201$、transition gain $+0.0335$ [0.0162, 0.0527]；但 calibration 选中的是 $h=1,\delta=-1$，即过去一拍动作。由于生成 prompt 用当前 action 预测下一帧，这个结果只支持 control/observation latency，不可直接作为 command verifier。

A2.3 因而施加架构约束：固定 $\delta=0$，target 只能是从当前命令开始的 action-segment mean，$h$ 与空间尺度仍只由 calibration 选择。它复用同一 test，故是开发 gate 而非新增独立证据；必须通过同一四项门槛才允许拟合 candidate verifier。若失败，不能用 $\delta=-1$ 的正结果包装 action-conditioned reward。C2 仍用官方 `low_var_kl`、完整 $2\times2$、截断 return 与 candidate-shuffled control 验证 temporal correspondence。

A2.3 按预注册门槛为 RED，但属于统计近失配而非方向否定：$R^2=+0.012$、permutation $p=1/201$、transition gain $+0.0381$ [0.0204, 0.0581] 均通过，retrieval mean $0.524$，仅其 90% lower bound $0.497<0.5$。四个 test episode retrieval 为 0.508/0.521/0.480/0.585。不得事后降低阈值；A2.4 保持四项门槛不变，以 5-fold nested episode cross-fitting 让 20 个 episode 各做一次 outer test，内层独立选择 $(h,P,\alpha)$。它估计 split 稳定性，不构成独立新数据复现；只有 cross-fit GREEN 才进入候选 gate。

A2.4 完整 GREEN：OOF $R^2=+0.0258$，retrieval $0.524$ [0.512, 0.538]，transition-over-state gain $+0.0409$ [0.0222, 0.0611]，permutation $p=1/201$。正 $R^2$ 集中在 translation 三维（+0.110/+0.143/+0.077），rotation 三维为负。Gate B 因而冻结单步接口 $h=1,P=8\times10,\delta=0$，用 OOF $[R_d^2]_+$ 作为不可调 reliability weights；每个 episode 的 generated candidates 由未训练该 episode 的 outer-fold verifier 评分。除非 matched command 相对跨 episode shuffled command 显著提高与独立 pixel-delta motion readout 的组内相关，否则不得进入 GRPO。

Gate B1（80 contexts，$K=16$，generation seed 7401）按门槛为 RED：nondegenerate 1.0、median $|\rho(R^{act},R^{RC})|=0.301$，但 $\rho(R^{act},\mathrm{dmotion})=+0.064$ [−0.020, +0.141]、matched-minus-shuffled $+0.063$ [−0.037, +0.162]。三个 secondary 方向一致：top-bottom dmotion +0.018、mean $\rho$ raw-fidelity +0.137、LPIPS top-bottom +0.00333。鉴于每个 context 仅观测一次随机 candidate group，B2 固定同一 contexts、改用独立 generation seed 17401；两个 replication 先按 context 平均，再按 episode bootstrap。阈值不变，B1 不被覆盖；combined RED 后不再增加 seed。

Gate B2 最终 RED，RCAV 分支终止。独立 seed 几乎复现 B1 均值（dmotion rho +0.057；matched-minus-shuffled +0.062），说明弱正方向不是单 seed 偶然；但两次 context-average 后仍为 rho $+0.060$ [−0.012,+0.127]、command delta $+0.063$ [−0.035,+0.153]。因此真实帧上的 command observability 已成立，生成候选上的 command-specific utility 未成立。不得增加第三 seed、改用更有利 secondary，或将 RCAV 接入 GRPO；该分支只作为诊断边界，不构成贡献。

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

除四个常规配对差值外，必须按 §6.2 的 $I_s$ 报告 difference-in-differences。不得用“full 臂均值最低”替代交互检验，也不得在 $n=5$ provisional gate 后直接声称 C3。正式顺序为：先用现有冻结 multi-step cache 做 `Calibration--Credit Coupling` 机制审计；再完成官方 `low_var_kl` 四臂 seeds 0--4；若 primary 交互 mean 为负、至少 4/5 同向且 `return-RC` 为四臂最低，才扩到固定 seeds 0--9。

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

## 13. 外部验证边界与结果

外部平台不构成第三项方法贡献，也不用于事后重选 RC 指标、$\gamma$、学习率或训练步数。
它只检验 C1/C2 是否依赖 RT-1 的 CNN-FSQ/Llama 组合。

### 13.1 与 RLVR-World 的层级关系

首先必须澄清比较层级：RLVR-World 是一种对预训练世界模型进行后训练的框架，而
iVideoGPT 和 IRIS 首先是世界模型架构及其原始训练方案。因此，准确的问题不是“另外两个
RLVR 框架与 RLVR-World 有何区别”，而是“同一套 verifier/GRPO 干预能否接入不同的
tokenized autoregressive world models”。三者共享的最小结构为
$s\xrightarrow{E}z\xrightarrow{p_\theta(\cdot\mid z,a)}\hat z\xrightarrow{D}\hat s$：视觉帧经
有损离散编码器变成 token，自回归 Transformer 在动作条件下预测未来 token，再由 decoder
还原图像。这个共同结构使 raw target 与 decoder-reachable target 的错配在三个平台上都可定义。

| 维度 | RLVR-World 视频分支 | iVideoGPT / VP2-RoboSuite | IRIS / Atari |
| --- | --- | --- | --- |
| 原始目标 | 对世界模型做 verifiable-reward 后训练 | 预训练并适配可交互视频世界模型 | 学习世界模型，并在想象轨迹中训练 actor-critic |
| 视觉编码 | 单步 per-frame FSQ；多步 compressive FSQ | conditional VQGAN/VQVAE 式双编码器-解码器，future frame 压缩为 dynamics tokens | 单帧 VQVAE，$64\times64$ 图像压成 16 个、词表大小 512 的 token |
| 动力学模型 | LLaMA 式自回归 Transformer | LLaMA 式自回归 Transformer | GPT-2/minGPT 式自回归 Transformer |
| 动作空间 | RT-1 连续机器人动作，经量化后条件化 | VP2-RoboSuite 连续机器人动作，经线性投影注入 frame slot | Atari 离散动作，作为序列 token 与视觉 token 交错 |
| 世界模型原始训练 | token MLE 后再用 GRPO 后训练 | token cross-entropy；原文没有用 GRPO 优化世界模型 | transition/termination 用 cross-entropy，reward head 用 MSE 或 cross-entropy；原文没有用 GRPO 优化世界模型 |
| “奖励”的含义 | decoded prediction 与真实未来帧的 full-reference 指标，直接更新世界模型 | 原文的 reward prediction/MPC score 服务控制；不是候选对真实帧的 RLVR verifier | 环境 reward 由世界模型预测并训练 actor-critic；不是候选对真实帧的 RLVR verifier |
| 我们能验证什么 | C1 诊断、RC 训练转化与 C2 Temporal Return 主协议 | C1 跨机器人数据与跨 tokenizer；H8 可作 C2 外部压力测试 | C1 跨模型架构、跨视觉域与跨动作空间；当前不承担 C2 或训练收益主张 |

这两个外部平台形成的是“近域控制 + 远域探针”，而不是两个同质复现。iVideoGPT 与
RLVR-World 同属一个技术谱系，模型作者和 LLaMA 式 token generation 都高度接近，因此它
不能单独证明跨架构；它的价值是尽量少改动生成范式，只更换机器人数据、分辨率和 tokenizer
实例，检验 C1/C2 是否只是 RT-1 checkpoint 的偶然现象。RT-1 与 VP2-RoboSuite 都是固定视角下
的桌面机器人操作，任务语义和主观画面构图高度重合；差异主要来自真实采集与仿真渲染、Google
Robot 与 Sawyer embodiment、$256\times320$ 与 $64\times64$ 分辨率，以及 FSQ 与 VQ tokenizer。
因此 VP2 只能称为 near-domain controlled transfer，不能称为强视觉跨域验证。IRIS 来自独立作者和代码体系，改用
VQVAE、GPT-2 式 Transformer、Atari 离散动作及 imagined actor-critic，因而提供真正的
架构与领域外推；但正因为 IRIS 原文不包含 GRPO，我们只能把它表述为将同一诊断接口接到
外部生成器上的验证，不能声称复现了“IRIS-RLVR”。

选择这两者还基于可执行性而非事后挑选：二者均提供官方代码和公开 checkpoint，均能恢复
真实动作条件下的未来帧 token 生成，并能获得与候选严格配对的真实下一帧。缺少任一条件，
就无法在同一 candidate group 上比较 raw/reachable verifier，也无法进行配对 rank-disagreement
统计。论文中不把它们称为“最新方法”；iVideoGPT 为 NeurIPS 2024，IRIS 为 ICLR 2023，
选择依据是正交的可验证性与公开资产，而不是发表时间。

**VP2-RoboSuite/iVideoGPT 是主外部平台。** 它具有冻结压缩 tokenizer、动作条件离散
自回归 Transformer、公开 checkpoint 和机器人仿真轨迹；因此同一 RC 目标
$D(E(s'))$、raw full-reference evaluator 与 frame-block Temporal Return 都可以保持语义不变。
外部实验固定使用 raw/RC verifier $\times$ sequence/return credit 的完整 $2\times2$，测试 codec、
机器人任务与上游实现变化后是否仍有可复核主效应或交互；不声称重现或击败 VP2 原文的控制结果。

正式结果将 C1 与 C2 清楚分开。C1 同候选门控在 H2 与 H8 均通过严格 95% episode-bootstrap：
$\Delta\rho=+0.0932$ [0.0674, 0.1217] / +0.0753 [0.0408, 0.1104]，
$\Delta\mathrm{flip}=-0.0409$ [-0.0524, -0.0306] / -0.0346 [-0.0488, -0.0207]；
RC top 相对 raw top 的 post-quant latent gain 同样为正。原 token-Hamming gate 只保留为敏感性分析，
因为 VQ token ID 是类别编号，不具有等距几何。P2 的 3-seed 固定协议不支持 RC 或 Temporal Return
相对 raw-sequence 的 raw-GT pixel improvement；per-horizon scale equalization 和部分 RC 投影的小代价
pilot 也未通过训练准入。因此 VP2 加强 C1 机制外部有效性，但不支持 C2 或统一 downstream 增益。

**IRIS Atari 是许可证隔离的架构探针。** IRIS 的离散 autoencoder 和 Atari token Transformer
可检验 reachable-target diagnosis 是否超出连续机器人图像域。但其 upstream 代码为 GPL-3.0，
只能以独立进程读取/写入 rollout；DoR 不导入、复制或链接 IRIS 源码。Atari ROM provenance、
环境版本和 deterministic replay 未冻结前，IRIS 不进入论文结果或贡献表。现已固定 ALE 0.9.0、
action repeat 4、64x64 resize、4-frame token/action history、16 episode clusters、每游戏 128 windows、
$K=16$ 与两个 generation draws。Breakout/Pong 的 $\Delta\rho$ 分别为 +0.2952
[0.2188, 0.3751] / +0.4275 [0.2899, 0.5676]，$\Delta\mathrm{flip}$ 分别为 -0.1336
[-0.1697, -0.0995] / -0.2182 [-0.2899, -0.1477]，且 RC-top latent gain 的 95% CI 均严格为正。
主协议后的 stride-4 低重叠复核仍为 GREEN：Breakout/Pong 的 $\Delta\rho$ 为 +0.2230
[0.1543, 0.3005] / +0.3989 [0.2914, 0.5050]，$\Delta\mathrm{flip}$ 为 -0.0938
[-0.1247, -0.0650] / -0.1756 [-0.2227, -0.1287]。IRIS 因而只进入 C1 的跨架构
mechanism table，不承担 C2 或训练收益主张。

后续 raw/RC × 5-seed single-step conversion 在均匀随机动作 cache 上未通过 raw-GT LPIPS primary：
Breakout 为 3/5、Pong 为 1/5；但两个游戏的 exact-token mismatch 与 post-quant latent RMS 均为
5/5 改善。更重要的是 raw 与 RC 都未稳定胜过原 checkpoint。该结果定位为 random-policy stress
test，而不是 RC-vs-raw 的有效学习判决：公开 checkpoint 含 actor-critic，上游测试轨迹由该 policy
采样；我们当前 cache 的均匀随机动作造成分布错配，且 Pong 已接近像素天花板。下一次外部训练验证
只能使用独立的 actor-policy-matched manifest，并保留本结果，不能事后挑 step 10 或重调同一 test。

外部成功不能补救 RT-1 上不成立的主张；外部失败也不能由有利 seed、不同 evaluator 或视觉案例
替代。它只会使 C1/C2 的适用范围更清楚。
