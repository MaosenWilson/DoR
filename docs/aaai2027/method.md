# Method 中文工作稿

> 与 `story.md` 同步。正式 tex 目前仍只包含已验证的 RC verifier 与 Temporal-Return GRPO。RA-RC 是预注册候选更新规则；只有三平台共同判据通过后，才可将其写入 tex 的已成立贡献。

## 1. Problem Formulation

给定 $C$ 帧 context $s_{1:C}$ 和动作序列 $a_{1:H}$，tokenized video world model $\pi_\theta$ 自回归生成 $H$ 个 future-frame token blocks：

$$
o_i=(o_{i,1},\ldots,o_{i,H}),
\qquad o_i\sim\pi_\theta(\cdot\mid s_{1:C},a_{1:H}),
$$

其中 $i\in\{1,\ldots,G\}$ 表示同一条件下的 candidate trajectory。冻结 tokenizer $E$ 与 decoder $D$ 将 token block 解码为

$$
\hat s_{i,h}=D(o_{i,h}).
$$

RLVR 使用与真实 future frame $s_h'$ 的 full-reference distance 构造 reward，并由 GRPO 的组内相对优势更新策略。本文保持数据、world model、tokenizer、采样器和 KL anchor 不变，只研究 verifier target 与 temporal credit assignment。

## 2. Reconstruction-Induced Rank Corruption

### 2.1 Decoder-reachable output set

冻结 decoder 的输出受限于

$$
\mathcal S_D=\{D(z):z\in\mathcal Z\}.
$$

raw ground truth $s_h'$ 通常不严格属于 $\mathcal S_D$。即使策略预测 ground-truth token $E(s_h')$，其可达到的像素目标仍是

$$
\tilde s_h'=D(E(s_h')).
$$

定义 reconstruction residual：

$$
e_h=s_h'-\tilde s_h'.
$$

### 2.2 GRPO-relevant failure mode（C1-a：精确残差分解，2026-07-12 定稿）

对 MSE 与 LPIPS 这类加权平方特征距离（MSE 取 $\phi=\mathrm{id}$；LPIPS 为逐层线性加权的
平方特征距离，分解逐层成立），令

$$
u_i=\phi(\hat s_i),\quad v=\phi(s'),\quad \tilde v=\phi(D(E(s'))),\quad e=v-\tilde v,
$$

则 raw 与 RC verifier 之间是**精确恒等**而非近似：

$$
R_i^{\mathrm{raw}}=-\|u_i-v\|_W^2
= R_i^{\mathrm{RC}} + 2\langle u_i-\tilde v,\,e\rangle_W - \|e\|_W^2 .
$$

对候选对 $(i,j)$：

$$
\Delta R_{ij}^{\mathrm{raw}} = \Delta R_{ij}^{\mathrm{RC}} + b_{ij},
\qquad b_{ij} = 2\langle u_i-u_j,\,e\rangle_W .
$$

$-\|e\|_W^2$ 是组内常数，被 GRPO 去均值精确消除；**唯一能改变组内排序的是候选相关交互项
$b_{ij}$**。恒等式本身只证明 raw 与 RC 的排序差异来自哪里，并不自动证明 $b_{ij}$ 全是噪声：
某些 alternative token sequence 可能借助该交互更接近 raw GT。本文因此把现象保守地称为
reconstruction-induced rank disagreement；只有经过独立参照、局部可达性和训练结果验证后，
才把被 RC 修复的部分解释为 rank corruption。

在二元高斯近似下，若含扰动 reward 与诊断参照的相关系数为 $\rho$，pairwise sign disagreement 为

$$
p_{\mathrm{flip}}=\frac{\arccos(\rho)}{\pi}.
$$

该关系是已有相关高斯符号关系；本文贡献是将其操作化为 tokenized video verifier 的诊断，并通过同候选 raw/RC 重评分闭环验证。

## 3. Reconstruction-Calibrated Verifier

RC 使用冻结 codec 的 encoder-induced reachable reconstruction：

$$
\tilde s_h'=D(E(s_h')).
$$

逐帧 reward 为

$$
R^{\mathrm{RC}}_{i,h}
=-
\left[
\operatorname{MSE}(\hat s_{i,h},\tilde s_h')
+\operatorname{LPIPS}(\hat s_{i,h},\tilde s_h')
\right].
$$

两个距离沿用 RLVR-World 的等权口径，不加入 code、SSIM 或 learned fusion。候选和 target 均由同一冻结 decoder 产生，但最终 evaluator 仍计算 $d(\hat s_{i,h},s_h')$。

这里不把 $D(E(s'))$ 称为度量投影。一般情况下，encoder 并不求解

$$
\arg\min_{z\in\mathcal Z} d(D(z),s'),
$$

因此附近 token code 可能解码得比 $E(s')$ 更接近 raw GT。该区别构成 C1 的首要可证伪门。

### 3.1 设计边界

RC 适用于：策略输出受冻结 tokenizer-decoder 约束，并使用 decoded full-reference verifier 的设置。以下情况不由本文覆盖：直接像素生成、近无损 tokenizer、联合训练 decoder、纯 pre-decode reward，或没有可测 rank corruption 的场景。

### 3.2 C1-b：训练前 Reachability Audit（RIR）

组级 residual-interaction ratio：

$$
\mathrm{RIR}(q)=\frac{\operatorname{Std}_i\!\left(R_i^{\mathrm{raw}}-R_i^{\mathrm{RC}}\right)}
{\operatorname{Std}_i\!\left(R_i^{\mathrm{RC}}\right)+\epsilon}.
$$

分子只保留 $b$ 的候选相关部分（常数 $-\|e\|^2$ 被 Std 消除），故 RIR 直接度量"交互项相对
真实信号有多大"。用途是训练前诊断：raw/RC 差异是否只是常数、交互是否足以改变排序、当前
codec 是否需要校准、RC 修复量是否随 severity 增强。**诚实边界（预注册）**：RIR 的
context 级分类能力弱，只作 severity stratification，不作在线 gate、不作 context 权重——
floor-filter 与 Rank-Guard 的失败已证明把弱预测量升格为在线干预的后果。

### 3.3 C1-c：跨 codec、跨 horizon 验证（2026-07-12 实测，episode-cluster bootstrap 2000）

零训练审计，`scripts/audit_rank_calibration.py`，聚类单位=episode（组间非独立）：

| 层 | groups | $\rho$ raw→RC | $\Delta\rho$ [95% CI] | boot-p | flip raw→RC | $\Delta p_{\mathrm{flip}}$ | boot-p |
|---|---:|---|---|---|---|---|---|
| single-step CNN-FSQ（148 窗） | 148 | 0.850→0.866 | +0.0168 [+0.0035,+0.0289] | 0.0075 | 0.155→0.147 | −0.0082 | 0.030 |
| multi-step compressive FSQ（3 gen×64 窗×h2–7） | 1152 | 0.726→0.740 | +0.0145 [+0.0105,+0.0176] | <5e-4 | 0.234→0.225 | −0.0087 | <5e-4 |

逐 horizon h=2..7 全部显著（$\Delta\rho$ +0.0117~+0.0182，$\Delta p_{\mathrm{flip}}$
−0.0077~−0.0102，boot-p≤0.001）；$\arccos(\rho)/\pi$ 律在两种 codec、校准前后均吻合
（平均偏差 ≈0.03）。RIR 四分位剂量响应（multi，pooled）：$\Delta\rho$ =
+0.0083/+0.0120/+0.0173/+0.0203 单调，cluster-boot Spearman(RIR, $\Delta\rho$) = +0.120
[+0.027,+0.202]，p=0.007；single-step 趋势 +0.151（单侧 p=0.04，由最高分位驱动，
按 3.2 边界只作 stratification 证据）。**结论：C1 不再依赖单一 tokenizer——rank repair
在更强压缩的第二 codec 与全部 horizon 上成立，且修复量随 RIR severity 增强。**

外部 VQ tokenizer 的 decoder-free diagnostic 不能沿用 token Hamming 作为主几何，因为 VQ ID
只是类别编号。设 codebook lookup 为 $e(z)$，decoder 在去 patch 或卷积前实际消费的线性投影为
$P(e(z))$，则外部主参照固定为

$$
Q_i^{\mathrm{pre}}
=-
\left\|P(e(o_i))-P(e(E(s')))\right\|_{\mathrm{RMS}}.
$$

iVideoGPT 使用其 `dynamics_quantize.embedding` 后接 `post_quant_linear`；IRIS 使用
`tokenizer.embedding` 后接 `post_quant_conv`。该量只作同候选 rank-closure 的独立 readout，
不进入 RC reward。类别 Hamming 与未投影 codebook RMS 全量保留为敏感性分析。冻结 cache 的
95% episode-bootstrap 结果在 VP2 H2/H8 与 IRIS Breakout/Pong 上均显示 RC 提高
$\rho(R,Q^{\mathrm{pre}})$、降低 pairwise flip，并改善 RC-top 的 $Q^{\mathrm{pre}}$；完整数字见
`experiments.md` §10。该证据支持 rank-calibration mechanism 跨 codec/架构成立，但不替代
raw-GT downstream training 结果。IRIS 主分析使用 episode-cluster bootstrap；另将采样窗口步长从
1 增至 4，以降低同一 episode 内相邻 transition 的重叠，两个游戏的三项结论仍保持 95% CI
严格通过。该 stride-4 结果只作相关性敏感性分析，不替换冻结的主分析。

### 3.4 Candidate update: Raw-Anchored Reconstruction Calibration

纯 RC verifier 改善 reachable-space rank agreement，但它也把训练目标从 raw GT 改为 encoder-induced reconstruction；这解释了为什么 RT-1、VP2 与 IRIS 可以同时出现“rank/latent 改善”与“raw-GT fidelity 不稳定”。RA-RC 将二者写成有优先级的双 objective，而不是再做 reward 加权。

对同一 candidate group，令 $\ell_{\mathrm{raw}}$ 与 $\ell_{\mathrm{RC}}$ 为仅含 policy-gradient 项的 GRPO surrogate loss。两者使用同一 candidates、同一 token log-probability、同一 advantage normalization 和同一 credit mode。定义最小化梯度

$$
g_0=\nabla_\theta\ell_{\mathrm{raw}},\qquad
g_1=\nabla_\theta\ell_{\mathrm{RC}}.
$$

我们选择最接近 RC 更新、但 raw surrogate 的一阶下降至少匹配 raw-GRPO 的方向：

$$
g^*=\arg\min_g\frac12\|g-g_1\|_2^2
\quad\text{s.t.}\quad
\langle g,g_0\rangle\geq\|g_0\|_2^2.
$$

这是到闭半空间的欧氏投影，闭式解为

$$
g^*=g_1+\lambda^*g_0,\qquad
\lambda^*=\frac{[\|g_0\|_2^2-\langle g_1,g_0\rangle]_+}
{\|g_0\|_2^2+\epsilon}.
$$

若 RC 已达到 raw-GRPO 的一阶进度，$\lambda^*=0$；否则只沿 raw gradient 增加满足约束所需的最小分量。每个 candidate group 独立投影，随后按 batch-window 权重累积。共享正则项

$$
\ell_{\mathrm{KL}}=\beta_{\mathrm{KL}}\widehat D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

不属于两个 verifier objective，只反向传播一次并加到 $g^*$；最后沿用原始 gradient clipping 与 optimizer。实现必须记录 $\lambda^*$、约束触发率、投影前后的 raw-progress ratio、两个梯度 cosine 和梯度范数。

该构造借鉴多目标学习中的梯度投影思想。PCGrad 在任务梯度冲突时删除负投影，CAGrad 优化平均目标并控制最差局部改善；RA-RC 不把这些通用几何操作声明为新发明。本文待验证的新内容是：由 codec reachability 产生的 raw/RC 双 verifier、raw-GRPO progress half-space、逐 GRPO group 的实施方式，以及它能否跨 tokenized video world models 同时保留 raw fidelity 与 reachable consistency。参考原始来源：Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020；Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*, NeurIPS 2021。

边界同样明确：约束保证的是当前 sampled surrogate 的一阶几何，不保证有限步更新后的 held-out raw LPIPS；Adam 的预条件、gradient clipping、KL 和 sampling drift 都会改变参数空间中的实际步长。因此三平台 paired experiment 是方法成立的必要条件，而不是装饰性验证。

### 3.5 Rejected target refinement: Metric-Refined Reachable Target

对每个 GT token grid $z_0=E(s')$，在固定预算的合法 FSQ 邻域 $\mathcal N_B(z_0)$ 中搜索

$$
z_B=\operatorname{GreedyRefine}_B\left(z_0;d(D(z),s')\right),
$$

其中 $B=2$，每轮只在 reconstruction-error 最高的 8 个 latent cells 枚举所有合法一阶 FSQ
邻居，并按与训练 verifier 相同的 $d=\mathrm{LPIPS}+\mathrm{MSE}$ 接受下降最大的移动。定义
$\tilde s'_{\mathrm{MRRT}}=D(z_B)$。搜索未证明收敛，因此本文称其 metric-refined target，绝不写成
全局或局部 $\arg\min$。

独立于正式训练窗口的 64-window gate 中，64/64 目标改善；mean/median/q05 relative objective
gain 为 0.834%/0.770%/0.382%，平均 Hamming fraction 0.00601。LPIPS 在 64/64 下降，MSE
在 39/64 下降，说明联合目标收益由实际尺度更大的 LPIPS 主导。所有窗口均接受 2/2 moves，
明确否定 encoder reconstruction 的局部最优性，但也表明 MRRT 是 budgeted refinement。

为排除任意 target preprocessing，matched-random control 从两轮实际枚举位置的同一并集执行相同
token-cell Hamming 预算的随机合法移动。四臂为 raw GT、encoder-RC、MRRT 与 matched-random；
held-out evaluation 始终对 raw GT，不使用 MRRT 自评。三 seed 配对训练结果中，MRRT 相对 encoder-RC 的
LPIPS/MSE/PSNR/SSIM 均未显示正向趋势，也未优于 matched-random control。因此 MRRT 仅作为一个
**被否定的局部 target-refinement 假设**：它证明 encoder reconstruction 不是该度量的局部最优投影，
但不支持把更低 target reconstruction error 当作更好的 GRPO verifier。

## 4. Candidate extension for C1: RC Energy Verifier

> **状态：正式双组门控 RED，未进入训练或正式方法。** 分布效用预测显著改善，但 raw-GT LPIPS/MSE 违反预注册 non-inferiority boundary；本节保留为被否定的纯 distribution-score 扩展，结果不得反向修改距离、系数或绿灯条件。

### 4.1 Frozen multi-scale geometry

令 $x_i=D(o_i)$ 为同一条件下第 $i$ 个 decoded candidate，$\tilde y=D(E(s'))$ 为 reachable target。冻结特征映射由 RGB block 与 LPIPS-VGG 主干的归一化中间特征构成：

$$
\psi(x)=\frac1{\sqrt B}
\left[
\frac{\operatorname{vec}(x)}{\sqrt{n_0}\hat\sigma_0},
\frac{\operatorname{vec}(\bar\phi_1(x))}{\sqrt{n_1}\hat\sigma_1},
\ldots,
\frac{\operatorname{vec}(\bar\phi_L(x))}{\sqrt{n_L}\hat\sigma_L}
\right].
$$

$\bar\phi_l$ 在每个空间位置做 channel-$\ell_2$ normalization；$n_l$ 是 block 元素数。$\hat\sigma_l$ 是 base-policy calibration candidates 与 reachable target 的 block RMS 中位数，只在独立 calibration split 上估计一次并冻结。该构造把 pixel fidelity 与 VGG 多尺度 perceptual geometry 写成一个 Hilbert embedding，而不是训练后选择的标量 reward 权重。定义

$$
\Delta(x,x')=\|\psi(x)-\psi(x')\|_2.
$$

主配置固定 Energy exponent $\beta=1$；不扫描 diversity coefficient。

### 4.2 Reachable conditional Energy objective

对于条件预测分布 $Q_\theta(\cdot\mid q)$，定义 higher-is-better objective

$$
J_{\mathrm{RCE}}(Q_\theta,\tilde y)
=-\mathbb E_{X\sim Q_\theta}\Delta(X,\tilde y)
+\frac12\mathbb E_{X,X'\sim Q_\theta}\Delta(X,X').
$$

第一项是 reachable-target fidelity；第二项使 score 评价完整条件分布，而不是把每个 sampled future 当成彼此无关的点估计。对上述 objective 使用 score-function identity，pairwise 项的两个对称采样变量各贡献一次梯度，因此单个候选的 Monte Carlo influence reward 为

$$
R_i^{\mathrm{RCE}}
=-\Delta(x_i,\tilde y)
+\frac1{G-1}\sum_{j\ne i}\Delta(x_i,x_j).
$$

这解释了 candidate reward 中系数为 $1$ 而总体 objective 中为 $1/2$；该系数不是超参数。reward 被 stop-gradient，并沿用 vanilla GRPO 的组内标准化与 policy loss。$G=1$ 时方法不可定义；实现显式拒绝 $G<2$。

### 4.3 Two-group admission gate

对相同 calibration contexts 独立采样 candidate groups $A$ 与 $B$。训练侧可见的 in-group influence 为

$$
R^{A}_i=-\Delta(x_i^A,\tilde y)
+\frac1{G-1}\sum_{j\ne i}\Delta(x_i^A,x_j^A),
$$

out-of-group raw-target utility 为

$$
U_i^{A\rightarrow B}
=-\Delta(x_i^A,y)
+\frac1G\sum_j\Delta(x_i^A,x_j^B).
$$

主统计是组内 Pearson/Spearman gain：$\rho(R^A,U^{A\to B})-\rho(-\Delta(x^A,\tilde y),U^{A\to B})$，并对 A→B、B→A 分别做 episode-cluster bootstrap。两方向 90% CI 下界均须大于零；RC-Energy top candidate 相对 pointwise RC 的 raw-GT LPIPS margin 不得高于 $0.002$，MSE relative degradation 不得超过 $2\%$。跨 context shuffled pairwise term与 reversed pairwise sign 是负对照。任一条件失败即判 RED，不进入训练。

### 4.4 Factorial training design after a GREEN gate

单步训练固定为 target（raw/reachable）× scoring rule（pointwise/energy）的 $2\times2$。为避免把 scoring-rule 增益与度量变化混淆，四个因子臂严格共享上式同一个冻结 $\Delta$：`raw_energy_point`、`rc_energy_point`、`raw_energy`、`rc_energy`。此外重跑 `a0faithful` 与 `a0faithful_tok`，分别作为原 RLVR 与现有 RC 的外部基线，但不把它们用于估计 Energy 因子的因果增量。所有 arm 使用相同 model/data/generation seeds 与 vanilla GRPO；headline evaluation 始终相对 raw GT。只有该因子实验支持完整方法后，RC-Energy 才接入 multi-step Temporal Return。

## 5. Rejected C1 extension: Reachability-Constrained Action Verification

> **状态：generated-candidate Gate B 两个独立 generation seeds 均为 RED。** 本节只保留失败边界，不进入训练或正式贡献。

### 5.1 Two non-substitutable verification objectives

RC 保留为 decoder-reachable state anchor：

$$
R_i^{\mathrm{state}}
=-
\left[
\operatorname{MSE}(D(z_i),D(E(s')))
+\operatorname{LPIPS}(D(z_i),D(E(s')))
\right].
$$

动作转移由冻结的 pre-decode inverse verifier 评分。对真实相邻帧提取低容量特征

$$
x_t=
\left[
P(z_t),\;P(z_{t+1}-z_t),\;P(|z_{t+1}-z_t|)
\right],
$$

其中 $P$ 是固定的 $4\times5$ adaptive average pooling。使用 episode-disjoint train/calibration/test split 拟合 grouped ridge $f_\psi(x_t)$ 预测归一化 arm-motion action。主分析动作维度固定为 $\mathcal M=\{0,1,2,4,5,6\}$，不根据世界模型训练结果选维度。生成候选的 action score 为

$$
R_i^{\mathrm{act}}
=-
\frac{1}{|\mathcal M|}
\sum_{d\in\mathcal M}
\left(
\frac{f_\psi(x_i)_d-a_{t,d}}
{\hat\sigma_{\mathrm{res},d}+\epsilon}
\right)^2,
$$

其中 residual scale 只由 train+calibration real transitions 估计并冻结。这是一个 learned verifier，因此可验证性受 held-out action prediction 能力限制；Gate A 必须先证明它胜过 mean/shuffled controls，且在 generated candidates 上预测独立 motion readout。

### 5.2 State-anchored conflict projection

不把两个 reward 预先压成一个标量。分别构造 group-relative losses $\mathcal L_s,\mathcal L_a$ 及梯度 $g_s,g_a$。动作梯度的冲突分量被单向移除：

$$
\bar g_a
=g_a-
\frac{\min(0,\langle g_a,g_s\rangle)}
{\|g_s\|^2+\epsilon}g_s,
$$

$$
\alpha=
\min\left(1,\frac{\|g_s\|}{\|\bar g_a\|+\epsilon}\right),
\qquad
g=g_s+\alpha\bar g_a.
$$

因此 $\langle g,g_s\rangle\ge\|g_s\|^2$，在一阶近似下动作目标不会抵消 state-anchor 的下降方向。这不是全局约束保证；实验必须与 scalar sum、action-only 及 symmetric PCGrad 直接对照。

## 6. Temporal-Return GRPO

### 6.1 Sequence-level baseline

sequence-level GRPO 将有效 future frames 的 reward 平均为

$$
\bar R_i=\frac{1}{H'}\sum_{h\in\mathcal H}R^{\mathrm{RC}}_{i,h},
$$

并将同一个标准化优势广播给整条 trajectory。它不能区分 frame block 对后续 prediction errors 的延迟作用。

### 6.2 Frame-block reward-to-go

对第 $t$ 个 future-frame token block，定义 discounted future return：

$$
G^{\mathrm{TR}}_{i,t}
=
\sum_{h=t}^{H}
\gamma^{h-t}R^{\mathrm{RC}}_{i,h}.
$$

在每个时间位置独立做 group-relative normalization：

$$
A^{\mathrm{TR}}_{i,t}
=
\frac{G^{\mathrm{TR}}_{i,t}-\mu_t}
{\sigma_t+\epsilon},
$$

$$
\mu_t=\frac1G\sum_iG^{\mathrm{TR}}_{i,t},
\qquad
\sigma_t^2=\frac1G\sum_i(G^{\mathrm{TR}}_{i,t}-\mu_t)^2.
$$

$A^{\mathrm{TR}}_{i,t}$ 只作用于 $o_{i,t}$ 中的 visual tokens。当前多步协议跳过没有直接 action-conditioned reward 的第一个 future frame，但 later-frame return 仍可向其分配信用。

### 6.3 Policy objective

设 $\ell_{i,t}=\sum_{k\in o_{i,t}}\log\pi_\theta(o_{i,t,k}\mid o_{i,<t},q)$，训练目标为

$$
\mathcal L_{\mathrm{TR}}
=-
\frac1G\sum_{i=1}^{G}\sum_{t=1}^{H}
A^{\mathrm{TR}}_{i,t}\ell_{i,t}
+\beta_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
$$

我们保持 $\beta_{\mathrm{KL}}=0.001$ 与固定 base reference。正式实现采用 RLVR-World/VERL 视频
配方的 sampled low-variance KL：令 $k=\log\pi_{\mathrm{ref}}-\log\pi_\theta$，逐 token penalty 为
$\exp(k)-k-1$。早期多步结果误用了线性 $\log\pi_\theta-\log\pi_{\mathrm{ref}}$；由于每批 rollout
只做一次 on-policy update，该形式不等价于官方 estimator。因此既有 C2 数字暂列探索性证据，
必须使用修正实现做固定协议复核。horizon-aware KL 已实测无额外收益，不属于方法。

## 7. Rejected Extension: Rank-Reliable Temporal Return

> **Rejected：离线 replay gate 为 RED，不进入正式 Method/贡献；本节仅保留失败机制记录，Tex 应删除。**

### 7.1 Horizon reliability calibration

在与 train/eval 分离的 calibration episodes 上，对每个 horizon 构造：

- RC reward $R^{\mathrm{RC}}_{i,h}$；
- raw-GT evaluator，仅用于离线判决；
- pre-decode token-distance diagnostic $R^{\mathrm{code}}_{i,h}$。

估计同组相关 $\rho_h$，并通过 episode-cluster bootstrap 汇总：

$$
\hat p_{\mathrm{flip},h}=\frac{\arccos(\hat\rho_h)}{\pi},
\qquad
w_h=\operatorname{clip}(1-2\hat p_{\mathrm{flip},h},0,1).
$$

为保持平均 reward scale，对 horizon weights 归一化：

$$
\bar w_h=\frac{w_h}{H^{-1}\sum_u w_u}.
$$

### 7.2 Reliability-shaped return

$$
G^{\mathrm{RR}}_{i,t}
=
\sum_{h=t}^{H}
\gamma^{h-t}\bar w_hR^{\mathrm{RC}}_{i,h},
$$

其余 group normalization 与 policy objective 不变。

### 7.3 退化与负对照

- $w_h$ 常数：退化为 plain Temporal Return；
- shuffled $w_h$：检验收益是否来自 reliability 与 horizon 的正确对应；
- reversed $w_h$：检验方向性；
- code distance 只作 calibration diagnostic，不直接加入训练 reward。

该权重把 pairwise sign reliability 映射为 horizon objective coefficient，是有解释的设计，但不是无偏 correction。若离线重放未通过，不实现训练分支。

## 8. RCAV Action-Observability Dissection（rejected）

Gate A-v1 的 verifier 为 $q_\psi(a_t\mid P(z_t),P(z_{t+1}-z_t))$，其中 $P$ 是 $4\times5$ 平均池化。其 episode-disjoint mean $R^2=-0.739$，因此当前 learned reward 不具备进入 GRPO 的资格；该结果不等价于动作在视频中不可观测。

Gate A-v2 对视觉间隔 $h$ 和动作对齐偏移 $\delta$ 定义

$$
x_{t,h}=\left[P(z_t),P(z_{t+h}-z_t),P(|z_{t+h}-z_t|)\right],
$$

$$
y^{\mathrm{first}}_{t,h,\delta}=a_{t+\delta},\qquad
y^{\mathrm{mean}}_{t,h,\delta}=\frac1h\sum_{u=0}^{h-1}a_{t+\delta+u}.
$$

固定 $h\in\{1,2,3,4\}$、$\delta\in\{-1,0,+1\}$、$P\in\{4\times5,8\times10\}$，动作维度为 `0,1,2,4,5,6`。组合与 ridge 正则只由 train/calibration 选择，episode-disjoint test 对最终配置只读一次。连续 $R^2$ 是 primary；方向 balanced accuracy 检查粗粒度可辨识性，episode-matched retrieval 检查预测是否能在同 episode 动作候选中找回匹配动作。对选中配置另拟合 matched $z_t$-only predictor，完整 transition feature 必须在 episode-cluster bootstrap 下显著降低 normalized action error，排除状态/任务先验捷径。random-transition 和 episode-blocked split 只定位记忆或域捷径，不作为方法有效性的证据。

若最佳信号位于 $h>1$，后续候选 reward 验证动作段而非单步动作：

$$
R^{\mathrm{act}}_{i,t:h}
=\log q_\psi\!\left(\bar a_{t:t+h-1}\mid z_t,z_{i,t+h}\right).
$$

该定义与 Temporal Return 的时间范围一致，但在 real-transition gate 与 candidate-level independent readout 同时通过前，不属于正式贡献。

Gate A-v2 的 FSQ 实例已判 RED：episode-disjoint mean $R^2=-0.066$、retrieval $0.500$、permutation $p=0.945$，且 transition 相对 state-only gain 的 90% CI 全负。A2.2 不增加 learned capacity，而以冻结 RAFT 从真实帧得到 $F_{t\rightarrow t+h}$，构造

$$
x^{\mathrm{flow}}_{t,h}
=\left[P(s_t),P(F_x),P(F_y),P(|F_x|),P(|F_y|),P(\|F\|_2)\right].
$$

它沿用 A2.1 的全部时间、target、split 与统计门槛，是“decoded motion 中是否存在动作效应”的 oracle，而不是候选 reward。A2.2 达到统计门槛，但最佳配置为 $h=1,\delta=-1$；这与生成器使用当前 action 预测下一帧的接口不一致。A2.3 因此固定 $\delta=0$、$y=\frac1h\sum_{u=0}^{h-1}a_{t+u}$，仅允许 calibration 选择 $h$ 与空间尺度。其 fixed-split gate 仅 retrieval CI 下界未过门。A2.4 使用五折 nested episode cross-fitting：外层每折保留 4 个完整 episode，内层在其余 16 个 episode 中以 12/4 选择 $(h,P,\alpha)$，随后 refit 并只预测外层 episode；20 个 out-of-fold episode prediction 汇总后沿用同一四项门槛。过去动作可辨识不能替代当前命令一致性，cross-fit 通过前不训练 spatial verifier、不生成 action reward。

A2.4 已 GREEN。候选阶段不继续选择结构，而固定单步 $h=1$、$8\times10$ flow feature。由 cross-fit OOF per-dimension $R^2_d$ 冻结

$$
w_d=\frac{[R_d^2]_+}{\sum_u[R_u^2]_+},
\qquad
R_i^{\mathrm{act}}
=-\sum_dw_d\left(\frac{\hat a_{i,d}-a_d}{\hat\sigma_d}\right)^2.
$$

每个 real/generated episode 使用未见过该 episode 的 outer-fold ridge payload，避免状态记忆泄漏。Gate B 同时计算 matched command 与跨 episode circular-shuffled command；primary 要求 matched $R^{\mathrm{act}}$ 与独立 raw pixel-delta motion fidelity 的 episode-bootstrap lower bound $>0$，且 matched-minus-shuffled correlation lower bound $>0$。RC correlation 只检查非冗余，不作为动作有效性的替代证据。

**最终状态：rejected extension，不进入正式 Method。** 两个独立 candidate-generation seeds 的 mean effect 均为正且相近，但 combined episode-bootstrap CI 对 dmotion correlation 与 command specificity 均跨零。该结果支持 real-transition observability，不支持 generated-candidate reward utility。公式与实现保留用于边界分析，正式 RC-GRPO 不包含 $R^{\mathrm{act}}$。

## 9. Implementation Contract

固定设置：RT-1 `fractal20220817`、compressive FSQ tokenizer、Llama world model、HF sampling `top_k=100`、$G=16$、$T=8$、30 post-training steps、batch windows 2、KL 0.001。所有候选在 matched comparisons 中使用相同 model seed、window split 和 generation protocol。

训练 reward 只使用 reachable-target MSE/LPIPS；raw-GT metrics、flow/dmotion、DINO-KID 和任何 distributional metric 均为 evaluation-only。
