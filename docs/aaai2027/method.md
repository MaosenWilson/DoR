# Method 中文工作稿

本文只保留 Word 主线中的两个方法模块：可达性约束排序校准与视觉 token 块级 Temporal Return。
历史 reward panel、target search、action verifier、distribution reward、rank weighting 和通用 GRPO
替换均不属于当前 Method。

## 1. Problem Formulation

给定 context frames 与 actions

$$
q=(s_{1:C},a_{1:H}),
$$

冻结视觉 tokenizer 将图像编码为离散表示，世界模型 $\pi_\theta$ 自回归生成 $G$ 条 future-token
trajectories $o_i=(o_{i,1},\ldots,o_{i,H})$，冻结 decoder 得到 $\hat s_{i,h}=D(o_{i,h})$。
本文只更新 $\pi_\theta$，不训练 tokenizer 或 decoder。

原始 full-reference frame reward 为

$$
r^{\mathrm{raw}}_{i,h}=-d(\hat s_{i,h},s'_h),
\qquad
d(x,y)=\operatorname{MSE}(x,y)+\operatorname{LPIPS}(x,y).
$$

所有最终 evaluation 仍使用 raw $s'_h$。

## 2. Target-Set Mismatch as Rank Corruption

### 2.1 Codec-Reachable Set

冻结 tokenizer-decoder 的输出集合为

$$
\mathcal S_{\mathrm{codec}}=\{D(z):z\in\mathcal Z\}.
$$

raw target $s'_h$ 通常不严格属于该集合。它通过同一 codec 的可达重建为

$$
\tilde s'_h=D(Q(E(s'_h))),
$$

其中 $E$、$Q$ 和 $D$ 分别表示 encoder、quantizer 和 decoder。

### 2.2 Candidate-Dependent Residual Interaction

令 candidate 相对 reachable target 的误差为

$$
e_{i,h}=\hat s_{i,h}-\tilde s'_h,
$$

target reconstruction residual 为

$$
\delta_h=s'_h-\tilde s'_h.
$$

对平方误差有精确分解

$$
\|\hat s_{i,h}-s'_h\|_2^2
=\|e_{i,h}\|_2^2+\|\delta_h\|_2^2-2\langle e_{i,h},\delta_h\rangle.
$$

$\|\delta_h\|_2^2$ 对同组候选是常数，可被 group centering 消除；交互项
$-2\langle e_{i,h},\delta_h\rangle$ 随 candidate 改变，可能翻转 candidate pairs。对 LPIPS 等
非线性度量不声称该平方分解仍精确成立，而是通过 same-candidate rank audit 直接测量其影响。

## 3. Reachability-Constrained Rank Calibration

### 3.1 Reachability Audit

在任何训练前，对同一 candidate group 计算 raw 与 RC rewards。设两组分数与独立 decoder-input
representation readout 的相关性分别为 $\rho_{\mathrm{raw}}$ 与 $\rho_{\mathrm{RC}}$，并直接统计两种
target 对 candidate pairs 的 ordering disagreement。联合高斯假设下，相关分数的符号分歧概率为

$$
p_{\mathrm{flip}}=\frac{\arccos(\rho)}{\pi}.
$$

该关系是诊断模型，不作为本文原创定理。论文贡献在于把 target-set mismatch 操作化为 GRPO
实际消费的 rank corruption，并进行跨 codec/架构的同候选验证。

### 3.2 Reconstruction-Calibrated Verifier

RC verifier 保留原始 metric，只替换 target：

$$
r^{\mathrm{RC}}_{i,h}
=-d(\hat s_{i,h},\tilde s'_h).
$$

这不是减去常数重建残差（constant residual subtraction）。后者不会改变排序或 group-normalized advantage；RC 改变的是与每个
candidate 的比较参照，因此可以修复 candidate-dependent ordering。

### 3.3 Raw-Anchored Calibrated Update

RC target 与 raw evaluation 具有不同角色。为避免手工 reward 融合，对同一 candidates、同一
token log-probabilities 和同一 credit mode，分别构造 raw 与 RC 的 GRPO surrogate losses：

$$
g_0=\nabla_\theta\ell_{\mathrm{raw}},
\qquad
g_1=\nabla_\theta\ell_{\mathrm{RC}}.
$$

选择最接近 RC gradient、同时至少保持 raw-GRPO 一阶进度的方向：

$$
g^*=\arg\min_g\frac12\|g-g_1\|_2^2
\quad\mathrm{s.t.}\quad
\langle g,g_0\rangle\geq\|g_0\|_2^2.
$$

闭式解为

$$
g^*=g_1+\lambda^*g_0,
\qquad
\lambda^*=
\frac{[\|g_0\|_2^2-\langle g_1,g_0\rangle]_+}
{\|g_0\|_2^2+\epsilon}.
$$

若 RC direction 已满足 raw progress，$\lambda^*=0$；否则只加入满足约束所需的最小 raw-gradient
分量。KL regularization 只反向传播一次，不包含在两个 verifier gradients 中。

该约束只保证当前 sampled surrogate 的一阶几何，不保证有限步 AdamW 更新后的 held-out raw
LPIPS。因此跨平台 paired experiment 是必要证据。目前 RT-1 pilot 为正，普适 raw-GT gain 尚未
建立，论文必须保留这一边界。

## 4. Frame-Block Temporal-Return Credit Assignment

### 4.1 Sequence-Level Baseline

sequence-level GRPO 将有效 future-frame rewards 聚合为

$$
R_i^{\mathrm{seq}}=\frac{1}{|\mathcal H|}
\sum_{h\in\mathcal H}r_{i,h},
$$

再把同一 group-normalized advantage 广播给整个 future-token trajectory。

### 4.2 Temporal Return

对第 $t$ 个 future-frame token block，计算

$$
G_{i,t}=\sum_{h=t}^{H}\gamma^{h-t}r_{i,h}.
$$

每个时间位置只在同一条件下的 $G$ 个 candidates 之间标准化：

$$
A_{i,t}=
\frac{G_{i,t}-\mu_t}{\sigma_t+\epsilon},
\quad
\mu_t=\frac1G\sum_iG_{i,t},
\quad
\sigma_t^2=\frac1G\sum_i(G_{i,t}-\mu_t)^2.
$$

$A_{i,t}$ 只作用于第 $t$ 个 visual-token block。当前 RT-1 multi-step 接口中第一个 predicted frame
没有直接 verifier reward，但仍可通过后续 rewards 获得 temporal credit。

### 4.3 Policy Objective

令

$$
\ell_{i,t}=\sum_{k\in o_{i,t}}
\log\pi_\theta(o_{i,t,k}\mid o_{i,<t},q),
$$

则 Temporal-Return objective 为

$$
\mathcal L_{\mathrm{TR}}
=-\frac1G\sum_{i=1}^{G}\sum_{t=1}^{H}A_{i,t}\ell_{i,t}
+\beta_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
$$

正式实现固定 $\beta_{\mathrm{KL}}=0.001$，采用 RLVR-World/VERL 兼容的 sampled low-variance
KL。令 $k=\log\pi_{\mathrm{ref}}-\log\pi_\theta$，逐 token penalty 为

$$
\exp(k)-k-1.
$$

## 5. Combined RC-GRPO

多步完整流程依次执行：

1. 对同一 candidate rollouts 计算 raw 与 RC per-frame rewards；
2. 按相同 Temporal-Return rule 分别构造 raw 与 RC block-level advantages；
3. 计算两个 GRPO surrogate gradients并执行 raw-anchored projection；
4. 加入一次 reference-policy KL，完成 gradient clipping 和 optimizer update；
5. 始终相对 raw future frames 评测 LPIPS、MSE、PSNR 和 SSIM。

没有第三个 reward、rank weight、distribution term 或 optimizer replacement。

## 6. Implementation Contract

- candidate group：$G=16$，temperature 1.0，top-$k=100$；
- RT-1 multi-step：$T=8$，30 updates，batch windows 2，learning rate $10^{-5}$；
- temporal discount：$\gamma=0.95$；
- KL：coefficient 0.001，默认 `low_var_kl`（与 RLVR-World/VERL 对齐）；历史 linear-KL 可作并列/附录；
- checkpoint selection：默认 final step；允许在预声明评测时刻中择优，规则写入实验表注；
- paired comparison：尽量共享 split、seed、candidate schedule 和 evaluator；seed 不全时披露实际 \(n\)；
- final metrics：只对 raw GT，RC reconstruction 不进入 headline evaluation；
- logging：每步打印 elapsed/ETA，并记录约束触发率、gradient cosine 与 raw-progress ratio。

## 7. Method Boundary

当前 Method 不包含额外指标融合、pre-decode auxiliary reward、分布奖励或通用 GRPO optimizer
replacement。历史候选只在 `experiments.md` 的精简边界表中登记，不进入方法章节。
