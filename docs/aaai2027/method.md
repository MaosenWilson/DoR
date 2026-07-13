# Method 中文工作稿

> 与 `story.md` 同步。正式 Method 当前包含 RC verifier 与 Temporal-Return GRPO；Rank-Reliable Return 为预注册候选扩展，只有实验绿灯后才进入 tex。

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

### 3.4 Rejected target refinement: Metric-Refined Reachable Target

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

## 4. Temporal-Return GRPO

### 4.1 Sequence-level baseline

sequence-level GRPO 将有效 future frames 的 reward 平均为

$$
\bar R_i=\frac{1}{H'}\sum_{h\in\mathcal H}R^{\mathrm{RC}}_{i,h},
$$

并将同一个标准化优势广播给整条 trajectory。它不能区分 frame block 对后续 prediction errors 的延迟作用。

### 4.2 Frame-block reward-to-go

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

### 4.3 Policy objective

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

## 5. Rejected Extension: Rank-Reliable Temporal Return

> **Rejected：离线 replay gate 为 RED，不进入正式 Method/贡献；本节仅保留失败机制记录，Tex 应删除。**

### 5.1 Horizon reliability calibration

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

### 5.2 Reliability-shaped return

$$
G^{\mathrm{RR}}_{i,t}
=
\sum_{h=t}^{H}
\gamma^{h-t}\bar w_hR^{\mathrm{RC}}_{i,h},
$$

其余 group normalization 与 policy objective 不变。

### 5.3 退化与负对照

- $w_h$ 常数：退化为 plain Temporal Return；
- shuffled $w_h$：检验收益是否来自 reliability 与 horizon 的正确对应；
- reversed $w_h$：检验方向性；
- code distance 只作 calibration diagnostic，不直接加入训练 reward。

该权重把 pairwise sign reliability 映射为 horizon objective coefficient，是有解释的设计，但不是无偏 correction。若离线重放未通过，不实现训练分支。

## 6. Implementation Contract

固定设置：RT-1 `fractal20220817`、compressive FSQ tokenizer、Llama world model、HF sampling `top_k=100`、$G=16$、$T=8$、30 post-training steps、batch windows 2、KL 0.001。所有候选在 matched comparisons 中使用相同 model seed、window split 和 generation protocol。

训练 reward 只使用 reachable-target MSE/LPIPS；raw-GT metrics、flow/dmotion、DINO-KID 和任何 distributional metric 均为 evaluation-only。
