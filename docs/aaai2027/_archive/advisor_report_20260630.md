# 阶段汇报：多域可靠奖励校准用于 tokenized 视频世界模型 RLVR（2026-06-30）

> 目标：回应 6/27 会议中老师的核心关注：  
> 1. 解码前、解码后到底分别算什么 reward，不能混乱；  
> 2. code 层是特征层，不能误说“在 code 层算 LPIPS”；  
> 3. 计算出多域、多粒度误差后，直接交给 GRPO 会有什么问题，我们针对这个问题怎么设计；  
> 4. 论文需要 2-3 个清楚贡献，而不是只说 code reward 更好。

---

## 0. 先把之前容易引起误会的地方改正

我之前跟老师说过“在 code 层面算 LPIPS”，这个表述是不准确的，需要纠正。

**LPIPS 不是 code 空间的指标。** LPIPS 的计算路径是：

$$
\text{image}
\rightarrow
\text{VGG feature extractor}
\rightarrow
\text{feature distance}.
$$

也就是说，LPIPS 虽然是“特征层”感知距离，但它的输入仍然是图像。因此在 tokenized 视频世界模型里，只要用 LPIPS，就必须先把预测 token 经过 decoder 变成图像：

$$
o_i
\xrightarrow{\mathrm{decode}}
\hat s'_i
\xrightarrow{\mathrm{LPIPS}}
d(\hat s'_i,s').
$$

所以 LPIPS 属于：

> **解码后 post-decode 的特征感知指标**。

它不是：

> **解码前 pre-decode 的 token/code 特征指标**。

因此我现在的准确表述是：

> code 层 reward 不是 LPIPS，而是 tokenizer code/latent feature 空间的 full-reference 距离；LPIPS 是 decoder 后图像上的感知 reward 或 evaluator 指标。

---

## 1. 对 RLVR-World 原始 reward 的代码核实

老师在会议里问到：原文图中写了 “MSE/LPIPS/SSIM/...”，但实际代码是不是只用了 MSE 和 LPIPS？

我核实了服务器上的 RLVR-World video world model trainer。训练 reward 的核心逻辑是：

```python
if reward_fn == "mse":
    recon_loss = mean((real - pred)**2)
elif reward_fn == "mae":
    recon_loss = mean(abs(real - pred))

perceptual_loss = LPIPS(real, pred)

loss = recon_loss + perceptual_loss
reward = -loss
```

因此忠实 RLVR-World 训练 reward 是：

$$
R_i^{\mathrm{RLVR}}
=
-
\left[
d_{\mathrm{recon}}(\hat s'_i,s')
+
\mathrm{LPIPS}_{\mathrm{vgg}}(\hat s'_i,s')
\right],
$$

其中：

$$
d_{\mathrm{recon}}
=
\mathrm{MSE}
\quad\text{或}\quad
\mathrm{MAE}.
$$

SSIM、PSNR 等确实存在于 `Evaluator` 中，但主要用于 evaluation / logging，不是训练 reward 的组成项。

因此本文后续明确区分：

| 项目 | 在 RLVR 中的角色 |
|---|---|
| MSE/MAE | 训练 reward 的 reconstruction loss |
| LPIPS | 训练 reward 的 perceptual loss |
| SSIM/PSNR | evaluator / logging 指标 |
| 图中 “MSE/LPIPS/SSIM/...” | 泛指 task-specific verifiable metrics，不等于全部进入 reward |

---

## 2. 当前我们不再讲“code 是更好的 reward”

之前的故事有风险：如果只说 code reward 更好，会有两个问题。

第一，实验证据不够稳。早期 code 在 flow 上明显优于 pixel，但后续更完整实验显示 flow 排序不稳定；FD-DINO/KID 这类分布特征评测里，code 也没有稳定赢 pixel/pixel_tok。

第二，贡献太单薄。单独说“换成 code reward”容易被审稿人认为是一个 trick，而且 code 与 tokenizer 同源，容易被质疑 home advantage。

因此现在的论文主线改为：

> 在 tokenized video RLVR 中，verifiable reward 需要做 **reliability calibration**。我们不是简单换一个指标，而是分析 decoder 后 reward 为什么不可靠，并设计解码前、解码后、动态残差和 GRPO 可靠性加权的整体方案。

---

## 3. 当前完整 pipeline

### 3.1 不改 RLVR-World 主架构

输入：

$$
q=(s,a),
$$

其中 \(s\) 是 context frames，\(a\) 是动作。

世界模型采样一组候选 token：

$$
o_1,\ldots,o_G
\sim
\pi_\theta(\cdot|q).
$$

候选 token 可以解码成图像：

$$
\hat s'_i
=
\mathrm{decode}(o_i).
$$

每个候选得到一个标量 reward：

$$
R_i.
$$

GRPO 做组内归一化：

$$
A_i
=
\frac{R_i-\mathrm{mean}_{j}R_j}
{\mathrm{std}_{j}R_j+\epsilon}.
$$

策略梯度：

$$
\mathcal L_{\mathrm{PG}}
=
-\frac1G
\sum_{i=1}^G
A_i\log\pi_\theta(o_i|q).
$$

这里需要跟老师明确：

> GRPO 不是只挑 reward 最大的候选做 SFT；所有候选都会参与更新。高于组均值的候选提高概率，低于组均值的候选降低概率。

所以 reward 的**组内排序和相对间隔**直接决定 GRPO 更新方向。

---

## 4. 我们现在设计的 reward 分成三类

“解码前、解码后分别用什么要定义好”。我现在把它拆成三块。

---

### 4.1 解码前：tokenizer code/latent feature reward

世界模型输出的是 visual token。每个 token 可以映射到 FSQ code 向量：

$$
z_i
=
\mathrm{indices\_to\_codes}(o_i).
$$

真值帧也可先 encode，再映射到 code：

$$
z'
=
\mathrm{indices\_to\_codes}(\mathrm{encode}(s')).
$$

解码前 reward：

$$
R_i^{\mathrm{code}}
=
-
\mathrm{RMS}(z_i-z').
$$

这里 \(z_i,z'\) 是 tokenizer 的 code/latent feature，因此这是：

> pre-decode tokenizer-feature full-reference reward。

它不是 LPIPS，也不是 DINO；它是 tokenizer 自身 latent/code 空间的距离。

这么做的动机是：

1. 世界模型本来就输出 token，code reward 与输出空间直接对齐；
2. 不经过 decoder，因此不继承 decoder 的重建误差；
3. 作为对 post-decode reward 的补充监督。

因此，当前不再将“code 是唯一更好的 reward”作为主张。`code` 只是可靠奖励设计中的一个 pre-decode 分量。

---

### 4.2 解码后：pixel/perceptual reward，但目标改成 tokenizer 可达目标

原 RLVR reward 是：

$$
R_i^{\mathrm{RLVR}}
=
-
\left[
\mathrm{MSE}(\hat s'_i,s')
+
\mathrm{LPIPS}(\hat s'_i,s')
\right].
$$

问题是 tokenized model 即使预测完美 token，也只能解码到：

$$
\tilde s'
=
\mathrm{decode}(\mathrm{encode}(s')),
$$

而不是原始 \(s'\)。因此直接对原始 \(s'\) 算 post-decode reward，会把 tokenizer decoder 的不可约误差也算进候选误差里。

我们定义解码后 floor-cancelled reward：

$$
R_i^{\mathrm{pix\_tok}}
=
-
\mathrm{LPIPS}
(\hat s'_i,\tilde s').
$$

更忠实地对齐 RLVR 原始形式，也可以写成：

$$
R_i^{\mathrm{RLVR\_tok}}
=
-
\left[
\mathrm{MSE}(\hat s'_i,\tilde s')
+
\mathrm{LPIPS}(\hat s'_i,\tilde s')
\right].
$$

这里的关键不是“LPIPS 不准确”，而是：

> LPIPS 是合理的 perceptual metric，但作为 tokenized video RLVR 的训练 reward 时，它位于 decoder 后，因此会继承 tokenizer decoder 的 reconstruction floor。把目标换成 \(\tilde s'\) 可以让完美 token 预测得到真正的满分。

---

### 4.3 动态残差：补充动作导致的变化信号

只做单帧保真 reward 容易被静态外观主导。为了让 reward 更关注动作导致的变化，我们在 code 空间定义 motion residual：

$$
\Delta z_i
=
z_i-z_t,
\qquad
\Delta z'
=
z'-z_t,
$$

其中 \(z_t\) 是当前 context 最后一帧的 code。

不能简单用：

$$
\|\Delta z_i-\Delta z'\|
$$

因为它会退化成：

$$
\|z_i-z'\|.
$$

所以我们使用方向和幅度：

$$
R_i^{\mathrm{dyn}}
=
\cos(\Delta z_i,\Delta z')
-
\gamma
\left|
\log
\frac{\|\Delta z_i\|+\epsilon}
{\|\Delta z'\|+\epsilon}
\right|.
$$

最终可构造：

$$
R_i^{\mathrm{pixel\_tok\_dyn}}
=
z(R_i^{\mathrm{pix\_tok}})
+
\lambda z(R_i^{\mathrm{dyn}}),
$$

或：

$$
R_i^{\mathrm{code\_dyn}}
=
z(R_i^{\mathrm{code}})
+
\lambda z(R_i^{\mathrm{dyn}}).
$$

这部分的定位是：

> 在 pre-decode code feature domain 中补充一个与静态外观更正交的动态监督。

---

## 5. 一堆 reward 直接交给 GRPO 会有什么问题？

这是当前方案的关键问题：

我们不是简单计算很多误差然后直接相加丢给 GRPO。因为这样会有三个问题。

### 5.1 问题一：不同域的误差尺度不同

例如：

- code RMS；
- MSE；
- LPIPS；
- motion residual cosine；

这些量纲和范围都不同。直接相加会让某个数值尺度大的项主导，而不是让真正可靠的项主导。

### 5.2 问题二：不同域的可靠性不同

post-decode LPIPS/MSE 经过 decoder；pre-decode code 不经过 decoder。它们受 tokenizer floor 的程度不同。

我们定义 reward noise floor：

$$
\phi_{\mathrm{tok}}^{(d)}
=
\mathbb E_{s'}
\left[
d(
\mathrm{decode}(\mathrm{encode}(s')),
s')
\right].
$$

LPIPS/SSIM 这类指标的 floor 更明显；MSE 小一些；code 基本不经过 decoder floor。

所以不同 reward 分量不能默认同等可靠。

### 5.3 问题三：GRPO 消费的是组内排序

GRPO 不是优化 reward 的绝对值，而是通过：

$$
A_i
=
\frac{R_i-\mathrm{mean}(R)}
{\mathrm{std}(R)+\epsilon}
$$

使用组内相对排序。

如果某个 reward 分量带有较大噪声，它可能翻转候选排序，导致 GRPO 对错误候选增加概率。

我们已有理论和实证：

$$
P_{\mathrm{flip}}
=
\frac{\arccos(\rho)}{\pi}.
$$

5 seed 下实测 flip rate 与公式高度吻合。这说明：

> reward 设计不是只追求更多指标，而是要保证组内排序可靠。

---

## 6. 我们针对这个问题的 GRPO 设计

针对“直接给 GRPO 会有什么问题，以及如何改”的问题，当前方案分为两个层次。

### 6.1 第一层：reward 先做可靠性校准再融合

不是直接：

$$
R_i
=
R_i^{code}
+
R_i^{mse}
+
R_i^{lpips}.
$$

而是先尺度归一、再按可靠性加权：

$$
R_i
=
-
\sum_{m\in\mathcal M}
w_m
\frac{d_m(i)}{s_m}.
$$

其中：

- \(s_m\)：离线估计的尺度；
- \(w_m\)：由 floor / rank reliability / 实验验证确定；
- \(\mathcal M\)：pre-decode code、post-decode pixel/perceptual、motion residual 等。

需要诚实说明：早期 `dorw` 地板加权融合没有稳定超过 code，说明“融合”本身不是天然贡献。当前融合必须服务于可靠性校准，而不是为了堆指标。

### 6.2 第二层：GRPO advantage 做 rank-reliability weighting

对于 reward 间隔很小的候选，其排序可能只是噪声造成的。我们给 advantage 加一个局部排序可信度：

$$
c_{ij}
=
2\Phi
\left(
\frac{|R_i-R_j|}
{\sqrt{2}\sigma_\eta}
\right)
-1.
$$

候选 \(i\) 的权重取相邻排序中最不可靠的 gap：

$$
w_i
=
\min_{j\in \mathcal N(i)}
c_{ij}.
$$

然后：

$$
\tilde A_i
=
w_i A_i,
\qquad
\tilde A_i
\leftarrow
\tilde A_i
-
\frac1G
\sum_j
\tilde A_j.
$$

这样做的直觉是：

- reward gap 大：排序可信，保留更新；
- reward gap 小：可能是噪声，不让它强烈指导 policy；
- 不硬删样本，不破坏 GRPO 主流程。

这部分可以作为我们的 GRPO 改进点：

> floor-aware / rank-reliability GRPO。

---

## 7. 目前已经验证的东西

### 7.1 已经比较稳的验证

**验证 1：RLVR 训练本身有效。**  
FD-DINO/KID 初步评估显示所有 RL 训练后的模型都明显好于 base。这说明在这个框架下，GRPO 微调是有效的。

**验证 2：reward noise floor 存在且度量依赖。**  
tokenizer encode-decode 后与原 GT 的差异在 LPIPS/SSIM/MSE 等度量上均非零。感知/结构指标的 floor 更明显。

**验证 3：floor 会破坏组内排序。**  
rank flip 理论：
$$
P_{\mathrm{flip}}
=
\frac{\arccos(\rho)}{\pi}
$$

与 5 seed 实测高度吻合。这是目前最硬的理论和实证资产。

**验证 4：pixel_tok 的负对照支持 floor-cancellation。**  
LPIPS floor 大时，`pixel -> pixel_tok` 的动态指标提升更稳定；MSE floor 小时，`mse -> mse_tok` 增益弱且不稳定。说明收益确实和 floor 有关。

### 7.2 正在验证的东西

**动态残差 reward。**  
当前 `dyn_lambda=0.25` 的实验表明：

- `pixel_tok_dyn` 比 `pixel_tok` 在 flow 上多数 seed 提升；
- dmotion 5/5 seed 提升；
- `code_dyn` 比 `code` 也多数 seed 提升 flow；
- 但 LPIPS/PSNR/SSIM 明显变差。

因此当前结论不是“最终方法已成功”，而是：

> motion residual 是有效动态信号，但 \(\lambda=0.25\) 太强，需要降低到 0.10 或 0.05 找 Pareto 点。

---

## 8. 当前想好的 2-3 个贡献

### 贡献一：定义并刻画 tokenized video RLVR 中的 reward noise floor

我们指出，RLVR-World 的 verifiable reward 在视频 tokenized world model 中并不天然 clean。只要 reward 在 decoder 后图像上计算，就会继承 tokenizer decoder 的不可约重建误差：

$$
\phi_{\mathrm{tok}}^{(d)}
=
\mathbb E
\left[
d(
\mathrm{decode}(\mathrm{encode}(s')),
s')
\right].
$$

这个概念不是已有官方术语，而是本文针对 tokenized RLVR 场景定义的问题。

### 贡献二：解释该 floor 如何破坏 GRPO 的组内排序

把 reward 写成：

$$
R_i=R_i^\star+\eta_i.
$$

我们证明并实测验证，reward noise 会造成 pairwise rank flip：

$$
P_{\mathrm{flip}}
=
\frac{\arccos(\rho)}{\pi}.
$$

由于 GRPO 使用组内归一化 advantage，这种 rank corruption 会直接影响 policy update。

这解决老师说的“直接丢给 GRPO 会有什么问题”：问题不是 reward 绝对值，而是组内相对排序会被不同可靠性的误差项污染。

### 贡献三：提出多域可靠奖励校准，而不是简单多指标相加

方法包括：

1. **pre-decode code reward**：在 tokenizer code/feature 空间直接比较 \(z_i,z'\)，避免 decoder floor；
2. **post-decode reachable-target reward**：将目标从 \(s'\) 改成 \(\tilde s'=\mathrm{decode}(\mathrm{encode}(s'))\)，减少 decoder floor 对 LPIPS/MSE reward 的污染；
3. **motion-residual reward**：在 code 空间比较 \(\Delta z_i,\Delta z'\)，补充动作导致的动态变化监督；
4. **rank-reliability GRPO**：对 reward 间隔小、排序不可靠的候选降低 advantage 权重。

如果需要压成三条贡献，可以写成：

1. reward noise floor；
2. rank corruption mechanism；
3. reliability-calibrated reward and GRPO。

---

## 9. 需要明确的几个关键表述

### 9.1 关于“code 层是否应该用特征评价指标”

之前“code 层算 LPIPS”的表述是不准确的。LPIPS 虽然是 VGG 特征距离，但输入是图像，因此它属于 decoder 后的感知指标。当前 code 层计算的是 tokenizer 自身 code/latent feature 的 full-reference 距离：

$$
R^{code}_i
=
-\mathrm{RMS}(z_i-z').
$$

它训练时是 pre-decode feature reward；但评估时不能只用 code RMS，而要用 LPIPS/PSNR/SSIM、flow、FD-DINO/KID 等外部指标验证，避免同源循环。

### 9.2 关于“LPIPS 是否不准确”

这里不是说 LPIPS 不准确。LPIPS 是合理的 perceptual metric，也可以作为 evaluator。但在 tokenized world model 的 training reward 中，它位于 decoder 后，因此会继承 decoder 的 reconstruction floor。当前观点是：LPIPS 作为 post-decode reward 需要校准，尤其要考虑 tokenizer 可达目标，而不是直接对原始 GT。

### 9.3 关于“为什么要解码前和解码后都算”

解码后 reward 直观，对图像保真有意义，但会受 decoder floor 影响；解码前 code reward 与模型输出空间直接对齐，不受 decoder floor 影响，但可能缺少图像感知语义。因此当前不是二选一，而是把它们作为多域可靠监督：pre-decode 负责 token/code 对齐，post-decode 负责 perceptual/pixel fidelity，motion residual 负责动态变化。

### 9.4 关于“这么多误差直接给 GRPO 有什么问题”

问题在于 GRPO 消费的是组内排序。多域误差尺度不同、可靠性不同，直接相加会让高噪声分量污染排序。当前方案用 reward floor 和 rank flip 理论解释这个问题，再通过尺度归一、floor-cancelled target、motion residual、rank-reliability advantage weighting 来解决。

---

## 10. 下一步需要讨论的问题

### 问题一：最终方法主线是否定为“reliability-calibrated reward”

不再把“code reward 更好”作为主线，而是把 code、pixel_tok、dyn、rankrel 都放到可靠奖励校准框架下。

### 问题二：动态残差是否值得作为第三贡献的一部分

当前 \(\lambda=0.25\) 已证明动态残差能提高 flow/dmotion，但牺牲保真。下一步我计划跑：

$$
\lambda=0.10
$$

如果能在较小 LPIPS 损失下保持 flow 提升，就可以作为方法贡献；否则作为机制和消融。

### 问题三：GRPO 改进采用 rank-reliability 还是只作为 ablation

rankrel 与 C2 理论最对应，但是否提升还需要实验。如果有效，则作为 GRPO 可靠性设计；如果无效，则如实写成负结果，说明主要收益来自 reward target/domain 设计。

### 问题四：多域 reward 融合怎么写得不弱

不能写成简单加权求和。应写成：

> 多域 reward 的问题不是“项越多越好”，而是“各项可靠性不同，直接交给 GRPO 会污染组内排序”。我们的融合必须以 reliability calibration 为核心。

---

## 11. 简短总结

当前已将之前容易混乱的地方重新梳理清楚。RLVR 原训练 reward 实际是 recon loss 加 LPIPS，SSIM/PSNR 主要是 evaluator。LPIPS 虽然是特征感知指标，但输入是图像，所以属于 decoder 后指标；“code 层算 LPIPS”是不准确的表述。当前设计是：解码前用 tokenizer code/latent feature 距离，解码后用 LPIPS/MSE 但目标改成 tokenizer 可达重建，另外加 code 空间 motion residual 作为动态监督。核心问题不是简单多算几个误差，而是这些误差尺度和可靠性不同，直接交给 GRPO 会污染组内排序；当前方案用 reward noise floor 和 rank flip 理论解释这个问题，再通过 floor-cancelled target、motion residual、rank-reliability GRPO 做校准。现在已验证 floor 和 rank flip 机制，`pixel_tok` 负对照支持去地板，motion residual 初步能提高动态指标但需要调小权重。下一步重点需要确认：`reliability-calibrated reward` 是否作为主线，以及动态残差和 `rankrel` 哪个更适合作为第三贡献。
