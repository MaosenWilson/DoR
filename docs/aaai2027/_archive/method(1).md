# 方法草稿：Codec-Calibrated Reward 与 Codec-Aware Spatial-Temporal GRPO

> 本文档用于梳理论文方法部分的核心思路。当前重点不是设计一个复杂 reward，而是利用一个相对简单、可解释的 **codec-calibrated reward**，支撑后续 **GRPO 改进**。因此，reward 设计在本文中主要作为必要的信号校准模块，方法重点放在 **统一单步/多步的 codec-aware segment-level GRPO**。

---

## 1. 问题背景与动机

在 RLVR-World 风格的视频世界模型后训练中，模型通常先预测视觉 token，再通过 visual decoder 解码成图像帧，然后用解码后的预测帧与 ground-truth 帧计算 LPIPS、MSE 等指标作为 reward。

原始 reward 可以写成：

\[
R = - \left(
 w_L \operatorname{LPIPS}(\hat{x}_{t+1}, x_{t+1})
 +
 w_M \operatorname{MSE}(\hat{x}_{t+1}, x_{t+1})
\right).
\]

其中：

- \(x_{t+1}\)：GT 下一帧；
- \(\hat{x}_{t+1}\)：模型预测 token 解码后的帧；
- \(w_L,w_M\)：LPIPS 与 MSE 的权重。

但是，对于 tokenized video world model，即使直接把 GT 帧编码再解码：

\[
\tilde{x}_{t+1}=D(E(x_{t+1})),
\]

也会和原始 GT 帧存在非零误差。例如当前观察到：

\[
\operatorname{LPIPS}(D(E(x_{t+1})),x_{t+1}) \approx 0.112.
\]

这说明 decoded-frame reward 中包含一个由视觉 tokenizer/decoder 引入的不可避免误差。本文中更建议称其为：

- **codec reconstruction floor**；
- **decoder-induced reward floor**；
- **codec-induced reward bias**。

不建议在论文中直接称为 noise，因为它并不一定是随机噪声，而更像是视觉 codec 的系统性重建下限。

本文方法解决两个相关问题：

1. **Reward 校准问题**：decoded LPIPS/MSE 会惩罚 tokenizer/decoder 自身带来的重建下限；
2. **GRPO credit assignment 问题**：标准 GRPO 用一个 rollout-level scalar reward 监督整段视觉 token，无法区分不同时间步、不同空间区域的局部预测质量。

因此，本文提出一个统一框架：

\[
\boxed{
\text{Codec-Calibrated Token-Visual Reward}
+
\text{Codec-Aware Spatial-Temporal GRPO}
}
\]

其中，reward 设计服务于 GRPO 改进，不作为本文唯一或主要创新点。

---

## 2. 统一符号定义

设时间 \(t\) 的输入上下文为：

\[
c_t=(x_{\leq t},a_{\leq t}),
\]

其中：

- \(x\)：视觉观测；
- \(a\)：动作或控制输入。

策略模型从旧策略 \(\pi_{\theta_{\text{old}}}\) 采样未来 \(H\) 步视觉 token：

\[
\hat{z}_{i,1:H}
\sim
\pi_{\theta_{\text{old}}}(\cdot\mid c_t),
\]

其中：

- \(i\in\{1,\dots,G\}\)：同一输入下第 \(i\) 个 rollout；
- \(G\)：GRPO group size；
- \(H\)：预测 horizon；
- \(H=1\)：单步单帧预测；
- \(H>1\)：多步视频预测；
- \(\hat{z}_{i,h}\)：第 \(i\) 个 rollout 在未来第 \(h\) 步预测的 visual token grid。

GT token 为：

\[
z^*_h=E(x_{t+h}),
\]

其中 \(E\) 是 visual tokenizer encoder。

预测 token 解码为图像帧：

\[
\hat{x}_{i,t+h}=D(\hat{z}_{i,h}),
\]

其中 \(D\) 是 visual tokenizer decoder。

每一帧被划分为 \(K\) 个空间 segment：

\[
\mathcal{S}_{h,1},\mathcal{S}_{h,2},\dots,\mathcal{S}_{h,K}.
\]

每个 reward 单元由一个时间-空间二元组表示：

\[
(h,k),
\]

其中：

- \(h\in\{1,\dots,H\}\)：未来预测步；
- \(k\in\{1,\dots,K\}\)：空间 segment。

单步预测是该统一表达的特例：

\[
H=1.
\]

多步预测是一般形式：

\[
H>1.
\]

---

## 3. Reward 设计：仅作为校准模块

本节 reward 设计不是本文重点，而是为了缓解 decoded-frame reward 的 codec reconstruction floor，并为后续 GRPO 提供更可靠的 segment-level reward。

### 3.1 Codec reconstruction floor

对于每个 GT 未来帧，先计算 tokenizer 重建结果：

\[
\tilde{x}_{t+h}=D(E(x_{t+h})).
\]

第 \((h,k)\) 个时间-空间 segment 的 codec reconstruction floor 定义为：

\[
b_{h,k}
=
d(\tilde{x}_{t+h,k},x_{t+h,k}),
\]

其中 \(d(\cdot,\cdot)\) 可以是 LPIPS、MSE，或者二者加权组合。

如果同时使用 LPIPS 和 MSE，则：

\[
b_{h,k}=w_L b^L_{h,k}+w_M b^M_{h,k},
\]

其中：

\[
b^L_{h,k}=\operatorname{LPIPS}(\tilde{x}_{t+h,k},x_{t+h,k}),
\]

\[
b^M_{h,k}=\operatorname{MSE}(\tilde{x}_{t+h,k},x_{t+h,k}).
\]

由于 LPIPS、MSE、reward 标准差的尺度不同，实际训练中不建议直接使用原始 \(b_{h,k}\)，而应归一化：

\[
\tilde{b}_{h,k}
=
\frac{b_{h,k}}{\operatorname{mean}(b)+\epsilon}.
\]

也可以使用更鲁棒的归一化方式：

\[
\tilde{b}_{h,k}
=
\frac{b_{h,k}-\operatorname{median}(b)}{\operatorname{MAD}(b)+\epsilon}.
\]

### 3.2 Decoder 后校准 reward

模型预测帧和 GT 的原始 decoded distance 为：

\[
d^{\text{raw}}_{i,h,k}
=
d(\hat{x}_{i,t+h,k},x_{t+h,k}).
\]

为了去除 codec reconstruction floor，定义 residual distance：

\[
\Delta d_{i,h,k}
=
\left[d^{\text{raw}}_{i,h,k}-b_{h,k}\right]_+.
\]

对应的 decoded reward 为：

\[
r^{\text{dec}}_{i,h,k}
=
-\Delta d_{i,h,k}.
\]

如果 LPIPS 和 MSE 都参与 reward：

\[
r^{\text{dec}}_{i,h,k}
=
-\left[
 w_L\left(\operatorname{LPIPS}_{i,h,k}-b^L_{h,k}\right)_+
 +
 w_M\left(\operatorname{MSE}_{i,h,k}-b^M_{h,k}\right)_+
\right].
\]

这里的作用是：如果模型已经预测出与 GT token 非常接近的 token，则不再因为 visual codec 自身的重建下限继续惩罚模型。

### 3.3 Decoder 前 token-space reward

为了减少完全依赖 decoded RGB/frame space 的问题，引入 decoder 前 token-space reward。

若使用 codebook embedding distance：

\[
r^{\text{tok}}_{i,h,k}
=
-
\frac{1}{|\mathcal{S}_{h,k}|}
\sum_{p\in\mathcal{S}_{h,k}}
\left\|
C_{\hat{z}_{i,h,p}}
-
C_{z^*_{h,p}}
\right\|_2^2,
\]

其中：

- \(p\)：segment 内 visual token 的索引；
- \(C_j\)：codebook 中第 \(j\) 个 token 的 embedding；
- \(\hat{z}_{i,h,p}\)：预测 token index；
- \(z^*_{h,p}\)：GT token index。

也可以用 exact-match reward 作为 ablation：

\[
r^{\text{tok-match}}_{i,h,k}
=
\frac{1}{|\mathcal{S}_{h,k}|}
\sum_{p\in\mathcal{S}_{h,k}}
\mathbf{1}[\hat{z}_{i,h,p}=z^*_{h,p}].
\]

建议主方法使用 codebook embedding distance，exact-match 仅作为对照。

### 3.4 Token-visual reward 融合

定义 decoded reward 的可信度：

\[
q_{h,k}
=
\exp(-\alpha \tilde{b}_{h,k}),
\]

其中 \(\alpha>0\) 控制对 codec floor 的敏感程度。

直观解释：

- \(\tilde{b}_{h,k}\) 小：codec reconstruction floor 小，decoded LPIPS/MSE 更可信，\(q_{h,k}\) 较大；
- \(\tilde{b}_{h,k}\) 大：codec reconstruction floor 大，decoded reward 可信度降低，\(q_{h,k}\) 较小，更依赖 token-space reward。

最终 fused reward 定义为：

\[
\boxed{
r_{i,h,k}
=
(1-q_{h,k})r^{\text{tok}}_{i,h,k}
+
q_{h,k}r^{\text{dec}}_{i,h,k}
}
\]

这个 reward 只是后续 GRPO 的输入信号。本文重点不是声称 reward 本身非常复杂，而是使用该 reward 支撑更合理的 segment-level policy optimization。

---

## 4. GRPO 改进：Codec-Aware Spatial-Temporal GRPO

### 4.1 标准 GRPO 的局限

标准 GRPO 对第 \(i\) 个 rollout 计算一个 scalar reward：

\[
R_i.
\]

然后在 group 内归一化：

\[
A_i
=
\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
\]

该 advantage 会被分配给整个 rollout 中所有生成 token。

对于视频世界模型，这种做法过于粗糙。原因包括：

1. 单帧中只有局部区域预测差时，所有视觉 token 都会收到同样惩罚；
2. 多步预测中，某个未来步出错时，不应把相同 credit 分配给所有时间步；
3. 不同空间区域或时间步的 codec reconstruction floor 不同，decoded reward 的可靠性也不同。

因此，需要从 rollout-level advantage 改为 spatial-temporal segment-level advantage。

### 4.2 Segment-level advantage

对于每个时间-空间 segment \((h,k)\)，基于 \(G\) 个 rollout 的 segment reward 计算 group mean：

\[
\mu_{h,k}
=
\frac{1}{G}\sum_{i=1}^{G}r_{i,h,k},
\]

以及 group standard deviation：

\[
\sigma_{h,k}
=
\operatorname{Std}(r_{1,h,k},\dots,r_{G,h,k}).
\]

普通 segmental GRPO 可以写成：

\[
A^{\text{seg}}_{i,h,k}
=
\frac{r_{i,h,k}-\mu_{h,k}}{\sigma_{h,k}+\epsilon}.
\]

本文进一步引入 codec-aware denominator：

\[
\boxed{
A^{\text{seg}}_{i,h,k}
=
\frac{r_{i,h,k}-\mu_{h,k}}
{\sigma_{h,k}+\gamma \tilde{b}_{h,k}+\epsilon}
}
\]

其中 \(\gamma\geq 0\) 控制 codec reconstruction floor 对 advantage 缩放的影响。

该项的含义是：

> 当某个 segment 的 codec reconstruction floor 较大时，该 segment 的 decoded reward 更不可靠，因此它对策略更新的影响应被适度降低。

特殊情况：

- \(\gamma=0\)：退化为普通 segment-level GRPO；
- \(K=1,H=1\)：退化为单步 full-frame GRPO；
- \(\lambda=0\) 时进一步退化为标准 rollout-level GRPO。

### 4.3 Global advantage

纯 segment-level advantage 可能过度关注局部 patch，而忽视整帧或整段 rollout 的全局一致性。因此保留一个 global rollout reward：

\[
R_i
=
\sum_{h=1}^{H}
\omega_h
\frac{1}{K}
\sum_{k=1}^{K}r_{i,h,k}.
\]

其中 \(\omega_h\) 是时间权重，可用 discount：

\[
\omega_h=\delta^{h-1},
\]

其中 \(\delta\in(0,1]\)。

global advantage 为：

\[
A^{\text{global}}_i
=
\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
\]

### 4.4 最终 blended advantage

最终给第 \((h,k)\) 个 segment 使用的 advantage 为：

\[
\boxed{
\hat{A}_{i,h,k}
=
\lambda A^{\text{seg}}_{i,h,k}
+
(1-
\lambda)A^{\text{global}}_i
}
\]

其中 \(\lambda\in[0,1]\)：

- \(\lambda=0\)：只使用 global advantage，即标准 GRPO 风格；
- \(\lambda=1\)：只使用 segment-level advantage；
- \(0<\lambda<1\)：同时保留局部 credit assignment 和全局一致性。

建议初始使用：

\[
\lambda=0.7.
\]

随后做 ablation：

\[
\lambda\in\{0,0.3,0.5,0.7,1.0\}.
\]

---

## 5. 最终 GRPO Objective

对于第 \(i\) 个 rollout 中第 \(t\) 个生成 token，定义 policy ratio：

\[
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(\hat{z}_{i,t}\mid c_t,\hat{z}_{i,<t})
}{
\pi_{\theta_{\text{old}}}(\hat{z}_{i,t}\mid c_t,\hat{z}_{i,<t})
}.
\]

记第 \(t\) 个 token 所属的时间-空间 segment 为：

\[
m(t)=(h(t),k(t)).
\]

则该 token 使用的 advantage 为：

\[
\hat{A}_{i,m(t)}=\hat{A}_{i,h(t),k(t)}.
\]

最终 clipped GRPO loss：

\[
\boxed{
\mathcal{L}_{\text{CAST-GRPO}}
=
-
\mathbb{E}_{i,t}
\left[
\min
\left(
\rho_{i,t}(\theta)\hat{A}_{i,m(t)},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon_c,1+\epsilon_c)
\hat{A}_{i,m(t)}
\right)
\right]
+
\beta_{\text{KL}}
D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})
}
\]

其中：

- \(\epsilon_c\)：PPO/GRPO clipping range；
- \(\beta_{\text{KL}}\)：KL penalty 系数；
- \(\pi_{\text{ref}}\)：reference policy。

该目标保留了 GRPO 的 critic-free 特性，同时把 advantage 从 rollout-level 扩展到 codec-aware spatial-temporal segment-level。

---

## 6. 单步与多步的统一表达

### 6.1 单步单帧预测

单步情况下：

\[
H=1.
\]

则：

\[
r_{i,h,k}\rightarrow r_{i,1,k},
\]

\[
\hat{A}_{i,h,k}\rightarrow \hat{A}_{i,1,k}.
\]

此时方法退化为：

\[
\text{Codec-Aware Spatial Segmental GRPO}.
\]

主要用于验证：

1. codec-calibrated reward 是否改善 next-frame prediction；
2. token-space reward 是否能补充 decoded LPIPS/MSE；
3. spatial segment-level advantage 是否改善局部视觉 token 的 credit assignment。

### 6.2 多步视频预测

多步情况下：

\[
H>1.
\]

此时每个 rollout 包含多个未来步：

\[
r_{i,h,k},\quad h=1,\dots,H,\quad k=1,\dots,K.
\]

每个生成 token 根据其所属的时间步和空间 segment 使用对应 advantage：

\[
\hat{A}_{i,h(t),k(t)}.
\]

此时方法为：

\[
\text{Codec-Aware Spatial-Temporal Segmental GRPO}.
\]

主要用于验证：

1. 单步改进是否迁移到多步 rollout；
2. 多步误差累积是否降低；
3. temporal consistency 是否提升；
4. world model 的长 horizon 预测是否更稳定。

---

## 7. 训练与实现伪代码

```python
for batch in dataloader:
    context, gt_frames = batch

    # 1. 对同一个 context 采样 G 个 rollout
    rollouts = []
    logprobs_old = []
    for i in range(G):
        z_pred, old_logprob = sample_model(
            policy_old,
            context,
            horizon=H,
        )
        rollouts.append(z_pred)
        logprobs_old.append(old_logprob)

    # 2. 编码 GT frames，并通过 visual codec 重建
    z_gt = encoder(gt_frames)
    x_recon_gt = decoder(z_gt)

    # 3. 计算 codec reconstruction floor: b[h, k]
    b = compute_codec_floor(
        x_recon_gt,
        gt_frames,
        segments,
    )
    b_norm = normalize_floor(b)

    # 4. 计算每个 rollout 的 fused reward
    rewards = []
    for i in range(G):
        x_pred = decoder(rollouts[i])

        r_tok = token_reward(
            rollouts[i],
            z_gt,
            codebook,
            segments,
        )

        r_dec = calibrated_decoded_reward(
            x_pred,
            gt_frames,
            b,
            segments,
        )

        q = torch.exp(-alpha * b_norm)
        r_fused = (1.0 - q) * r_tok + q * r_dec
        rewards.append(r_fused)

    # rewards shape: [G, H, K]
    rewards = torch.stack(rewards)

    # 5. segment-level group normalization
    mu_seg = rewards.mean(dim=0)  # [H, K]
    std_seg = rewards.std(dim=0)  # [H, K]

    A_seg = (rewards - mu_seg[None, :, :]) / (
        std_seg[None, :, :] + gamma * b_norm[None, :, :] + eps
    )

    # 6. global rollout-level advantage
    R_global = (
        horizon_weights[None, :, None] * rewards
    ).mean(dim=(1, 2))

    A_global = (R_global - R_global.mean()) / (
        R_global.std() + eps
    )

    # 7. blended advantage
    A_final = (
        lambda_seg * A_seg
        + (1.0 - lambda_seg) * A_global[:, None, None]
    )

    # 8. 将 segment-level advantage 映射回每个生成 token
    token_advantages = map_segment_advantage_to_tokens(
        A_final,
        rollouts,
        token_to_segment,
    )

    # 9. 计算 GRPO clipped objective
    new_logprobs = policy_current.logprob(context, rollouts)
    ratio = torch.exp(new_logprobs - logprobs_old)

    loss_pg = -torch.min(
        ratio * token_advantages,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        * token_advantages,
    ).mean()

    loss_kl = beta_kl * compute_kl(
        policy_current,
        policy_ref,
        context,
        rollouts,
    )

    loss = loss_pg + loss_kl
    loss.backward()
    optimizer.step()
```

---

## 8. 实现注意事项

### 8.1 Segment 粒度

初始建议：

- \(K=4\)：\(2\times2\) 空间网格；
- \(K=16\)：\(4\times4\) 空间网格。

不要一开始把每个 visual token 都作为一个 segment。这样 reward 方差会很大，GRPO 更新容易不稳定。

### 8.2 LPIPS 不适合过小 patch

LPIPS 在过小 patch 上可能不稳定。因此建议：

- MSE 可以做 patch-level；
- LPIPS 优先做 full-frame 或 large-patch；
- 若需要 segment-level LPIPS，可将 frame-level LPIPS broadcast 到该帧的所有 spatial segments。

### 8.3 Codec floor 需要归一化

当前观测到的 LPIPS floor 约为 0.112，这是非常重要的动机证据。但训练中不要直接把 0.112 这类原始数值加入 GRPO denominator。

建议使用：

\[
\tilde{b}_{h,k}=\frac{b_{h,k}}{\operatorname{mean}(b)+\epsilon},
\]

或者 rank-normalization / MAD normalization。

### 8.4 多步训练从短 horizon 开始

建议训练顺序：

\[
H=1\rightarrow H=3\rightarrow H=5.
\]

不要直接从很长 horizon 开始，因为多步 rollout 的 reward 方差和 credit assignment 难度会明显增加。

---

## 9. 实验设计建议

### 9.1 Reward ablation

| 方法 | Reward 设置 | 目的 |
|---|---|---|
| Original RLVR | decoded LPIPS + MSE | 原始基线 |
| Calibrated decoded reward | decoded LPIPS/MSE 减去 codec floor | 验证 floor calibration |
| Token reward | decoder-before token/codebook reward | 验证 token-space signal |
| Fixed fusion | token reward + decoded reward 固定加权 | 验证简单融合 |
| Codec-aware fusion | 使用 \(q_{h,k}\) 自适应融合 | 验证 codec-aware fusion |

说明：该组实验证明 reward 校准有用，但不是全文重点。

### 9.2 GRPO ablation

| 方法 | GRPO 设置 | 目的 |
|---|---|---|
| Standard GRPO | rollout-level advantage | 原始基线 |
| Spatial GRPO | spatial segment-level advantage | 验证单步空间 credit assignment |
| Spatial-temporal GRPO | temporal-spatial segment-level advantage | 验证多步 credit assignment |
| Codec-aware segmental GRPO | denominator 加入 \(\gamma\tilde{b}_{h,k}\) | 验证 codec-aware advantage damping |
| w/o codec-aware term | 去掉 \(\gamma\tilde{b}_{h,k}\) | 验证 codec floor 项贡献 |

### 9.3 Horizon ablation

| Training Horizon | Evaluation Horizon | 作用 |
|---|---|---|
| \(H_{\text{train}}=1\) | \(H_{\text{eval}}=1,3,5,8\) | 验证单步训练是否改善多步 rollout |
| \(H_{\text{train}}=3\) | \(H_{\text{eval}}=1,3,5,8\) | 验证短多步训练 |
| \(H_{\text{train}}=5\) | \(H_{\text{eval}}=1,3,5,8\) | 验证更长 horizon 训练 |

最重要的比较是：

1. 单步指标是否提升；
2. 多步 rollout 的误差累积是否减缓；
3. 单步训练是否能迁移到多步评估；
4. 多步训练是否进一步提升 rollout stability。

---

## 10. 方法贡献表述

论文中可以把贡献写成三点：

1. **Codec reconstruction floor analysis**  
   发现 RLVR 视频世界模型中 decoded-frame reward 存在 codec-induced reconstruction floor。例如 GT 帧经过 encoder-decoder 重建后，LPIPS 仍约为 0.112。这说明原始 decoded LPIPS/MSE reward 会包含 tokenizer/decoder 的系统性误差。

2. **Codec-calibrated token-visual reward**  
   提出一个轻量 reward 校准模块，将 decoder 前 token-space reward 与 reconstruction-floor-calibrated decoded reward 融合。该模块不是本文唯一重点，而是为后续 GRPO 提供更可靠的局部 reward。

3. **Codec-aware spatial-temporal GRPO**  
   将 GRPO 的 rollout-level advantage 扩展为统一的 spatial-temporal segment-level advantage。单步情况下该方法退化为空间 segmental GRPO；多步情况下自然扩展为时间-空间 segmental GRPO。同时根据 codec reconstruction floor 调整 segment advantage 的归一化尺度。

---

## 11. 方法摘要

本文方法最终可以概括为：

\[
\boxed{
\text{Codec-Calibrated Reward}
+
\text{Codec-Aware Spatial-Temporal GRPO}
}
\]

主公式如下。

### Fused reward

\[
r_{i,h,k}
=
(1-q_{h,k})r^{\text{tok}}_{i,h,k}
+
q_{h,k}r^{\text{dec}}_{i,h,k}.
\]

### Codec reliability

\[
q_{h,k}
=
\exp(-\alpha \tilde{b}_{h,k}).
\]

### Segment advantage

\[
A^{\text{seg}}_{i,h,k}
=
\frac{r_{i,h,k}-\mu_{h,k}}
{\sigma_{h,k}+\gamma \tilde{b}_{h,k}+\epsilon}.
\]

### Global advantage

\[
A^{\text{global}}_i
=
\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
\]

### Final advantage

\[
\hat{A}_{i,h,k}
=
\lambda A^{\text{seg}}_{i,h,k}
+
(1-\lambda)A^{\text{global}}_i.
\]

### GRPO loss

\[
\mathcal{L}_{\text{CAST-GRPO}}
=
-
\mathbb{E}_{i,t}
\left[
\min
\left(
\rho_{i,t}\hat{A}_{i,h(t),k(t)},
\operatorname{clip}(\rho_{i,t},1-\epsilon_c,1+\epsilon_c)
\hat{A}_{i,h(t),k(t)}
\right)
\right]
+
\beta_{\text{KL}}D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}}).
\]

单步预测通过设置 \(H=1\) 得到；多步预测通过设置 \(H>1\) 得到。因此，该表达式统一覆盖单步 RLVR 和多步 RLVR。
