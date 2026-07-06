# Rank-Label VPO：把 REAL 本地化到 tokenized 视频 RLVR（2026-07-04）

> 目的：在 reward 校准已经稳定、原有 GRPO 侧 segment / filter / Dr.GRPO 改造均未过线后，尝试一个真正替换原始 GRPO 的优化目标。本文档先落地设计，再改脚本。

## 1. 外部方法核查

已下载完整论文：

- `docs/aaai2027/papers/REAL_rewards_as_labels_2602.05630v4.pdf`
- 标题：*Rewards as Labels: Revisiting RLVR from a Classification Perspective*
- arXiv：2602.05630v4
- 官方代码：`external/REAL`，来源 `https://github.com/DeepExperience/REAL`

REAL 的核心不是新增 reward，也不是 reward 加权，而是把 verifiable reward 从 scalar weight 改成 categorical label。原文指出 GRPO-style objective 有两类梯度错配：

1. **Gradient Misassignment in Positives**：正样本里低置信、难学的 rollout 反而更新弱。
2. **Gradient Domination in Negatives**：负样本里高置信错误 rollout 梯度过强，主导更新。

REAL 用 length-normalized relative log-probability 作为 logit：

$$
\bar{s}_i=
\frac{1}{|o_i|}
\sum_t
\left[
\log \pi_\theta(o_{i,t}\mid q,o_{i,<t})
-
\log \pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
\right].
$$

原版是二元 verifiable reward $r_i\in\{0,1\}$，直接划分

$$
O^+=\{i:r_i=1\},\qquad O^-=\{i:r_i=0\}.
$$

带 anchor logit 0 的 REAL loss 为：

$$
\mathcal L_{\mathrm{REAL}}
=
\log\left(1+\sum_{j\in O^-}\exp(\bar{s}_j/\tau)\right)
+
\log\left(1+\sum_{i\in O^+}\exp(-\bar{s}_i/\tau)\right).
$$

直观解释：正样本要被推到 $\bar{s}>0$，负样本要被推到 $\bar{s}<0$。anchor 0 防止只拉开正负差距、但两边一起漂移。

## 1.1 对 REAL 论文和代码的保守判断

读者评论里有几条是成立的，不能忽略：

1. **DAPO / GSPO / GRPO 的对比高度依赖超参。** REAL 论文里 GRPO/DAPO 训练崩或不稳，不等于 REAL 在所有 RLVR 设置下更强。我们不能把它的 leaderboard 当作本文证据，只能把 REAL 当作一个可借鉴的 objective 设计。
2. **官方 `verl` 实现依赖 uid 分组与 all-gather 后 reshape。** 如果分布式 batch 中 group 不完整，或者 uid 排序/切分假设不成立，loss 就可能混组。我们的 `dor` 精简 harness 反而更干净：每次在单个 transition 的 $G=16$ 候选内部直接排序和计算 loss，不跨 batch 拼 group。
3. **REAL 原版是二元 reward。** 我们是连续视频 reward，不能直接照搬 $r\in\{0,1\}$。本地化必须通过组内 top/bottom rank labels，而不是把连续 reward 硬阈值成绝对正确/错误。
4. **CISPO / SAPO 可能是更强对手。** CISPO（MiniMax-M1）clip detached importance-sampling weight；SAPO 用 soft adaptive gate 替代 hard clipping。它们主要解决 off-policy ratio 和 clipping 的稳定性。当前 `dor` harness 是 on-policy、单次更新、无 PPO epochs，因此 CISPO/SAPO 的主优势暂时没有完全触发。若 RLVPO pilot 过线，正式论文至少要把 CISPO/SAPO 放进 related work；若要把“优化器替换”做成强主张，后续应补一个 lightweight CISPO/GSPO/SAPO-style 对照，而不是只比 vanilla GRPO。

因此本文档采用的立场是：

> 不声称 REAL 原论文一定公平或最强；只借用“verifiable reward as labels + anchor classification loss”这个机制，并在我们自己的完整 group、连续 reward、codec floor 场景里重新验证。

## 2. 为什么适合我们

我们当前最强输入不是二元正确/错误，而是连续的 calibrated verifiable reward：

$$
R_i^{\mathrm{RC}}
=
z_G\big(-\mathrm{LPIPS}(\hat{s}'_i,\tilde{s}')\big)
+
0.10\,z_G(R_i^{\mathrm{dyn}}),
\qquad
\tilde{s}'=D(E(s')).
$$

原始 GRPO 把这个连续值变成 advantage：

$$
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon},
\qquad
\mathcal L_{\mathrm{GRPO}}=-\frac{1}{G}\sum_i A_i\log\pi_\theta(o_i\mid q).
$$

这仍然会让中间候选参与梯度，而中间候选恰好是最容易被 reward floor 翻转的区域。我们的地板机理已经说明，GRPO 真正在乎组内排序；当排序受噪声腐蚀时，把所有候选按连续强度加权未必是最稳的选择。

因此本地化策略是：**只把 calibrated reward 用作组内 rank label 生成器，而不是直接作为梯度权重。**

## 3. 本地化公式：Rank-Label VPO

对同一转移 $q$ 采样 $G=16$ 个候选，先用 `pixel_tok_dyn` 计算 $R_i$。按组内分位数划分：

$$
O^+=\{i:R_i\ge Q_{1-\rho_+}(R)\},
\qquad
O^-=\{i:R_i\le Q_{\rho_-}(R)\}.
$$

默认：

$$
\rho_+=\rho_-=0.25,
$$

即 top-4 为正、bottom-4 为负，中间 8 个候选不进 loss。若组内 reward spread 太小：

$$
\max_i R_i-\min_i R_i < m,
$$

则该组跳过或后续改成 dynamic resampling。第一版默认 $m=0$，保证 pilot 不因过滤丢 batch。

序列级 logit 仍用 REAL 的 relative log-prob：

$$
\bar{s}_i=
\frac{1}{T}
\sum_{t=1}^{T}
\left[
\log \pi_\theta(o_{i,t}\mid q,o_{i,<t})
-
\mathrm{stopgrad}\big(\log \pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})\big)
\right].
$$

在当前 `dor` 精简训练里，rollout policy 与更新前 policy 是同一个模型；实现上取

$$
\log \pi_{\theta_{\mathrm{old}}}
=
\mathrm{stopgrad}(\log \pi_\theta),
$$

所以 $\bar{s}_i$ 的数值初始为 0，但梯度非零。这与 REAL 官方实现一致：loss 通过当前 log-prob 的梯度推动正样本上升、负样本下降。

### 为什么横轴/score 用 log-ratio，而不是直接用 $\pi_\theta/\pi_{\mathrm{old}}$

两者信息等价：

$$
\log\frac{\pi_\theta(o)}{\pi_{\mathrm{old}}(o)}
=
\log\pi_\theta(o)-\log\pi_{\mathrm{old}}(o).
$$

但 log-ratio 更适合自回归序列：

1. **可加性**：序列概率是 token 概率连乘，直接 ratio 会变成大量小数连乘，数值不稳；log-ratio 可按 token 求和/平均。
2. **长度归一方便**：$\bar{s}=\frac1T\sum_t\log r_t$ 是每 token 平均更新幅度，固定 320 token 时仍然是稳定尺度。
3. **梯度分析更清楚**：policy gradient 本来就是 $\nabla\log\pi_\theta$，所以观察梯度幅度随 log-ratio 变化，比随 ratio 变化更线性、更可解释。

读者指出 “$\log\pi=z-C$，$\pi_\theta$ 与 $\pi_{\mathrm{old}}$ 的 $C$ 不同，所以 log-ratio 不只是 raw logit 差” 是对的。严格地说：

$$
\log\frac{\pi_\theta(a)}{\pi_{\mathrm{old}}(a)}
=
\big[z_\theta(a)-z_{\mathrm{old}}(a)\big]
-
\big[\log Z_\theta-\log Z_{\mathrm{old}}\big].
$$

所以横轴不是单个 token raw logit 的差，而是**归一化后 log-prob 的差**。这反而是应该使用的量，因为策略更新真正关心的是概率分布变化，不是未归一化 logit 变化。我们文中不能把它简称成“logit difference”，应写成 **relative log-probability score**。

最终 loss：

$$
\boxed{
\mathcal L_{\mathrm{RLVPO}}
=
\log\left(1+\sum_{j\in O^-}\exp(\bar{s}_j/\tau)\right)
+
\log\left(1+\sum_{i\in O^+}\exp(-\bar{s}_i/\tau)\right)
}
$$

默认 $\tau=0.5$，沿用 REAL 论文与代码中的稳定设置。

## 4. 与原始 GRPO 的差别

| 项 | Vanilla GRPO | Rank-Label VPO |
|---|---|---|
| reward 用法 | 连续 scalar weight | 只生成 top/bottom labels |
| 中间候选 | 全部参与 | 默认忽略，降低翻转风险 |
| 梯度形态 | $A_i\log\pi$，幅度随 advantage 走 | 分类式 anchor loss，梯度有界 |
| 目标粒度 | rollout-level scalar | sequence-level relative log-prob |
| 论文定位 | RLVR-World 原始优化器 | 替换 GRPO 的 policy optimization 创新点 |

## 5. 最小实现点

只改 `dor` 精简 harness，不动模型、tokenizer、生成器、reward 计算：

1. `scripts/train_grpo.py`
   - `--adv_estimator real`
   - `--real_tau`
   - `--real_pos_frac`
   - `--real_neg_frac`
   - `--real_min_gap`

2. `src/dor/grpo.py`
   - 新增 `rank_label_loss(...)`
   - 在 `adv_estimator == "real"` 时，用 `pixel_tok_dyn` 等 reward 只产生 rank labels。
   - `seq_logp()` 继续提供 per-token logprob；old logprob 使用 `tok_logp.detach()`。

3. 默认推荐命令：

```bash
$P scripts/train_grpo.py \
  --adv_estimator real \
  --rewards pixel_tok_dyn \
  --modes gt_only \
  --seeds 0,1,2,3,4 \
  --steps 150 --K 16 \
  --eval_every 10 \
  --dyn_lambda 0.10 --dyn_gamma 0.25 \
  --real_tau 0.5 \
  --real_pos_frac 0.25 --real_neg_frac 0.25 \
  --out_dir outputs/real_vpo_ranklabel_lam010
```

## 6. Pilot 判据

必须同 sweep 配对比较：

- baseline：`adv_estimator=grpo + reward=pixel_tok_dyn`
- treatment：`adv_estimator=real + reward=pixel_tok_dyn`

主判据：

1. flow 稳定高于 baseline，至少 4/5 seed 提升。
2. dmotion 不下降，最好同步提升。
3. LPIPS/PSNR/SSIM 不显著变差。

如果 REAL-style loss 只提高动态但明显伤保真，不能作为主贡献；最多作为探索消融。

## 7. 预期失败模式

1. **连续 reward 转 label 太粗**：top/bottom 丢掉强度信息，可能训练慢。
2. **top/bottom 仍有翻转**：当组内 reward spread 很小时，label 本身不可靠。可用 `--real_min_gap` 或后续 dynamic resampling。
3. **固定 320 token 导致 sequence-level 优势有限**：相比 LLM 长短不一输出，我们没有长度偏置问题，REAL 的收益可能小。
4. **采样非确定性仍大**：必须同 sweep 配对，不接受跨批结论。

## 8. 如果 pilot 成功，论文贡献表述

建议名称暂定：

**Rank-Label Verifiable Policy Optimization (RLVPO)**。

贡献写法：

1. 发现 tokenized video RLVR 的 codec reconstruction floor 会腐蚀组内排序，并给出翻转率预测。
2. 提出 floor-calibrated verifiable reward，得到稳定的 verifier。
3. 提出 Rank-Label VPO：把连续 calibrated reward 转成组内 top/bottom labels，用分类式 sequence objective 替换原始 GRPO，避免中间噪声候选主导更新。

注意：正式论文里必须写清楚 REAL 是启发来源；我们的新点不是发明 classification RLVR，而是把它本地化到**连续视频 reward + rank labels + codec floor 机制**。
