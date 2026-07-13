# Preliminaries 中文正式工作稿

## 1. Tokenized Video World Modeling

给定 context frames $s_{1:C}$ 与 actions $a_{1:H}$，冻结 tokenizer $E$ 将视觉状态映射为离散 tokens，world model $\pi_\theta$ 自回归预测 future token trajectory $o=(o_1,\ldots,o_H)$，冻结 decoder $D$ 产生预测帧 $\hat s_h=D(o_h)$。本文只后训练 $\pi_\theta$，不更新 $E$ 或 $D$。

## 2. Verifiable Reward and GRPO

对同一条件采样 $G$ 个候选，每个候选得到可自动计算的 reward $R_i$。Vanilla GRPO 使用

$$
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}
$$

构造 group-relative advantage，并提高高于组均值候选的 token likelihood。因而，对 GRPO 而言，reward 的绝对偏移通常不重要，候选的组内顺序与相对间隔才是主要信息通道。

## 3. Multi-Step Video Rollouts

Multi-step world model 依次生成 $H$ 个 future-frame token blocks。Sequence-level baseline 将逐帧 rewards 聚合为单个 trajectory score，并把同一个 $A_i$ 广播到所有 blocks。本文把每个 frame block 视为时间信用单位，并利用 future return 表达早期 token 对后续 decoded frames 的影响。

## 4. Terminology

| term | definition |
|---|---|
| reconstruction residual | $s'-D(E(s'))$ |
| reconstruction-induced rank corruption | residual interaction 引起的 candidate pair ordering change |
| reachable target | $D(E(s'))$ |
| RC verifier | 对 reachable target 计算原 MSE+LPIPS 的 verifier |
| sequence-level GRPO | 整段 rollout 共享一个 group-relative advantage |
| Temporal-Return GRPO | 每个 frame block 使用其 future reward-to-go advantage |
| Rank-Reliable Return | 用 frozen horizon-wise flip reliability 调制 temporal return 的候选方法 |
