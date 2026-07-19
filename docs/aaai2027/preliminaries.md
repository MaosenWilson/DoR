# Preliminaries 中文工作稿

## 1. Tokenized Video World Model

给定 context frames $s_{1:C}$ 与 actions $a_{1:H}$，冻结 tokenizer 的 encoder $E$ 和 quantizer
$Q$ 将视觉状态映射为离散 tokens。世界模型 $\pi_\theta$ 自回归生成 future token blocks
$o=(o_1,\ldots,o_H)$，冻结 decoder $D$ 产生 $\hat s_h=D(o_h)$。本文只后训练
$\pi_\theta$，不更新 $E$、$Q$ 或 $D$。

## 2. Verifiable Reward and Group-Relative Optimization

对同一条件采样 $G$ 个 candidates，每个 candidate 得到可自动计算的 reward $R_i$。基础
group-relative advantage 为

$$
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
$$

策略更新提高高于组均值 candidates 的 token likelihood。候选共享的 reward 常数偏移会被组内
中心化消除，但 candidate-dependent 扰动会改变相对间隔和 ordering，因此不会自动消失。

## 3. Multi-Step Rollout

Multi-step world model 依次生成 $H$ 个 future-frame token blocks。Sequence-level baseline 聚合
逐帧 rewards 得到一个 trajectory score，并把同一个 advantage 广播给所有 blocks。本文把每个
future-frame block 作为时间信用单位，并使用它能够影响的 future rewards 构造 advantage。

## 4. Terminology

| term | definition |
|---|---|
| codec-reachable set | $\{D(z):z\in\mathcal Z\}$ |
| reachable target | $D(Q(E(s')))$ |
| target-set mismatch | raw target 通常不严格属于 codec-reachable set |
| 重建残差 (reconstruction residual) | $s'-D(Q(E(s')))$ |
| rank corruption | residual interaction 引起的 candidate ordering change |
| reachability audit | 同候选 raw/RC rank disagreement 诊断 |
| RC verifier | 对 reachable target 计算原 MSE+LPIPS |
| raw-anchored update | 保持 raw surrogate 一阶进度的 calibrated gradient projection |
| Temporal Return | future-frame token block 的 reward-to-go advantage |
