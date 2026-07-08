# Related Work 中文工作稿

> 标题：**RC-GRPO: Reconstruction-Calibrated Temporal Credit Assignment for Tokenized Video World Models**

## 0. 写作定位

Related Work 的目标不是堆引用，而是让审稿人快速确认三件事：

1. 我们基于已有 tokenized video world model / RLVR-World 框架，不声称重做世界模型架构。
2. 我们的问题不是“提出更多 reward 指标”，而是 tokenized verifier 的 reachable target mismatch。
3. 我们的 GRPO 侧贡献不是通用 optimizer 替换，而是 multi-step video rollout 中的 frame-block temporal credit assignment。

因此本节采用 4 个主题：

> tokenized video world models → verifiable reward post-training → visual tokenizers and metrics → fine-grained credit assignment

每个主题最后都要落到本文差异。

## 1. 段落地图

### P1. Tokenized video world models

要讲：

- World model 从 latent dynamics / model-based RL 到 video generative simulator。
- Tokenized video world model 把 future prediction 转成 sequence modeling。
- iVideoGPT、DWS、Genie、MAGVIT/VideoGPT 等支撑这一背景。

落点：

> 这些工作主要关注模型架构、预训练和仿真能力；本文关注的是预训练之后，用 verifiable rewards 微调 tokenized video world model 时训练信号是否可靠。

建议引用：

`worldmodels`, `rssm`, `dreamerv2`, `dreamerv3`, `videogpt`, `magvit`, `ivideogpt`, `genie`, `dws`

### P2. Verifiable reward post-training and GRPO

要讲：

- RLVR-World 把 world model 后训练转成 sample group + verifiable reward + GRPO。
- GRPO/RLVR 在 LLM reasoning 中常见，不依赖 learned value function。
- 相关 GRPO/RLVR 变体很多，但大多关注优化目标、clip、sample efficiency 或 sequence-level stability。

落点：

> 本文不把问题定义为通用 GRPO optimizer 选择，而是指出 tokenized video verifier 本身会把 decoder-side residual 带入组内排序；因此需要先校准 reward target。

建议引用：

`rlvrworld`, `grpo`, `deepseekr1`, `drgrpo`, `dapo`, `gspo`, `realrlvr`, `sapo`

### P3. Visual tokenizers, reconstruction limits, and evaluation metrics

要讲：

- VQ-VAE/VQGAN/FSQ 等视觉 tokenizer 都有 encode-decode reconstruction limit。
- Perception-distortion tradeoff 提供“重建上限/感知质量”的背景。
- SSIM/LPIPS/MSE 是 full-reference per-sample metrics；FID/KID/FVD/DINO/PRDC 是 distribution-level metrics。

落点：

> 既有评价指标主要用于报告生成质量；本文关心的是这些指标被放进 GRPO group ranking 后会发生什么。尤其 distribution-level metrics 不能直接作为 per-candidate verifier，而 post-decode full-reference metrics 又会继承 tokenizer floor。

建议引用：

`vqvae`, `vqgan`, `fsq`, `perceptiondistortion`, `ssim`, `lpips`, `fid`, `kid`, `fvd`, `dinov2`, `densitycoverage`, `precisionrecall`

### P4. Fine-grained credit assignment for GRPO-style training

要讲：

- Sequence-level advantage 对长序列生成过粗，这一点在 LLM reasoning、segment-level VLM generation、flow/diffusion generation 中都被观察到。
- 代表工作包括 GRPO-\(\lambda\)、SPO、SD-GRPO、Stepwise-Flow-GRPO、DAPO 等。

落点：

> 我们借鉴的是“coarse sequence-level credit 不够”的原则，但不是照搬 token-level 或 denoising-step credit。tokenized video 的自然信用单元是 future-frame token block，reward 是 frame-level full-reference verifier，且 verifier 先经过 reachable-target calibration。

建议引用：

`grpolambda`, `spo`, `sdgrpo`, `stepwiseflowgrpo`, `dapo`

## 2. 正式中文 Related Work v1

### Tokenized video world models

世界模型通过学习环境动态，为机器人控制、交互式仿真和模型式强化学习提供可采样的未来状态预测。早期工作主要在 latent dynamics 中建模环境状态转移，后来的视频生成式世界模型进一步把视觉预测本身作为建模目标。随着视觉 tokenizer 和 transformer 视频生成模型的发展，tokenized video world model 成为一种自然路线：图像帧被压缩为离散 token，未来视觉状态由自回归序列模型预测，再通过 decoder 映射回像素空间。iVideoGPT、Genie、DWS 等工作表明，预训练视频生成模型可以作为可交互的世界模拟器。然而，这些工作主要回答“如何预训练或构建一个可采样的视频世界模型”，而本文关注预训练之后的另一个问题：当这类模型用 verifiable reward 和 GRPO 进行后训练时，reward target 与 credit assignment 是否仍然匹配 tokenized video 的结构。

### Verifiable reward post-training and GRPO

基于可验证奖励的后训练框架把世界模型微调转化为一个 group-relative optimization 问题：对同一上下文采样多个候选未来帧，用已知真实帧计算 verifiable reward，再根据组内相对优势更新策略。RLVR-World 将这一思想应用到 world model 训练中，GRPO 则提供了无需 learned value function 的组内归一化更新形式。相关 RLVR/GRPO 变体进一步讨论了采样效率、clip 策略、长度偏置、classification-style rank labels 或大规模 reasoning 训练中的稳定性。本文与这些通用优化器改造不同：我们发现，在 tokenized video world model 中，即使优化器本身不变，post-decode verifier 也可能因为 tokenizer-decoder 的不可达误差而腐蚀组内排序。因此，有效干预点首先是 verifier target 的 reconstruction calibration，而不是简单替换 GRPO 目标函数。

### Visual tokenizers, reconstruction limits, and evaluation metrics

视觉 tokenizer 是 tokenized generation 的基础，但 lossy tokenization 不可避免会引入 encode-decode 重建残差。VQ-VAE、VQGAN 和 FSQ 等离散表示学习方法提高了图像或视频 tokenization 的可建模性，perception-distortion tradeoff 也说明了重建误差与感知质量之间的基本张力。在评价层面，SSIM、LPIPS 等 full-reference metrics 常用于逐样本保真度，FID、KID、FVD、DINO 特征距离、precision/recall 和 density/coverage 等指标则用于集合或分布层面的生成质量。本文不是重新提出一个视觉评价指标，而是研究这些指标被用作 GRPO verifier 时的训练后果：distribution-level metrics 不能直接给单个候选提供标准 per-sample reward，而 post-decode full-reference metrics 会继承 tokenizer reconstruction floor，并可能改变 GRPO 所依赖的候选排序。

### Fine-grained credit assignment for GRPO-style training

长序列强化微调中的一个共同问题是 sequence-level advantage 过粗：当一条生成轨迹包含多个语义步骤、推理段或生成阶段时，把同一个标量优势广播到所有 token 会混淆局部决策的责任。近期 GRPO-style 工作从不同角度讨论了这一问题，包括 eligibility-trace 式 token credit、segment-level policy optimization、vision-language long-form generation 的 segment decomposition、flow/diffusion generation 中的 stepwise credit assignment，以及大规模 RL 系统中的 token-level loss shaping。本文借鉴这一原则，但具体落点不同。tokenized video rollout 的自然信用单元不是文本 token、reasoning segment 或 denoising step，而是一个 future frame 对应的 visual-token block；同时，每个 block 的 reward 来自 frame-level full-reference verifier，并且 verifier target 需要先经过 reconstruction calibration。因此，RC-GRPO 将细粒度 credit assignment 本地化为 reconstruction-calibrated temporal return，而不是提出一个任务无关的通用 GRPO optimizer。

## 3. 英文 LaTeX 映射

LaTeX 中使用一个 `\section{Related Work}`，内部用 4 个 `\paragraph{...}`：

1. `Tokenized video world models.`
2. `Verifiable reward post-training.`
3. `Visual tokenizers and evaluation metrics.`
4. `Fine-grained credit assignment.`

边界：

- 不声称我们提出新的 video world model 架构。
- 不声称我们提出新的视觉评价指标。
- 不声称我们提出通用 GRPO optimizer。
- 不把负结果写成 related work 主线，只用来限定“generic optimizer swap”不是本文主张。

