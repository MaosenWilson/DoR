# Introduction 中文工作稿

> 标题：**RC-GRPO: Reconstruction-Calibrated Temporal Credit Assignment for Tokenized Video World Models**  
> 中文暂译：**RC-GRPO：面向 Tokenized 视频世界模型的重建校准式时间信用分配**

## 0. 写作定位

本文不是“提出一个更好的 reward”的论文，也不是“提出一个通用 GRPO optimizer”的论文。

更稳的定位是：

> 在 tokenized video world model 中，verifiable reward 经过 decoder 后并不天然可靠；单步实验揭示并校准 verifier 侧的重建地板，而多步 rollout 进一步暴露 sequence-level GRPO 信用分配过粗的问题。RC-GRPO 将 reconstruction-calibrated verifier 与 frame-level temporal credit assignment 结合，用于改善多步视频世界模型的可验证强化微调。

## 1. 术语表

| Canonical term | 中文解释 | 决策 |
|---|---|---|
| tokenized video world model | 通过视觉 tokenizer 把帧离散化，再用自回归模型预测未来 token 的视频世界模型 | 英文正文使用该词 |
| verifiable reward | 与真实未来帧或其可达重建目标比较得到的奖励 | 不写成 learned reward |
| reconstruction floor | tokenizer encode-decode 往返造成的不可约重建残差 | 我们自定义术语，需明确不是官方概念 |
| reachable target | $\tilde{s}=D(E(s))$，tokenizer 可达的 decoded ground truth | 用于 reconstruction-calibrated reward |
| RC-GRPO | Reconstruction-Calibrated GRPO | 方法名 |
| temporal credit assignment | 多步 rollout 中把优势分配到 future-frame block，而不是整条序列共享同一个 scalar advantage | 主方法贡献 |
| temporal-return advantage | 用未来帧 reward return 构造每个 frame block 的 advantage | 最终采用的多步 GRPO 形式 |

## 2. Introduction 段落地图

### P1. Field stake

目标：建立 video world model + tokenization + RL post-training 的重要性。

要点：

- Video world models 将 action-conditioned future prediction 作为可交互环境建模的核心。
- Tokenized autoregressive world models 通过视觉 tokenizer 把视频预测变成 sequence modeling。
- RL-style post-training with verifiable rewards 为这类模型提供了从采样候选中强化更好预测的路径。

建议引用：

`ivideogpt`, `genie`, `dreamerv3`, `rt1`, `openxembodiment`, `rlvrworld`, `grpo`

### P2. Bottleneck: verifiable does not mean clean

目标：提出本文第一个核心问题。

要点：

- RLVR-style 框架通常把 decoded predictions 和 ground-truth future frame 做 full-reference comparison。
- 但 tokenized model 的 decoder 本身有不可约 reconstruction floor。
- 这个 floor 与预测质量无关，却会进入 post-decode reward。
- GRPO 依赖组内相对排序，因此问题不是 reward 数值偏移，而是候选排序被腐蚀。

建议引用：

`vqvae`, `vqgan`, `fsq`, `lpips`, `ssim`, `perceptiondistortion`

### P3. Deeper gap: multi-step rollout needs temporal credit

目标：把论文主菜从 reward 校准自然推进到 GRPO 侧。

要点：

- 单步中，reachable-target calibration 可以修复 verifier target mismatch。
- 多步中，视频 rollout 是自条件过程，早期预测会影响后续帧。
- 原始 sequence-level GRPO 把整条 rollout 压成一个 scalar advantage，再广播到所有 token。
- 这会混淆不同 future frames 的责任，尤其在后期误差累积时更严重。
- 现有细粒度 GRPO credit assignment 工作说明 coarse sequence-level advantage 是普遍问题，但 tokenized video 的自然信用单元不是文本 token，而是 future-frame block。

建议引用：

`grpolambda`, `spo`, `sdgrpo`, `stepwiseflowgrpo`, `dapo`

### P4. Our approach

目标：定义 RC-GRPO。

要点：

- RC-GRPO 保留 RLVR-style sample group、verifiable reward、group-relative update。
- 第一部分：用 reachable target $D(E(s_t))$ 替代 raw ground truth，得到 reconstruction-calibrated verifier。
- 第二部分：在 multi-step rollout 中计算 frame-level verifiable rewards，并把 temporal-return advantage 作用到对应 future-frame token block。
- 不是替换世界模型架构，也不是引入 learned critic。

### P5. Evidence and contribution summary

目标：概括证据，但不堆数字。

要点：

- 单步离线诊断显示不同 reward 的 rank flip 与 $\arccos(\rho)/\pi$ 理论吻合。
- RC calibration 在单步和多步中都改善或稳定 full-reference fidelity。
- 在 multi-step fixed-step protocol 下，temporal-return RC-GRPO 优于 raw/sequence-level baselines，并且优势随 horizon 增长。
- 系统性负结果显示，直接替换通用 GRPO 变体并不能解决该问题；有效干预点在 video-structured verifier calibration 与 temporal credit assignment。

## 3. 结构审查与修改决策 v2

参考 **Pre-Trained Video Generative Models as World Simulators** 的 AAAI-2026 写法后，当前引言需要从“模块解释”改成“挑战拆解”。DWS 的结构不是一开始就讲实现细节，而是先把视频世界模型的场景、现有路线和关键困难讲清楚，再说明方法如何分别处理这些困难。因此，本稿采用：

> field stake → two failure modes → method response → evidence and contributions

具体修改决策：

- 把“reward 设计”降级为 verifier-side mismatch 的校准问题，不把它写成主菜。
- 把“multi-step temporal credit assignment”提前为第二个 failure mode，让 GRPO 侧贡献成为主线。
- 贡献列表改为三点：reconstruction-floor diagnosis、reconstruction-calibrated verifier、temporal-return GRPO。
- 结果表述保持克制：只写 fixed held-out protocol 下 full-reference fidelity 改善，不声称动态指标全面提升，不声称通用 GRPO 改进。

## 4. 正式中文引言 v2

视频世界模型在给定历史观测和动作的条件下预测未来视觉状态，可为机器人控制、交互式仿真和模型式强化学习提供可采样的环境模型。一个主流实现路线是将视觉观测 tokenized：先用视觉 tokenizer 把图像帧压缩为离散 token，再用自回归模型预测未来 token，最后由 decoder 还原为图像。这样，action-conditioned video prediction 被转化为序列建模问题，也使视频世界模型可以借鉴语言模型中的采样和强化微调范式。近期的 verifiable reward 框架进一步提出：对同一状态-动作上下文采样一组候选未来帧，用 full-reference verifier 将 decoded predictions 与真实未来帧比较，再通过 GRPO 强化组内得分更高的候选。这个流程的吸引力在于，它不需要 learned critic，而是直接利用可测量的预测误差进行后训练。

本文关注一个更具体但关键的问题：当世界模型依赖 lossy visual tokenizer 时，decoded full-reference reward 是否仍然是适合 GRPO 的训练信号，尤其是在 multi-step rollout 中？我们发现这里存在两个相互关联的失配。第一，verifier 通常把 decoded prediction 与 raw ground truth 比较，但 raw ground truth 并不一定处在 tokenizer-decoder 可达的图像空间中。第二，多步 rollout 通常被压缩为一个 sequence-level advantage，但视频预测是自回归过程，早期帧会成为后续帧的条件，错误会沿 horizon 传播。这两个问题不会破坏“可验证”本身：每个候选仍然能和已知目标比较；但它们会破坏 GRPO 真正消费的训练信号，即组内排序和时间信用分配。

第一个失配发生在 verifier 侧。lossy visual tokenizer 会在 ground-truth frame 的 encode-decode 往返中产生不可约残差，我们将其称为 reconstruction floor。这个残差并非任何候选预测造成，却会被 MSE、SSIM、LPIPS 等 post-decode metrics 一并计入 reward。对于普通评测，这可以理解为 tokenizer 的重建质量上限；但对于 GRPO，它会变成排序噪声。因为 GRPO 使用组内归一化优势，关键不是 reward 数值是否整体偏移，而是本应排在前面的候选是否因为 decoder-side residual 被排到后面。

第二个失配发生在 rollout 侧。单步预测只需要判断下一帧候选的相对好坏；多步视频世界模型则会把前一步生成结果继续作为后续预测条件。此时，一个后期帧错误可能来自当前帧 token，也可能来自更早帧的误差传播。标准 sequence-level GRPO 将整条 rollout 压成一个 scalar advantage，并把它广播到所有生成 token，因而忽略了视频预测的时间结构。已有一些 GRPO-style 工作讨论了长序列推理或生成过程中的细粒度信用分配，但 tokenized video world model 的自然信用单元不是文本 token，也不是 denoising step，而是一个 future frame 对应的 visual-token block。

为此，我们提出 **RC-GRPO**，即 reconstruction-calibrated temporal credit assignment for tokenized video world models。RC-GRPO 保留 sample group、verifiable reward 和 group-relative update 的基本框架，只在视频结构真正起作用的位置修改训练信号。首先，在 verifier calibration 中，我们把 raw ground-truth target 替换为 tokenizer-reachable target $\tilde{s}_t=D(E(s_t))$，使 decoded prediction 与 decoded reachable target 在同一可达空间中比较。其次，在 multi-step rollout 中，我们计算 frame-level verifiable rewards，并用 temporal-return advantage 为每个 future-frame token block 分配信用。这样，早期 token 可以通过其对后续帧质量的影响获得 credit，而不同 horizon 的预测误差也不会被混合成一个 rollout-level scalar。

本文的贡献有三点。第一，我们识别并量化 tokenized video GRPO 中的 reconstruction-floor failure mode，并验证 post-decode verifier 的 pairwise rank flips 与 $\arccos(\rho)/\pi$ 的理论关系在多个 reward spaces 上一致。第二，我们提出 reconstruction-calibrated verifier，使单步和多步训练都能使用 tokenizer-reachable target，避免把 decoder 不可达误差错误地计入候选预测质量。第三，我们提出 temporal-return GRPO，用 frame-level returns 替代 sequence-level advantage；在 RT-1 prediction 的固定 held-out protocol 下，该方法相对 sequence-level GRPO 改善 LPIPS、MSE、MAE、PSNR 和 SSIM 等 full-reference fidelity metrics，且收益主要出现在更长 horizon。我们同时报告 rank-label VPO、GSPO、segment-level variants 等通用 GRPO 替换的负结果，说明该任务的有效干预点不是简单更换优化器，而是同时处理 verifier-side reconstruction mismatch 与 rollout-side temporal credit assignment。

## 5. 英文 LaTeX 写作映射

中文 v2 转英文时采用 6 段结构：

1. **Context**：video world models + tokenized sequence modeling + verifiable post-training。
2. **Two failure modes**：verifier target mismatch + sequence-level temporal credit mismatch。
3. **Verifier gap**：post-decode verifier 受 reconstruction floor 影响，GRPO 关心组内排序。
4. **Temporal gap**：multi-step rollout 自条件，sequence-level advantage 太粗。
5. **Approach**：RC-GRPO = reachable target calibration + temporal-return frame-block advantage。
6. **Contributions and boundary**：rank-flip 诊断、multi-step full-reference fidelity、负结果边界。

## 6. 当前引言中的边界

必须保留的边界：

- 不声称 RC-GRPO 是通用 GRPO 改进。
- 不声称 code reward 普遍优于 pixel/perceptual reward。
- 不声称动态指标全面提升；当前 `dmotion` 不支持这一点。
- 不声称完整训练设置击败官方 RLVR，只能写在相同 held-out protocol / fixed-step readout 下的结果。
- 不把 reconstruction floor 写成已有官方术语；它是本文定义的诊断概念。

## 7. 需要后续补齐

- 将中文引言继续压缩为更短的 AAAI camera-ready 版本。
- 确认每个 citation key 是否最终保留。
- Introduction 不应塞入过多数字；详细结果放 Experiments。
- 需要决定是否在最后贡献列表显式列 3 点，还是用一段自然语言总结。AAAI 风格可以列点，但不宜过长。
