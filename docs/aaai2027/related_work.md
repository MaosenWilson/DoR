# Related Work 中文正式工作稿

> 文献必须在 BibTeX 和原文中逐条核验。下列 arXiv/会议条目均为已检索到的真实来源；正式 tex 只使用已经进入 bibliography 的 citation keys。

## 1. Tokenized Video World Models

VideoGPT、iVideoGPT 等方法把视频压缩为离散或连续视觉 tokens，再以自回归模型学习 action-conditioned future prediction。Tokenization 提高了序列建模效率，但生成结果被限制在 codec 的可达输出空间，重建质量由 tokenizer rate-distortion trade-off 限制。已有研究通常把 reconstruction quality 作为 tokenizer 的表示能力问题；本文关注一个不同后果：冻结 codec 的不可达 raw target 如何影响 decoded verifier 给出的 candidate ordering。

应保留并核验的代表性来源：

- VideoGPT；
- iVideoGPT；
- Phenaki / MAGVIT 类 tokenized video generation；
- FSQ；
- rate-distortion / neural compression 基础来源。

## 2. RLVR for World Models

RLVR-World（Wu et al., NeurIPS 2025，arXiv:2505.13934）把语言和视频世界建模统一为序列预测，并以任务指标作为 verifiable rewards。其视频实验对 decoded predictions 使用 MSE、LPIPS 等 full-reference metrics，再由 GRPO 更新 token policy。我们的工作保持该基础流程、模型和 evaluator，不提出新的通用 RLVR 框架；区别在于分析 decoded verifier 的 reachable-target mismatch，并重新设计 multi-step frame-block credit。

需要明确避免的表述：

- 不说 RLVR-World “reward 错了”；它的 reward 对 raw-frame fidelity 是合法 evaluator；
- 不把复用其架构写成贡献；
- 不把我们的 held-out protocol 结果写成复现并击败官方完整训练设置。

## 3. Perceptual Metrics and Codec Reconstruction

MSE、SSIM/MS-SSIM、LPIPS、DISTS、GMSD、FSIM 和 HaarPSI 从像素、结构和 learned feature 等角度测量 full-reference fidelity。LPIPS/DINO 等 feature-level metric 若在 decoded image 上计算，仍属于 post-decode evaluation；“像素 vs 特征”不等于“有无 codec floor”。FID/FVD/KID 等 distributional metrics 是集合统计，不能直接替代 RLVR 所需的 per-candidate reward。

本文不以增加 metric 数量作为创新。额外 post-decode metrics、pre-decode code/gradient、learned fusion、adaptive guard 与 local-floor pooling 均未超过最小 RC verifier；这些结果只界定当前设置的边界。

## 4. Fine-Grained Credit Assignment in Visual GRPO

Flow-GRPO 将 GRPO 引入 flow-matching generation，但初始方法通常把最终 outcome reward 广播到整条 denoising trajectory。2026 年的工作开始研究 step-aware credit：

- Stepwise-Flow-GRPO，*Stepwise Credit Assignment for GRPO on Flow-Matching Models*，arXiv:2603.28718；
- TurningPoint-GRPO，*Alleviating Sparse Rewards by Modeling Step-Wise and Long-Term Sampling Effects in Flow-Based GRPO*，arXiv:2602.06422，公开 demo code；
- OTCA，*Learning to Credit the Right Steps: Objective-aware Process Optimization for Visual Generation*，arXiv:2604.19234；
- SAGE-GRPO，针对 video diffusion exploration reliability 与 stepwise constraints，公开代码。

这些工作支持细粒度 credit 的必要性，但研究对象主要是 diffusion/flow denoising steps。我们的 temporal units 是 action-conditioned autoregressive future frames：早期 frame tokens 会成为后续预测条件，逐帧 reward 也可以直接对真实 future frames 验证。我们进一步把 frozen codec 引起的 rank reliability 与 temporal credit 联系起来。

## 5. Reliability-Aware Policy Optimization

AdaGRPO（arXiv:2606.08480）等工作根据 reward discriminability 或样本可靠性选择性施加 GRPO，说明可计算 reward 在不同样本上不一定同样可信。我们的 Rank-Reliable Return 候选与其共享“按可靠性分配优化压力”的原则，但可靠性来源不同：我们使用 tokenized video verifier 的 horizon-wise pairwise flip probability，而不是推荐 ranker 或 policy difficulty。

目前没有检索到直接使用

$$
1-2\arccos(\rho_h)/\pi
$$

调制 autoregressive video temporal returns 的论文或代码。因此该组合若验证成功可作为新方法；若失败，不得借相邻文献包装成已有方法的必然延伸。

## 6. Positioning Summary

本文位于三条工作线的交叉点：tokenized video world models 提供冻结 codec 与 action-conditioned rollout；RLVR-World 提供 decoded verifier 和 GRPO 后训练；stepwise visual GRPO 提供细粒度 temporal credit 的近期背景。我们的特定问题是：**当 decoded reward 的 group-relative ranking 受 codec 与 horizon 共同影响时，如何校准 verifier 并分配可信的 future-frame credit。**
