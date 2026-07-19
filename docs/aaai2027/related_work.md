# Related Work 中文工作稿

> 正式 LaTeX 只引用已经在原文、官方 proceedings 或官方项目页核验的条目。本稿不再列入与当前
> 两项贡献无关的 GRPO 变体或尚未核验的 2026 年方法。

## 1. Tokenized Video World Models

VideoGPT、iVideoGPT 等方法把视频压缩为视觉 tokens，再以自回归模型学习未来状态或动作条件
转移。VQ-VAE、FSQ 及相关 neural codec 工作说明 tokenization 在压缩率、重建失真与序列建模
成本之间进行权衡。既有研究通常把 reconstruction quality 视为表示能力或生成上界；本文关注
它在 post-training 中的另一后果：当 decoded candidates 与 codec 无法精确生成的 raw target
比较时，target residual 如何改变 group-relative candidate ordering。

## 2. Verifiable-Reward Post-Training and GRPO

GRPO 使用同一 prompt 或条件下的 sample group 构造相对优势，无需单独训练 value model。
RLVR-World 将这一范式用于世界模型，以 MSE、LPIPS 等任务指标评分 decoded predictions。本文
保留其 candidate sampling、full-reference metrics 和 group-relative update，不提出新的通用
RLVR 框架。区别在于：我们分析并校准 tokenized-video verifier 的 target support，并重新定义
multi-step autoregressive video 的时间信用单位。

现有通用 GRPO 改进主要调整 advantage normalization、clipping、importance ratio 或样本选择。
这些 optimizer interventions 不直接解决 raw target 与 frozen codec 输出集合的失配；本文因此只把
它们作为优化背景，不把通用优化器替换纳入贡献主线。

## 3. Full-Reference Metrics and Codec Reconstruction

MSE、SSIM 和 LPIPS 分别从像素、结构和 learned-feature 空间衡量单样本 full-reference fidelity。
它们若在 decoded image 上计算，均属于 post-decode evaluation；“像素指标与特征指标”的区别
不等于“是否经过 decoder”。FID、FVD、KID 等集合统计可评价分布，但不能直接替代本任务所需的
per-candidate verified reward。

本文不把增加 metric 数量作为贡献。RC 保留原始 MSE+LPIPS，改变的是 target support。最终
evaluation 仍相对 raw GT，因此该设计不是通过更换评测标准获得自洽提升。

## 4. Temporal Credit Assignment

经典 reinforcement learning 使用 return、n-step return 与 eligibility traces 把延迟 outcome 分配
给较早决策。近期生成模型后训练也开始把 trajectory-level reward 分解到 reasoning steps、segments
或 denoising stages。RC-GRPO 不声称发明 reward-to-go；它针对 action-conditioned autoregressive
video 定义 frame-block credit unit：一个未来帧的完整 visual-token block 会成为后续生成条件，
因而接收从当前位置到 rollout 末端的 verified frame rewards。

与 diffusion/flow 的 denoising-step credit 不同，这里的 temporal position 对应可单独解码和相对
真实未来帧验证的状态预测。候选身份打乱和 return 截断控制用于检验收益是否真正依赖该自回归
时间对应，而不是一般性的 reward splitting 或 rescaling。

## 5. Positioning

本文位于 tokenized video world modeling、verifiable-reward post-training 和 temporal credit
assignment 的交叉处。C1 处理“同组 candidates 应与什么 target 比较”，C2 处理“未来帧质量应
归因给哪些 visual-token blocks”。二者共同约束 post-training signal，但不依赖新的世界模型架构。
