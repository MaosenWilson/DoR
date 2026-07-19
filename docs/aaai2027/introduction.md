# Introduction 中文工作稿

## Paragraph Structure

1. 动作条件视频世界模型与可验证奖励后训练的价值；
2. 冻结视觉 codec 引起的 target-set mismatch；
3. multi-step sequence-level GRPO 的时间信用错位；
4. RC-GRPO 的两部分方法；
5. 两项贡献、证据与边界。

## Draft

动作条件视频世界模型通过预测动作作用下的未来视觉状态，为机器人规划、交互式仿真和基于模型
的决策提供环境动态。许多视频世界模型先用视觉 tokenizer 将图像压缩为离散 token，再由自回归
模型生成未来 token trajectory。最大似然训练优化 token likelihood，却不直接优化最终解码视频的
像素或感知保真度。可验证奖励后训练因此从同一条件采样多条候选未来，以 MSE、LPIPS 等自动
指标评分，并利用 group-relative policy optimization 直接提高组内较优候选的概率。这一范式无需
人工偏好标注，但其可靠性取决于 verifier 给出的相对偏好是否适合驱动策略更新。

在 tokenized 视频模型中，可计算的 full-reference error 不必然产生可靠的 candidate ordering。
模型的预测必须经过冻结 decoder，因此只能落在该 tokenizer-decoder 的可生成集合中，而 raw
future frame 通常不能被该接口精确重建。raw target 与其 codec reconstruction 之间的 residual
不仅产生一个候选共享的常数项，还会通过与 candidate error 的交互改变组内相对间隔甚至翻转
candidate pairs。由于 GRPO 主要消费组内排序，这种 target-set mismatch 会直接进入策略梯度；
简单减去常数重建残差并不能修复排序。

多步自回归生成进一步引入时间信用错位。Sequence-level GRPO 将整段未来预测聚合为一个 reward，
并把同一优势广播给所有 future-frame visual-token blocks。然而，较早 frame block 会成为后续预测
条件，较晚 block 不能反向影响已经生成的帧。统一 rollout scalar 无法表达这种定向依赖；另一种
直观方案是为每帧只使用当前 reward，但它又切断了早期预测对未来误差的延迟影响。因此，视频
后训练不仅需要可靠地比较同组 candidates，还需要把 future quality 分配给真正能够影响它的
visual-token blocks。

为同时处理两个错位，我们提出 RC-GRPO，并保持候选采样、冻结视觉 tokenizer、decoder、世界
模型架构和最终 raw-frame evaluation 不变。在 verifier 一侧，可达性审计在同一 candidate group
上量化 raw target 与 codec-reachable target 造成的 rank disagreement；RC verifier 随后保留原始
MSE+LPIPS，只把比较目标替换为 $D(Q(E(s')))$。为避免校准方向牺牲 raw fidelity，我们在同一
candidates 上构造 raw 与 RC 的 group-relative surrogates，并把 RC gradient 投影到至少保持 raw
surrogate 一阶进度的半空间。在 temporal-credit 一侧，第 $t$ 个 future-frame token block 获得从
$t$ 到 rollout 末端的 discounted frame rewards，并只在相同时间位置的候选组内标准化。

本文有两项贡献。第一，我们把冻结 codec 的 target-set mismatch 识别并验证为 GRPO 的 candidate-
ranking failure，提出由可达性审计、RC verifier 和 raw-anchored update 组成的可达性约束排序校准
框架。CNN-FSQ、压缩 FSQ 与 NVIDIA Cosmos DV-FSQ 三个独立 codec 实例上，encode–decode 重建误差均不可忽略；RT-1 与 VP2-RoboSuite 的 same-candidate audits 又在不同 tokenizer 和生成架构上显示
rank agreement 提高、pairwise flips 减少。Cosmos 只用于独立 codec 重建误差审计，不被写成候选
排序实验；跨平台 raw-GT training conversion 仍作为明确边界。第二，我们提出 future-frame
token-block Temporal Return，把后续帧质量分配给
能够影响它的前序 visual blocks。正式 RT-1 `low_var_kl` 协议中，该方法相对 sequence-level RC
在训练未见的 8 episodes / 32 windows 上取得五个 fidelity metrics 的更好均值，其中 MSE 为 5/5
paired seeds 改善，LPIPS、LPIPS-last、PSNR 和 SSIM 为 4/5。现有 paired $t$ tests 尚未达到双侧
0.05，本文不声称全面改善 motion、diversity 或 distributional realism。

## Claim Boundary

- C1 的跨平台证据目前是 rank mechanism，不是跨平台 raw-pixel performance gain；
- C2 当前为 episode-disjoint、跨五指标的方向证据，最强读数是 MSE 5/5；
- 主文优先报告正式 `low_var_kl`；旧 linear-KL 可作为附录或敏感性，协议标签须清楚；
- 不声称统计超加和，除非新的交互检验明确支持；
- 所有最终指标相对 raw future frames，不能用 RC target 自评。
