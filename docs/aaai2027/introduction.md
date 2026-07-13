# Introduction 中文正式工作稿

## 段落任务

1. 世界模型后训练的意义，以及 RLVR 的吸引力；
2. tokenized video verifier 的输出空间错位；
3. multi-step GRPO 的时间信用错位；
4. 统一视角与方法；
5. 贡献与证据边界。

## 正式中文稿

世界模型需要根据当前视觉状态与动作预测未来状态，其价值取决于预测是否足以支持理解、规划和控制。自回归视频世界模型通常先把视频压缩为离散视觉 token，再以序列建模方式预测未来。最大似然训练优化 token likelihood，却不直接优化使用者关心的像素保真或感知质量。RLVR-World 因而提出用可自动计算的 MSE、LPIPS 等预测指标作为 verifiable rewards，并通过 group-relative policy optimization（GRPO）直接后训练世界模型。这一范式绕开人工偏好标注，为视频生成模型提供了简洁的任务对齐接口。

然而，verifiable 并不意味着对策略学习可靠。Tokenized video model 不直接输出任意像素图像；其预测必须经过冻结 decoder，因此只能落在 decoder 的可达输出集合内。真实 future frame 通常不严格属于该集合。即使策略预测了 ground-truth visual tokens，decoder 仍会留下 tokenizer reconstruction residual。该残差中与候选无关的常数部分会被 GRPO 组内去均值消除，但候选预测误差与残差的交互项会改变 pairwise ordering。于是，一个数值完全可计算的 decoded reward 仍可能向 GRPO 提供错误的相对偏好。

多步 rollout 进一步引入时间信用错位。Sequence-level GRPO 将整段 future prediction 压缩成一个 reward，并把同一优势广播给所有 future tokens，无法区分早期 frame block 对后续误差传播的责任。简单地改成逐帧优势也不能解决这一问题，因为它只奖励当前帧并切断延迟影响。近期视觉生成中的 stepwise GRPO 工作同样表明，生成轨迹内部的不同步骤不应共享粗粒度 outcome credit；但 diffusion/flow denoising steps 与 action-conditioned autoregressive future frames 具有不同的因果语义，也没有处理冻结 video codec 引起的 verifier rank corruption。

我们从 group-relative ranking reliability 统一处理这两个问题。首先，我们把 raw target 与其 tokenizer-reachable reconstruction 之间的失配操作化为 reconstruction-induced rank corruption，并用候选排序相关性预测 pairwise flip probability。基于该诊断，我们提出 reconstruction-calibrated（RC）verifier：将 ground truth 通过同一冻结 tokenizer-decoder 投影后再计算原有 MSE+LPIPS reward。其次，我们将 multi-step GRPO 的信用单位改为 future-frame token block，通过 temporal return 把后续预测质量分配给能够影响它的前序 block。进一步的候选扩展使用各 horizon 实测的 rank-flip probability 调制 temporal return；该扩展只有在离线重放与 paired training 通过后才进入正式方法。

当前证据支持两项贡献。第一，我们建立了从 reachable-target mismatch、相关性预测的排序翻转、同候选 RC 重评分到 paired training 改善的诊断—修复闭环。第二，我们提出 frame-block temporal-return GRPO：在当前 5-seed RT-1 multi-step protocol 下，其 full-reference fidelity 均值优于 sequence-level RC，frame-only 对照显著退化，且相对收益随 rollout horizon 增长。我们同时报告边界：额外 reward components、通用 GRPO 替换和 horizon-aware KL 未产生稳定增益，motion 与 distributional realism 也不支持全面改善。固定 $n=10$ 扩种、完整 verifier×credit 因子对照和 rank-reliable return 门控用于决定最终投稿版本的主张强度。

## 当前贡献句

1. **Verifier diagnosis and calibration.** 我们识别 tokenized video RLVR 中由不可达 raw target 引起的 candidate-dependent rank corruption，并用最小 RC verifier 完成机制与训练闭环。
2. **Video-structured temporal credit.** 我们将 GRPO 的 sequence-level advantage 重构为 frame-block temporal returns，并通过 frame-only、gain、KL 和 horizon 分析隔离有效结构。

## C3 通过后的替换句

只有 Rank-Reliable Return 通过预注册门控和训练后，才把第二条升级为：

> We introduce rank-reliable temporal credit assignment, which weights future-frame returns by horizon-wise verifier reliability measured from pairwise rank corruption, unifying reconstruction calibration and temporal credit under a single ranking principle.

## 边界

- 不声称 reward-to-go 本身是新理论；新颖性必须来自 tokenized video frame blocks、verifier reliability 和实证闭环；
- 不声称全面改善 motion/diversity/distribution；
- 不把负结果数量写成贡献；
- 不在 C3 验证前修改 tex 的标题和 contribution list。
