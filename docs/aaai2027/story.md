# RC-GRPO：统一故事、贡献与证据边界

## 1. Core Claim

Tokenized 视频世界模型的可验证奖励后训练存在两个相互衔接的学习信号错位：decoded candidate
被拿去和模型无法精确生成的 raw frame 比较，可能改变 GRPO 消费的组内排序；multi-step GRPO
又把一个 rollout-level advantage 广播给不同时间位置的 visual-token blocks，无法表达早期预测对
后续帧误差的定向影响。RC-GRPO 分别校准比较参照和时间信用单位，同时保持世界模型、视觉
tokenizer、decoder、候选采样和最终 raw-GT evaluation 不变。

## 2. Why It Matters

动作条件视频世界模型通过生成未来视觉状态支持规划、仿真和基于模型的决策。RLVR-style
post-training 的吸引力在于：从同一状态和动作条件采样候选未来，使用 MSE、LPIPS 等自动指标
评分，再由 GRPO 提高组内较优候选的概率。然而，reward 可计算不等于其相对偏好适合驱动策略
更新。GRPO 主要消费候选间的相对排序，因此 target support 与时间信用中的结构性错位不会因为
reward 是 full-reference metric 而自动消失。

## 3. Existing Pipeline and Two Gaps

给定条件 $q=(s_{1:C},a_{1:H})$，世界模型采样 $G$ 条未来 token trajectories，冻结 decoder
生成 $\hat s_{i,h}$。原始流程计算

$$
r^{\mathrm{raw}}_{i,h}
=-
\left[
\operatorname{MSE}(\hat s_{i,h},s'_h)
+\operatorname{LPIPS}(\hat s_{i,h},s'_h)
\right],
$$

再把逐帧分数聚合为 rollout reward 并构造 group-relative advantage。

### 3.1 Target-Set Mismatch

冻结 tokenizer-decoder 只允许候选落在其可生成集合

$$
\mathcal S_{\mathrm{codec}}=\{D(z):z\in\mathcal Z\}
$$

中，而 raw future frame $s'_h$ 通常不严格属于该集合。令其可达重建为

$$
\tilde s'_h=D(Q(E(s'_h))).
$$

对平方误差，raw-target reward 可分解为 candidate error、与候选无关的 reconstruction term，
以及 candidate error 与 target residual 的交互项。组内中心化只能消除常数项，不能消除随候选
变化的交互项，因此 raw 与 reachable target 可能给出不同的 candidate ordering。真正需要诊断
的是同一 candidate group 上两种 target 造成的 rank disagreement，而不是只报告 reconstruction
floor 的绝对大小。

### 3.2 Temporal-Credit Mismatch

Sequence-level GRPO 将整个未来 rollout 压缩为一个标量，并把同一优势施加到所有 future-frame
token blocks。自回归视频中，较早 block 会成为后续预测条件，而较晚 block 不能反向影响已经生成
的帧。统一标量因此混合了不同 block 的责任；仅使用当前帧 reward 的 frame-only advantage 又会
切断延迟影响。

## 4. Method

### 4.1 Reachability Audit and RC Verifier

可达性审计在训练前对完全相同的 candidate group 分别计算 raw 与 reachable-target reward，报告
Spearman rank correlation、pairwise disagreement 和与 decoder-input representation error 的一致性。
它判断 target mismatch 是否实际进入 GRPO 的排序通道。

RC verifier 保留原始 MSE 与 LPIPS，只把比较目标替换为同一冻结 codec 的重建：

$$
r^{\mathrm{RC}}_{i,h}
=-
\left[
\operatorname{MSE}(\hat s_{i,h},\tilde s'_h)
+\operatorname{LPIPS}(\hat s_{i,h},\tilde s'_h)
\right].
$$

最终评测仍全部相对 raw $s'_h$，避免用训练 target 自我证明。

### 4.2 Raw-Anchored Calibrated Update

单纯改用 RC target 可能改善 reachable-space ordering，却不保证有限步训练后的 raw-GT fidelity。
因此同一 candidate group 同时构造 raw 与 RC 的 GRPO surrogate gradients。更新方向取最接近 RC
gradient、同时不低于 raw surrogate 一阶进度的半空间投影。该模块的目标是让 raw verifier 充当
保真锚点、RC verifier 充当排序校正方向，而不是手工加权两个 reward。

该更新在 RT-1 三种子 pilot 中通过接线和方向门，但 VP2 尚未建立跨平台 raw-GT training
gain。因此它是当前方法链的一部分，也是投稿前必须继续验证的薄弱环节；不能提前声称普适提升。

### 4.3 Frame-Block Temporal Return

对第 $t$ 个 future-frame token block，计算它能够影响的后续帧回报

$$
G_{i,t}=\sum_{h=t}^{H}\gamma^{h-t}r_{i,h},
$$

并只在相同时间位置的 $G$ 个 candidates 间标准化。所得 $A_{i,t}$ 只权重第 $t$ 个 visual-token
block 的 log probability。这样保留 future reward-to-go，同时避免把一个 rollout scalar 广播到
全部时间位置。

## 5. Contributions

### C1. Reachability-Constrained Rank Calibration

本文把 raw target 与冻结 codec 可生成集合的失配操作化为 candidate-ranking failure，而不是把
reconstruction residual 当作会被 GRPO 自动消除的常数。贡献包括同候选可达性审计、RC verifier
以及 raw-anchored calibrated update。该重建误差本身在 CNN-FSQ、压缩 FSQ 与 NVIDIA Cosmos DV-FSQ
三个独立 codec 实例中均为非零；在具有可复用候选生成器的 RT-1 与 VP2-RoboSuite 上，同候选
审计进一步显示 rank agreement 提高、pairwise flips 减少。两类证据分别回答“失配是否跨
codec 存在”和“校准能否跨世界模型修复排序”，但不把 Cosmos 冒充为候选排序实验。跨平台
raw-GT training conversion 仍是证据边界。

### C2. Frame-Block Temporal-Return Credit Assignment

本文针对 action-conditioned autoregressive video，把 GRPO 的时间信用单位改为 future-frame
visual-token block，并向每个 block 分配其可影响的后续 frame rewards。正式 `low_var_kl` RT-1
协议的 episode-disjoint 复评中，Temporal Return 相对 sequence-level RC 在五个 fidelity metrics
上均取得更好均值：MSE 为 5/5 paired seeds 改善，LPIPS、LPIPS-last、PSNR 和 SSIM 为 4/5；所有
paired $t$ tests 仍未达到双侧 0.05。时间对应控制和 rollout-length stress test 决定最终主张强度。

## 6. Current Evidence

| claim | evidence | status |
|---|---|---|
| lossy FSQ codecs induce non-negligible encode–decode reconstruction residuals | CNN-FSQ, compressive FSQ, Cosmos DV-FSQ | supported across three codec instances |
| raw/reachable mismatch changes group ranking | same-candidate audits across RT-1 and VP2 | supported |
| RC reduces rank corruption | RT-1 and VP2 $\Delta\rho>0$, $\Delta$flip$<0$ | supported |
| raw-anchored update gives universal raw-GT gain | RT-1 pilot positive; VP2 conversion not established | unsupported universally |
| Temporal Return improves RT-1 tail fidelity | episode-disjoint, LPIPS-last 4/5, paired $p=0.149$ | directional support |
| Temporal Return improves full-rollout LPIPS | episode-disjoint, LPIPS 4/5, paired $p=0.125$ | directional support |
| Temporal Return improves RT-1 MSE | episode-disjoint, 5/5, exact one-sided $p=0.031$ | strongest current training evidence |
| benefit comes from correct candidate-time correspondence | $L=1/3$/shuffled controls | pending |
| RC and Temporal Return are statistically super-additive | $2\times2$ interaction crosses zero | unsupported; claim removed |

## 7. Remaining Experiments

1. **Temporal correspondence controls.** 比较 $L=1$、$L=3$、full return 与 candidate-shuffled
   return，验证 future reward 必须沿同一 candidate trajectory 对齐。
2. **Horizon-length stress test.** 在前一控制通过后比较 $T=4,6,8$，不按长度单独调参。
3. **RoboNet external validation.** 数据下载和 provenance 完成后，使用公开 action-conditioned
   iVideoGPT checkpoint 扩展跨机器人域审计；未产生结果前不进入主表。

## 8. Claim Boundaries

可以写：RC 改善跨平台 same-candidate rank agreement；Temporal Return 在 episode-disjoint RT-1
复评中取得五个 fidelity metrics 的更好均值，并在 MSE 上呈 5/5、其余指标呈 4/5 方向一致性。

谨慎写 / 默认不写：RC 或 raw-anchored update 在三个平台都改善 raw pixels；全面改善 motion、
diversity 或 distributional realism；击败 RLVR-World 的完整训练 recipe；从四臂最低均值推导
超加和贡献。旧 linear-KL 与 `low_var_kl` 结果可以并存，但必须分栏/分表标注实现，不能静默
当成同一协议。

## 9. Canonical Terminology

| term | meaning |
|---|---|
| codec-reachable set | 冻结 tokenizer-decoder 能生成的图像集合 |
| reachable target | $D(Q(E(s')))$ |
| target-set mismatch | raw target 通常不严格位于 codec-reachable set |
| reconstruction-induced rank corruption | target residual interaction 引起的 candidate ordering change |
| reachability audit | 同候选 raw/RC 排序分歧诊断 |
| RC verifier | 对 reachable target 计算原 MSE+LPIPS 的 verifier |
| raw-anchored update | 保持 raw surrogate 一阶进度的 calibrated gradient projection |
| Temporal Return | future-frame token block 的 reward-to-go advantage |
