# RC-GRPO 投稿前审稿人审计

## Review Setup

- Input scope：当前 story、method、experiments 与已有 RT-1 结果；
- Assessment boundary：Rank-Reliable Return、固定 $n=10$、raw-return 因子臂和长度泛化尚未完成；
- Shared claim：tokenized video RLVR 存在输出空间 rank corruption 与时间信用错位，RC verifier 和 frame-block temporal returns 分别处理两者；
- Visible evidence：同候选 flip 闭环、5-seed RC/return 结果、frame-only 强消融、horizon trend、多个负结果；
- Missing evidence：确认性 $n=10$、完整 $2\times2$、跨 horizon length、第二 tokenizer/dataset、C3 方法结果。

## Reviewer 1

- **Overall assessment：** 问题定位有价值，rank corruption 的机制链是目前最独特部分；但当前 full method 仍由 target projection 与经典 reward-to-go 构成，算法新颖性不足以自动达到强接收。
- **Who would be interested：** world-model post-training、video RLVR、tokenized generation 和 fine-grained policy optimization 研究者。
- **Major strengths：** 区分 constant floor 与 candidate-dependent perturbation；同候选闭环避免只凭最终训练曲线讲机制；对 decoded feature metrics 与 pre/post-decode 轴的澄清准确。
- **Major concerns：** Temporal Return 本质接近标准 Monte Carlo reward-to-go；若没有 rank-reliable temporal component 或更强的 video-specific causal evidence，标题中的 credit assignment 可能被视为直接移植。
- **Technical failings to resolve：** 完成 fixed $n=10$；补 raw-return 形成 factorial interaction；证明 later-horizon signature 不是 seed/horizon 伪重复；C3 必须有 replay gate、weight controls 与 paired training。
- **Assessment：** originality 中等，technical motivation 强，method novelty 当前偏弱，readability 经重构后可接受。
- **Recommendation posture：** borderline；C3 成功或 $2\times2$+length scaling 形成强证据后可上调。

## Reviewer 2

- **Overall assessment：** 实验纪律优于一般小规模论文，但确认性统计与外部有效性仍不足。
- **Who would be interested：** 关注 RLVR 稳定性、verifier hacking、视频预测评测和 reproducible RL 的读者。
- **Major strengths：** matched seeds、raw evaluator 不随训练 target 改变、明确报告 DINO/motion 边界、没有把失败的 GSPO/REAL 结果隐藏成成功。
- **Major concerns：** 当前 return-vs-seq 的 primary LPIPS 只有 $n=5,p=0.212$；大量历史变体可能让审稿人担心 multiple experimentation；单一 RT-1 数据与单一 codec 限制普适性。
- **Technical failings to resolve：** 固定 $n=10$ 后停止；预先指定 primary/secondary endpoints；episode-cluster 处理 calibration；完整报告所有 seeds；最好增加 $T=4/6/8$ stress test。
- **Assessment：** technical soundness 尚可，但主要性能主张目前是 trend，不是确认性结论；significance 依赖新增实验。
- **Recommendation posture：** weak reject 到 borderline；统计补强是硬门槛。

## Reviewer 3

- **Overall assessment：** 新故事比“多指标 reward engineering”清晰得多，但论文必须保持克制，避免把系统性负结果包装成贡献。
- **Who would be interested：** 希望理解何时 verifiable metric 能转化为有效 policy signal 的生成模型和 RL 社区。
- **Major strengths：** “verifiable does not imply reliable ranking”是清楚的入口；输出空间与时间轴的双层结构适合一张方法总图；边界结果增强可信度。
- **Major concerns：** C1/C2 若作为两个独立模块仍显拼接；Rank-Reliable Return 是最自然的统一件，但目前完全未验证。标题、摘要和图示不能提前使用 C3。
- **Technical failings to resolve：** 统一术语；主图展示 raw target mismatch、frame-block return 和 reliability profile；Results 只保留与 claim 直接相关的负证；删除旧 CAST/MG-RC 叙事痕迹。
- **Assessment：** readability 和 reuse 潜力较好；当前 broad significance 仍受单一 benchmark 限制。
- **Recommendation posture：** borderline，取决于方法统一和实验闭环。

## Cross-Review Synthesis

- **Consensus strengths：** reconstruction-induced rank corruption 是可信且有辨识度的机制；RC 不更换最终 evaluator；frame-only 强消融支持 future-return structure。
- **Consensus technical risks：** Temporal Return 的经典性、$n=5$ 统计不足、缺 raw-return 因子臂、C3 尚无证据、单一数据与 codec。
- **Where emphasis differs：** Reviewer 1 更关注算法新颖性；Reviewer 2 更关注统计与 protocol；Reviewer 3 更关注统一叙事和过度主张。
- **Broad-interest readout：** 如果论文证明 codec-induced ranking reliability 可以预测并改进 temporal credit，它会超出一个 RT-1 reward trick；若只剩 target reconstruction + reward-to-go，意义更局部。
- **Most important issues：** 依次完成 fixed $n=10$、raw-return factorial arm、Rank-Reliable replay gate；前两项是现有论文硬门槛，第三项决定方法上限。

## Risk / Unsupported Claims

- “首次提出 temporal return”不支持；
- “空间/metric/GRPO 设计空间已穷尽”不支持；
- “改善动态与分布质量”被现有数据否定；
- “Rank-Reliable Return 理论保证恢复干净梯度”不支持；
- “全面击败 RLVR-World”不支持，只能比较相同 held-out protocol。
