# RC-GRPO 投稿前审稿人审计

## Review Setup

- 评估范围：当前 Word 主线、`story.md`、`method.md` 与正式 `low_var_kl` 结果；
- 两项贡献：Reachability-Constrained Rank Calibration；Frame-Block Temporal Return；
- 已成立证据：三个 FSQ codec 的不可忽略的编解码重建误差、RT-1/VP2 same-candidate rank repair；RT-1 Temporal Return tail fidelity；
- 尚缺证据：episode-disjoint 复评、时间对应控制、长度压力测试、RoboNet 正式审计；
- 已删除主张：第三项贡献、统计超加和及跨平台普适 raw-GT gain。

## Reviewer 1：Novelty and Mechanism

**总体判断。** C1 的问题识别具有辨识度，但审稿人会把 $D(Q(E(s')))$ 称为显然的 target
preprocessing，也会指出 $\arccos(\rho)/\pi$ 是经典高斯符号关系。论文必须把新意放在“target-set
mismatch 如何进入 GRPO 的 candidate-ranking channel、同候选审计和跨 codec 闭环”，而不是声称
新数学定理。

**主要攻击。** Raw-anchored update 本质上使用通用 gradient projection；若跨平台 raw-GT gain
不能成立，它更像防护性工程而非方法贡献。C1 当前最稳的部分是 diagnosis 与 RC rank repair，
不是性能普适性。

**投稿前要求。** 明确区分常数重建残差与候选相关交互（candidate-dependent interaction）；保留 raw
evaluation；报告 VP2 training boundary，并明确 Cosmos 只做 codec floor audit；不得将 RT-1 三 seed
pilot 写成普适证明。

## Reviewer 2：Temporal Credit and Causality

**总体判断。** Reward-to-go 是经典构造，Temporal Return 的新颖性不能来自求和公式本身，而必须
来自 action-conditioned autoregressive frame blocks、per-horizon group normalization 和与 future
frame verification 对齐的实证设计。

**主要攻击。** episode-disjoint 复评中，Temporal Return 的五指标均值都更好，但 LPIPS 与
LPIPS-last 仅 4/5、paired $p=0.125/0.149$；MSE 虽为 5/5，paired $t$ test 也只有 $p=0.074$。
没有 candidate-shuffled control 时，收益仍可能来自 reward rescaling 或额外 normalization，而非
正确 temporal correspondence。

**投稿前要求。** 完成 $L=1/3$/full/shuffled controls；若通过，再做 $T=4/6/8$ stress test。保持
full-rollout LPIPS 为 primary、LPIPS-last 为预先指定 temporal endpoint，不事后交换角色。

## Reviewer 3：Experimental Rigor and Scope

**总体判断。** 三平台 rank audit 改善了外部有效性，但 raw-GT training gain 目前只在 RT-1 pilot
出现。8-episode/32-window 的 episode-disjoint 复评已解决原 readout 的 episode overlap，但完整
组合相对原始 sequence+raw 仅 2/5 seeds 改善 LPIPS，不能包装成稳健 headline gain。

**主要攻击。** 大量历史变体会引发 multiple experimentation 质疑；若新旧 KL 实现静默混写、
或不披露 seed/\(n\)/checkpoint 规则，可信度会受损。

**投稿前要求。** 主表写清 episode-disjoint manifest、每个 arm 的 seed 数、实现版本与 checkpoint
选择规则。旧 linear-KL 只可作附录敏感性，标签须清楚。负结果可压成一张边界表；RoboNet 未完成
前不把其写成已验证贡献。

## Cross-Review Synthesis

**共识优势：** 问题入口清楚；same-candidate design 能隔离 target 变化；三平台 rank mechanism
不是单一 RT-1 偶然；Temporal Return 在 tail endpoint 上已有配对支持。

**共识风险：** C1 可能被视为 target preprocessing，C2 可能被视为标准 reward-to-go；raw-anchored
update 缺乏跨平台 performance conversion；full-rollout primary 仍为趋势。

**强投稿优先闭环（可按算力裁剪）：**

1. C1：残差分解 + 三平台 same-candidate audit + raw evaluation + training boundary 说明；
2. C2：主实现下的 paired multi-step 结果 + 尽量有 shuffled/truncated 或 horizon 控制；
3. 评测：尽量 episode-disjoint 复评；图/表披露 seed 与 checkpoint 规则；历史实现可并列但须标注。

## Unsupported Claims

- RC/raw-anchored update 在三个平台都改善 raw-GT fidelity；
- $\arccos(\rho)/\pi$ 是本文原创理论；
- Temporal Return 或 reward-to-go 本身由本文首次提出；
- RC 与 Temporal Return 具有统计超加和作用；
- 方法全面改善 motion、diversity 或 distributional realism；
- 在不同训练 recipe 下全面击败 RLVR-World。
