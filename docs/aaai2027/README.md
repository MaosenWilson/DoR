# AAAI-2027 文档索引

本目录只保留当前论文需要的文档。历史方案、逐轮开发记录和已完成命令由 Git history 保存，
不再复制到工作区。

## Canonical Documents

| file | single responsibility |
|---|---|
| `RC-GRPO_摘要与引言核心段落.docx` | 当前主线与拟固定贡献的中文沟通稿 |
| `story.md` | 唯一故事线、贡献结构和主张边界 |
| `method.md` | 当前方法公式与实现合同 |
| `experiments.md` | 正式协议、可复核结果、精简负结果和缺失证据 |
| `RUN.md` | 仅保留尚未完成且允许执行的命令 |
| `introduction.md` | 中文 Introduction 工作稿 |
| `preliminaries.md` | 问题定义、GRPO 和术语 |
| `related_work.md` | 与当前两项贡献直接相关的文献定位 |
| `reviewer_audit.md` | 当前版本的投稿风险审计 |
| `reference_plan.md` | 引用职责和真实性核验清单 |

## Source of Truth

1. Word 文档决定论文讲哪两个问题、采用哪条方法主线；
2. `experiments.md` 决定哪些结果可以写成已证实事实；
3. `method.md` 决定公式、符号和实现；
4. `story.md` 负责把主线与证据边界合并；
5. Markdown 完成核对后才同步到 LaTeX。

若 Word 中的拟定主张尚未通过 `experiments.md` 的证据门，只能在写作稿中标为待确认，不能
用叙事优先级覆盖实验事实。

## Current Contribution Contract

- **C1：Reachability-Constrained Rank Calibration.** 发现 raw target 与冻结视觉 tokenizer
  可生成集合失配会改变候选组内排序；方法由可达性审计、RC verifier 和 raw-anchored update
  组成。跨平台 rank mechanism 已成立，跨平台 raw-GT training gain 尚未成立。
- **C2：Frame-Block Temporal-Return Credit Assignment.** 将 multi-step GRPO 的信用单位从
  rollout scalar 改为 future-frame token block，并向每个 block 分配其可影响的后续帧回报。
  RT-1 正式实现下已有五种子方向证据，episode-disjoint 复评与时间对应控制待完成。

不存在当前 C3；已判负的 reward 扩展、辅助 verifier 和通用 GRPO 替换均不再写成候选贡献。

## Workflow

先更新 `story.md` 的 claim-evidence contract，再写代码；结果进入 `experiments.md`，通过后同步
到 Method、Introduction 和 TeX。

## Reporting Policy（可放宽，非钉死）

主表数字的**默认优先**是完整 paired seeds、固定 final checkpoint、正式 `low_var_kl` 实现。
允许在论文呈现时做**有记录的选择**，例如：

- 在完整 seed 池中报告均值 / 中位数，并可视需要强调一致方向（wins）；
- 在预声明的评测时间点集合内选择 checkpoint（final 优先，必要时用 last-K 中较好的一步，需在表注写明规则）；
- 将 linear-KL 或其他历史实现放在附录 / 敏感性分析，或在主文注明协议差异后并列；
- 若个别 seed 因工程故障缺失，报告可用子集并披露 \(n\)，不伪装成完整池。

底线只有两条：**不编造未跑出的数字**；**主文数字必须能回溯到 JSON artifact 与协议说明**。
「最有说服力的呈现」优先于机械执行某条硬规则。
