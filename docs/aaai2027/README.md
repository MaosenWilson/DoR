# AAAI-2027 文档索引

## Canonical Documents

| file | role |
|---|---|
| `story.md` | 唯一故事线、贡献与证据状态 |
| `method.md` | 当前方法公式与 pending C3 定义 |
| `experiments.md` | 可复核结果、统计协议、缺失实验 |
| `RUN.md` | 仅包含尚需执行的命令 |
| `introduction.md` | 中文正式 Introduction 工作稿 |
| `related_work.md` | 中文 Related Work 与真实文献定位 |
| `preliminaries.md` | 问题定义和 GRPO 基础 |
| `reviewer_audit.md` | 三审稿人投稿前审计 |
| `reference_plan.md` | 引用核验与 BibTeX 计划 |

## Source of Truth Order

出现冲突时按以下优先级：

1. `story.md` 的 claim/status；
2. `experiments.md` 的结果事实；
3. `method.md` 的公式；
4. 其他中文写作稿；
5. Git history preserves superseded material; the working tree contains only the
   current paper contract.

## Status Labels

- `supported`：已有直接证据；
- `supported trend`：方向一致但确认性统计不足；
- `pending`：已设计但未验证；
- `rejected`：按预注册判负；
- `archived`：不再服务当前论文。

## Workflow

先更新 `story.md` 的 claim-evidence contract，再写代码和运行；结果进入 `experiments.md`，通过后才同步到 Method/Introduction/tex。日期型临时材料只放本地 `tmp/`，不得提交或在文档目录创建新的状态稿。
