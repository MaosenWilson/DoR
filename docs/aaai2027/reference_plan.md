# 参考文献计划

> 目标：参考 AAAI-26 论文的引用规模，但不为了凑数量而加文献。所有进入最终 bib 的文献必须能定位到真实论文、项目页或官方出版记录。

## 1. AAAI-26 参考论文统计

参考论文：

**Pre-Trained Video Generative Models as World Simulators**  
AAAI-26, 9 pages.

本地文件：

`docs/aaai2027/papers/AAAI2026_PreTrained_Video_Generative_Models_as_World_Simulators.pdf`

结构观察：

- Abstract
- Introduction
- Related Work
- Preliminaries
- Method
- Experiments
- Conclusion and Limitation
- Acknowledgments
- References

参考文献数量：约 **62** 条。

引用类型分布：

- video generation / video world models：约 20 条。
- model-based RL / world models：约 15 条。
- datasets / environments / benchmarks：约 8 条。
- video prediction metrics and perception metrics：约 5 条。
- architecture / foundation model references：约 6 条。
- downstream RL / offline RL algorithms：约 8 条。

对我们论文的启发：

1. AAAI 方法论文可以接受 50 条上下的 reference scale。
2. 引言和 related work 不需要堆满引用；重要的是每组引用服务一个明确论点。
3. 实验指标、数据集、基础架构、优化算法都需要有对应来源。

## 2. 我们当前 bib 状态

当前 `docs/aaai2027/references.bib` 与 `docs/AuthorKit27/aaai2027.bib` 一致，共 **52** 条。

已覆盖：

- RLVR / GRPO：`rlvrworld`, `grpo`, `deepseekr1`, `ppo`, `rloo`, `drgrpo`, `dapo`, `gspo`, `realrlvr`, `sapo`
- fine-grained credit assignment：`grpolambda`, `spo`, `sdgrpo`, `stepwiseflowgrpo`, `oar`, `noiseawaregrpo`
- tokenized video / world models：`ivideogpt`, `genie`, `dreamerv3`, `rt1`, `openxembodiment`
- world-model / video-prediction related work：`dws`, `worldmodels`, `rssm`, `dreamerv2`, `videogpt`, `magvit`, `maskvit`, `fitvid`, `avid`, `irasim`, `gamengen`, `learninginteractive`
- tokenizers / reconstruction：`fsq`, `vqvae`, `vqgan`, `perceptiondistortion`
- metrics：`lpips`, `ssim`, `dists`, `fid`, `kid`, `fvd`, `cmmd`, `fddinov2`, `dinov2`, `densitycoverage`, `precisionrecall`, `raft`

当前判断：

- 52 条已经接近完整 AAAI 方法论文规模。
- 后续不再为了数量加文献；只在正文出现新 claim、新 baseline 或新 metric 时补。
- 不建议强行追到 62 条；参考论文覆盖的是更宽的 video generation + MBRL + offline RL 场景，而我们论文范围更窄。

## 3. 主文推荐引用规模

目标主文引用数量：**45--52** 条。

分配建议：

| 位置 | 数量 | 作用 |
|---|---:|---|
| Introduction | 10--14 | 建立 video world model、RL post-training、tokenized verifier 的问题 |
| Related Work | 20--25 | 三组：video world model / RLVR-GRPO / credit assignment |
| Method | 6--8 | RLVR, GRPO, tokenizer, LPIPS/SSIM 等必要来源 |
| Experiments | 8--10 | RT-1/Open X-Embodiment, metrics, baselines |
| Limitations / Appendix | 3--5 | drift、distribution metrics、negative variants |

## 4. 建议保留的核心文献

必须保留：

- `rlvrworld`：直接基座。
- `grpo`：GRPO 来源。
- `ivideogpt`：tokenized interactive video world model 基座。
- `fsq`：tokenizer。
- `lpips`, `ssim`：主要 full-reference metrics。
- `rt1`, `openxembodiment`：机器人数据来源。
- `grpolambda`, `spo`, `stepwiseflowgrpo`：temporal/fine-grained credit assignment 相关定位。
- `vqvae`, `vqgan`, `perceptiondistortion`：重建地板和 tokenized generation 背景。

建议保留但不必都在主文展开：

- `drgrpo`, `dapo`, `gspo`, `realrlvr`, `sapo`：作为“通用 GRPO/RLVR variants”的相关工作或负结果背景。
- `fid`, `kid`, `fvd`, `dinov2`, `fddinov2`, `densitycoverage`, `precisionrecall`：如果正文或附录写分布级 evaluator，则保留；否则部分移到附录或删掉。
- `dreamerv3`, `genie`：world model 背景。

## 5. 可能需要新增的文献

以下只列“方向”，不直接加入 bib，等具体写到相关段落时再核验。

### Video prediction / tokenized video

- 已补：`videogpt`, `magvit`, `maskvit`, `fitvid`。
- 已补：`dws`, `avid`, `irasim`, `gamengen`, `learninginteractive`。
- 暂不补：Genie 2、Vid2World、Open-Sora Plan、Playable Game Generation、The Matrix。除非 related work 需要专门讨论 “interactive video generation” 的最新分支。

### RL and world models

- 已补：`worldmodels`, `rssm`, `dreamerv2`。
- PPO 已有；如果实验不直接比 PPO，不需要更多 model-free RL 文献。

### Metrics

- PSNR 原始或标准引用可补；当前 `ssim` 与 `lpips` 已够。
- FVD 已有；若不写 video distribution metric，可弱化。

### Credit assignment / RL optimization

- 只保留和“coarse sequence-level advantage”直接相关的文献。不要把所有 2025--2026 GRPO 变体都堆上来。

## 6. 需要人工谨慎核验的条目

这些不是说一定假，而是最终投稿前应手动核一次 BibTeX：

- `rloo`
- `genie`
- `dists`
- `fddinov2`
- `cmmd`
- `noiseawaregrpo`

## 7. 当前引用策略

最终写作原则：

1. **不以数量取胜**。AAAI-26 参考论文约 62 条，我们目标 45--50 条即可。
2. **每组引用只承担一个论点**。例如 world model 背景一组，GRPO 背景一组，credit assignment 一组。
3. **负结果相关文献少引**。REAL/GSPO/SAPO 可以出现在 related work 或 ablation 描述中，但不要制造“必须全面公平对比所有 LLM RL optimizer”的审稿期待。
4. **指标文献跟着指标走**。如果主文不展示 FD-DINO/KID/PRDC，就不要在 Introduction 里提前占引用。
