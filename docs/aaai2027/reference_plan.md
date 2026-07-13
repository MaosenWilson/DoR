# 参考文献与真实性核验计划

> 正式 bibliography：`docs/AuthorKit27/aaai2027.bib`。不以接近参考论文数量为目标；每条引用必须支持正文中的具体论点，并由论文原文、官方 proceedings 或官方项目页核验。

## 1. 主线必需来源

### World models and tokenized video

- World Models；RSSM/Dreamer；VideoGPT；iVideoGPT；FSQ；VQ-VAE/VQGAN；RT-1/Open X-Embodiment。

### RLVR and policy optimization

- RLVR-World，NeurIPS 2025，arXiv:2505.13934；
- GRPO / DeepSeekMath；PPO；RLOO；Dr.GRPO（只在确有正文需要时保留）。

### Metrics and codec limits

- LPIPS；SSIM；perception-distortion tradeoff；DISTS；KID/FVD/DINOv2 仅在 boundary experiment 出现时引用。

### Temporal credit assignment

- Stepwise-Flow-GRPO，arXiv:2603.28718；
- TurningPoint-GRPO，arXiv:2602.06422；
- OTCA，arXiv:2604.19234；
- AdaGRPO，arXiv:2606.08480；
- SAGE-GRPO 官方论文/代码，只有在核验论文元数据后加入 bib。

## 2. 引用职责

| manuscript claim | citation family |
|---|---|
| tokenized world models predict future visual tokens | VideoGPT/iVideoGPT/FSQ |
| RLVR directly optimizes task metrics | RLVR-World |
| GRPO uses group-relative advantages | GRPO source |
| decoded perceptual fidelity | LPIPS/SSIM |
| stepwise visual credit is active research | Stepwise-Flow/TP-GRPO/OTCA |
| reward reliability can gate optimization | AdaGRPO/SAGE-GRPO |
| distributional readout differs from per-sample reward | KID/FVD/DINOv2 sources |

## 3. 必须人工复核的现有 BibTeX keys

以下条目可能真实，但投稿前必须逐字段核对作者、标题、venue/year 和 arXiv ID：

- `grpolambda`, `spo`, `sdgrpo`, `oar`, `noiseawaregrpo`；
- `gspo`, `realrlvr`, `sapo`；
- `genie`, `dists`, `cmmd`, `fddinov2`。

未核验条目不得仅因为已经在 `.bib` 中就进入正文。

## 4. 待新增条目

在正式 related-work tex 写入前，为 TP-GRPO、OTCA、AdaGRPO 和 SAGE-GRPO 建立经过原文核验的 BibTeX。若找不到官方作者/venue 信息，保留 arXiv `@misc`，不猜测会议录用状态。

## 5. 删除原则

- 正文不讨论的 generic GRPO variants 删除；
- 不展示对应 metric 的评价文献删除；
- 不用综述替代原始论文；
- 不用 GitHub README 支撑论文中的性能数字；代码仓库只用于 implementation attribution。
