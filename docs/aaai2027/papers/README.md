# Upstream Paper Sources

本目录保存外部平台与 RLVR-World 对照所使用的开放获取论文原文。论文版本、来源与本地
文件固定如下；架构比较以这些原文为准，不以二手博客或仓库简介替代。

| 论文 | 会议/版本 | 官方开放来源 | 本地文件 | SHA-256 |
| --- | --- | --- | --- | --- |
| iVideoGPT: Interactive VideoGPTs are Scalable World Models | NeurIPS 2024；arXiv v3 (2024-10-31) | https://arxiv.org/abs/2405.15223 | `iVideoGPT_NeurIPS2024_arXiv2405.15223.pdf` | `e79681befb89de01b4ecbb1abe0f4fa15b60420fe195371f95dcc8b607251984` |
| Cosmos World Foundation Model Platform for Physical AI | arXiv:2501.03575v3 (2025-07-09) | https://arxiv.org/abs/2501.03575 | `Cosmos_World_Foundation_Model_arXiv2501.03575.pdf` | `c3c61f0a6f9a32700fc47d9898db573628d72bdea07562ca319ceba046e95c84` |
| RLVR-World: Training World Models with Reinforcement Learning | NeurIPS 2025；arXiv v2 (2025-10-25) | https://arxiv.org/abs/2505.13934 | `RLVR-World_NeurIPS2025_arXiv2505.13934.pdf` | `8fbbaf999af996d52e26954c281b43c8ec8bdc44e0a2f28b69a4c29e8afbc1a4` |

当前主线角色：iVideoGPT/VP2 是跨 tokenizer/生成架构的候选排序控制，RLVR-World 是
post-training 对照框架，NVIDIA Cosmos DV-FSQ 是独立 codec 重建误差审计。
主实验 / 图表 / 贡献主张仅覆盖上述主线平台。