# Upstream Paper Sources

本目录保存外部平台与 RLVR-World 对照所使用的开放获取论文原文。论文版本、来源与本地
文件固定如下；架构比较以这些原文为准，不以二手博客或仓库简介替代。

| 论文 | 会议/版本 | 官方开放来源 | 本地文件 | SHA-256 |
| --- | --- | --- | --- | --- |
| iVideoGPT: Interactive VideoGPTs are Scalable World Models | NeurIPS 2024；arXiv v3 (2024-10-31) | https://arxiv.org/abs/2405.15223 | `iVideoGPT_NeurIPS2024_arXiv2405.15223.pdf` | `e79681befb89de01b4ecbb1abe0f4fa15b60420fe195371f95dcc8b607251984` |
| Transformers are Sample-Efficient World Models (IRIS) | ICLR 2023；arXiv v2 (2023-03-01) | https://arxiv.org/abs/2209.00588 | `IRIS_ICLR2023_arXiv2209.00588.pdf` | `6b583542ba35dc99a36127dd3b7650aac5d28479f3fe0e5e47f741768ae587b4` |
| RLVR-World: Training World Models with Reinforcement Learning | NeurIPS 2025；arXiv v2 (2025-10-25) | https://arxiv.org/abs/2505.13934 | `RLVR-World_NeurIPS2025_arXiv2505.13934.pdf` | `8fbbaf999af996d52e26954c281b43c8ec8bdc44e0a2f28b69a4c29e8afbc1a4` |

实验角色：iVideoGPT/VP2 是同技术谱系的近域控制，IRIS 是独立架构与离散动作域的远域探针，
RLVR-World 是 post-training 对照框架。iVideoGPT 和 IRIS 原文均未使用 GRPO 后训练其世界模型。
