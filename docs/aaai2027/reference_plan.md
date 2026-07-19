# 参考文献与真实性核验计划

正式 bibliography 为 `docs/AuthorKit27/aaai2027.bib`。每条引用必须由论文原文、官方 proceedings
或官方项目页核验；不以参考文献数量为目标，不引用与当前 C1/C2 无关的历史实验分支。

## 1. Required Citation Families

| manuscript responsibility | primary sources to retain |
|---|---|
| tokenized video generation/world models | VQ-VAE、VideoGPT、iVideoGPT、FSQ |
| action-conditioned robotics data/model | RT-1/Open X-Embodiment、RLVR-World、VP2/RoboNet upstream |
| group-relative policy optimization | GRPO original source、RLVR-World implementation source |
| full-reference fidelity | MSE convention、SSIM、LPIPS |
| neural codec limits | rate-distortion/perception-distortion primary sources |
| gradient projection context | PCGrad、CAGrad；只说明通用几何来源 |
| delayed credit | Monte Carlo/n-step return primary RL sources；已核验的 visual step-credit work |
| boundary metrics | KID/FVD/DINOv2 only if retained in final experiments |

## 2. Citation Responsibilities

- `codec-reachable set` 是本文采用的操作性术语，不能伪称某篇先前论文已使用完全相同英文；引用
  应支持 tokenizer output constraint、quantization 和 reconstruction trade-off。
- $\arccos(\rho)/\pi$ 必须引用或明确为经典 bivariate-Gaussian sign relation，不能写成本文定理。
- Raw-anchored projection 必须引用 gradient-projection/multi-objective context，同时明确本文的新对象
  是 raw/RC verifier pair，而非欧氏半空间投影本身。
- Temporal Return 必须引用经典 return/credit assignment；新颖性表述限定在 autoregressive video
  frame-block interface 和验证协议。
- GitHub 只用于 implementation attribution，不支撑论文性能数字。

## 3. Verification Checklist

对每个实际进入 TeX 的 key 核对：标题、作者顺序、venue/arXiv、年份、DOI/URL、正文 claim 是否由
该来源直接支持。无法确认会议录用状态时使用经过核验的 arXiv `@misc`，不猜测 venue。

以下历史类别如不再出现在正文，应从 bibliography 删除：大量通用 GRPO variants、未进入最终
方法的 reward metrics，以及只服务于失败候选搜索的引用。

## 4. Final Audit

1. `rg '\\cite' AnonymousSubmission2027.tex` 导出所有 citation keys；
2. 与 `aaai2027.bib` 双向检查，删除未引用条目；
3. 对保留条目逐项做真实性核验；
4. 编译确认无 undefined citations；
5. 检查每个 novelty claim 都由“现有工作做到什么/尚缺什么”而非泛泛引用支撑。
