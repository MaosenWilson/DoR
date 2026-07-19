# C1/C2 叙事弧与 FSQ 家族泛化计划

> 2026-07-18。论文的 generality 论证主线 + Cosmos/ChronoDreamer 扩展计划。
> 本文是叙事与计划文档，非结果。标 ✅ 为已验证，🔜 为待做。名字已校准。

## 1. 叙事弧（逻辑呈现顺序，不必等于实验时间顺序）

**层一 —— 基础模型（iVideoGPT 家族）暴露 C1。**
tokenized 视频世界模型的基础是"FSQ tokenizer + 自回归"。iVideoGPT 是这一族的代表。
从它出发就能分析出 **C1（编解码重建误差问题）**：encode→quantize→decode 必然引入重建残差，真值
落在模型不可达的空间之外。这是 **tokenizer 的固有属性，不依赖具体任务**。

**层二 —— RL 后训练（RLVR-World）暴露 C1 的后果 + C2。**
RLVR-World 在 iVideoGPT 式 tokenized 世界模型上做 RLVR 后训练（GRPO + 可验证奖励）。
在这个 RL 场景里，两个问题同时显现：
- **C1 的后果**：编解码重建误差的候选相关残差腐蚀 GRPO 组内排序（GRPO 只消费排序）；
- **C2（长序列 GRPO 问题）**：多步自回归 rollout 的终局质量随 horizon 衰减，而序列级优势
  无法区分不同 future-frame token blocks 的下游责任——我们用 frame-block Temporal Return
  对齐信用单位。GAE 仅作为降低 return 方差的辅助消融，不是 headline 方法。

**层三 —— 泛化到 FSQ/LFQ 家族。**
既然 C1 是 FSQ 量化的固有属性、C2 是长序列自回归的固有问题，那么**其他用类似 tokenizer 的
视频世界模型是否也有同样的长序列生成问题?** 这就引出 Cosmos-Tokenizer、ChronoDreamer 等
FSQ 家族模型的验证。

## 2. 当前已验证的状态（诚实）

| 平台 / 架构 | tokenizer | C1（排序修复） | C2（时序信用） |
|---|---|---|---|
| RLVR-World / RT-1（Llama AR） | CNN-FSQ ✅ + 压缩 FSQ ✅ | ✅ 成立 | ✅ 正式 K=16/low-var-KL 下 Temporal Return 方向性成立；GAE0.7 仅在 K=8 内部评测优于 return |
| iVideoGPT / VP2（GPT 式） | FSQ | ✅ 成立（Δρ +0.09/+0.08） | ❌ null（仿真近天花板，无 headroom） |
| Cosmos DV-FSQ（独立视频 codec） | causal temporal DV-FSQ | ✅ 不可忽略的编解码重建误差（LPIPS 0.190） | — 无匹配候选世界模型，只做重建误差审计 |

**要点**：
- **C1 的编解码重建误差已在三个 FSQ codec 实例上复现**（CNN-FSQ、压缩 FSQ、Cosmos DV-FSQ）；
  排序修复已在 RT-1 single/multi 与 iVideoGPT-VP2 的候选组上成立。两类证据不能混为一个数字，
  但共同排除了“只针对单一 tokenizer 的 target preprocessing”这一解释。
- **C2 只在 RT-1（真实/复杂视频、有 headroom）获得训练支持**；VP2 仿真平台上结果为 null，
  只能报告为适用边界，不能把“无 headroom”写成已经验证的唯一因果解释。C2 定位 = 深度
  （真实视频长序列），不是广度。
- iVideoGPT **保留作 C1 证据，不作 C2 战场**。

## 3. Cosmos / ChronoDreamer 扩展状态

### 3.1 先定位：哪个更"真实/复杂"（决定能否承担 C2）

- **Cosmos**（NVIDIA）：FSQ tokenizer，world foundation model 规模；**C1 重建误差审计已完成**，
  LPIPS-VGG floor 为 0.190（零世界模型、零训练）。
- **ChronoDreamer**（arXiv:2512.18619）：用 Cosmos FSQ 的**动作条件机器人世界模型**；若数据/场景
  偏真实,有 headroom,才可能承担 C2；若也是近天花板仿真,只作 C1。

### 3.2 用户提出的长序列退化测试（决定 C2 候选）—— 采纳

**核心筛选实验（零训练,复用 `headroom_audit.py`）**：让候选模型生成多帧,测
**per-horizon 质量随 horizon 的下降幅度（LPIPS 增长 / SSIM 下降）**。

- 下降越大 = headroom 越足 = 越受长序列问题影响 = **越适合展示 C2 的提升**；
- 近天花板（下降小)= 无空间 = 只作 C1。

用这个把 Cosmos / ChronoDreamer(以及任何新平台)分成"C1-only"和"C1+C2 候选"两类,
**先审计再决定训不训**,不盲目烧训练。

### 3.3 C1 重建误差审计(所有 FSQ 平台通用,零训练)

对任一 FSQ/LFQ tokenizer:取若干真实帧 → encode→decode → 测编解码重建误差(LPIPS/MSE);
有世界模型的再测候选排序腐蚀(raw vs RC 翻转)。不可忽略的重建误差 = C1 普适的直接证据。

## 4. 论文最终 generality 结构

> **C1（重建残差改变 GRPO 组内排序）**:量化 tokenizer 的固有问题,在 FSQ/LFQ 家族多架构普适
> ——RLVR-World(CNN-FSQ/压缩 FSQ)+ iVideoGPT-VP2 的排序审计 + Cosmos DV-FSQ 的独立重建误差审计。
> **C2（长序列 GRPO 时序信用）**:长序列自回归的固有问题,在有 headroom 的真实/复杂视频
> RT-1 上诊断并用 frame-block Temporal Return 修复；GAE 扫描只作为辅助方差消融。

## 5. 待办

- ✅ Cosmos-Tokenizer 权重与 encode/decode 接口已核验;
- ✅ Cosmos C1 重建误差审计完成（LPIPS-VGG 0.190）;
- 🔜 长序列退化测试(headroom_audit)判 Cosmos/ChronoDreamer 是 C1-only 还是 C2 候选;
- ✅ K=8 λ 扫描完成：GAE0.7 相对 return 的 LPIPS/LPIPS-last/MSE 为 3/3 改善，但相对 sequence
  baseline 均值更差，因此不升级为主方法；
- iVideoGPT VP2 保留 C1、不主张 C2(近天花板一句话说明)。
