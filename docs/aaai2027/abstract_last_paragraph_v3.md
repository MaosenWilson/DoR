# 摘要末段修订（v3）

对照：`引言2.0_核对修订版.docx` 摘要。  
用户意见：最后一句过长；贡献呈现需更清晰。  
数字：E1 episode-disjoint，return_rc vs seq_rc（与主表一致）。

---

## 问题诊断（原末段）

原结尾把 **C1 机制证据** 与 **C2 训练数字** 揉进同一超长句，导致：

1. 审稿人抓不住「两条贡献各靠什么数」；  
2. 重建残差/跨架构排序（广）与 RT-1 长序列增益（深）层级不清；  
3. 单句堆相对百分比 + seed 计数，节奏差。

---

## 推荐替换（中文，直接贴摘要后半）

> ……原始真实帧仍用于最终评测，因而校准不会改写任务目标。  
> **实验上，我们把证据拆成两层。**  
> **（C1）** 在 CNN-FSQ、压缩 FSQ 与 Cosmos DV-FSQ 上均观测到不可忽略的编解码重建误差；在 RT-1 与 VP² 的同候选审计中，可达目标重评分一致提高排序相关性并降低成对翻转，表明重建残差会腐蚀组相对学习信号，且该问题不局限于单一 tokenizer。  
> **（C2）** 在固定的 RT-1 episode-disjoint 协议下，帧块时间回报相对序列级重建校准基线，将完整序列 LPIPS、末帧 LPIPS 与 MSE 平均降低 1.39%、1.93% 与 1.91%（MSE 为 5/5 种子改善，其余主要保真指标为 4/5）。  
> 长序列改进集中在有预测 headroom 的设置；近天花板仿真平台用于界定适用范围，而非夸大生成增益。

---

## 更短版（若摘要字数紧）

> ……原始真实帧仍用于最终评测。  
> 跨 FSQ 编解码器的不可忽略的编解码重建误差，以及 RT-1/VP² 同候选审计中的 ρ 提升与 flip 下降，支持可达性约束的排序校准（C1）。  
> 在 RT-1 episode-disjoint 协议下，帧块时间回报相对序列级 RC 基线使 LPIPS/末帧 LPIPS/MSE 平均降低 1.39%/1.93%/1.91%（MSE 5/5，其余主指标 4/5）（C2）。  
> 生成质量改进报告于有 headroom 的长序列设置；近天花板平台仅作边界。

---

## 英文对应（可选）

> We separate two evidence layers.  
> **(C1)** Non-negligible encode–decode reconstruction residuals appear on CNN-FSQ, compressive FSQ, and Cosmos DV-FSQ; same-candidate audits on RT-1 and VP² raise rank correlation and cut pairwise flips, showing that codec residuals can corrupt group-relative preferences beyond a single tokenizer.  
> **(C2)** Under a fixed episode-disjoint RT-1 protocol, frame-block temporal return reduces full-rollout LPIPS, last-frame LPIPS, and MSE by 1.39%, 1.93%, and 1.91% versus sequence-level RC (MSE 5/5 seeds; other primary fidelity metrics 4/5).  
> We claim generation gains where headroom exists; near-ceiling simulators define scope rather than inflated qualitative wins.

---

## 与旧稿数字对齐检查

| 陈述 | 值 | 源 |
|---|---|---|
| LPIPS −1.39% | (0.19657−0.19933)/0.19933 | `msp_episode_disjoint_eval_summary.json` |
| LPIPS-last −1.93% | (0.20767−0.21176)/0.21176 | 同上 |
| MSE −1.91% | (0.013556−0.013820)/0.013820 | 同上 |
| MSE 5/5；其余 4/5 | wins 字段 | 同上 |
| 三 FSQ 编解码重建误差 | 0.052 / ~0.077 / 0.190 | rank_preservation + 叙事/Cosmos 审计（camera-ready 前冻结 JSON） |

**不要**在摘要写：RA-RC 普适双赢、超加和、iVideoGPT 像素改进、击败完整 RLVR recipe。

---

## 完整摘要拼接示例（推荐版全文）

条件视频世界模型通常在离散视觉表征中自回归生成候选未来，再将候选解码到像素空间，并依据其与真实未来的距离进行组相对后训练。然而，候选图像只能落在冻结视觉编解码器能够重建的输出集合内，而原始真实帧通常位于该集合之外。由此产生的重建残差不仅抬高评价下界，还可能改变同一候选组的相对排序；在长序列中，传统做法又将单个序列级优势广播给所有未来视觉 token，无法区分不同时间位置对后续误差的责任。为此，我们提出 RC-GRPO：首先以冻结编解码器对真实帧的可达重建作为训练期比较参照，并以同组候选的秩相关与成对翻转率审计校准是否必要；随后以完整帧 token 块为责任单元，为每个时间位置分配由当前帧至序列末端的时间回报。原始真实帧仍用于最终评测，因而校准不会改写任务目标。实验上，我们把证据拆成两层。（C1）在 CNN-FSQ、压缩 FSQ 与 Cosmos DV-FSQ 上均观测到不可忽略的编解码重建误差；在 RT-1 与 VP² 的同候选审计中，可达目标重评分一致提高排序相关性并降低成对翻转，表明重建残差会腐蚀组相对学习信号，且该问题不局限于单一 tokenizer。（C2）在固定的 RT-1 episode-disjoint 协议下，帧块时间回报相对序列级重建校准基线，将完整序列 LPIPS、末帧 LPIPS 与 MSE 平均降低 1.39%、1.93% 与 1.91%（MSE 为 5/5 种子改善，其余主要保真指标为 4/5）。长序列改进集中在有预测 headroom 的设置；近天花板仿真平台用于界定适用范围，而非夸大生成增益。
