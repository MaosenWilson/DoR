# 方法稿（旧版，已被 docs/aaai2027/method.md 取代）

**论文标题**：*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models.*
（旧标题 *Dynamics over Reconstruction* 已弃用：经 V1–V6，故事重心从"动态优先"改为"奖励噪声地板 + 在哪个空间/对齐哪个目标算"。以 `docs/aaai2027/method.md` 为准。）

> **定位更新（2026-06-15）**：诊断证明像素 LPIPS 可验证奖励被 tokenizer 重建噪声主导（帧间运动 0.040 <
> 重建地板 0.053），RL 实际在优化纹理而非动态。**DoR 主线 = 动态对齐 / 地板校准的可验证奖励**；下文的
> "共识塑形优势（vote）"**降级为二阶去噪组件 / 一个 ablation**（共识信号弱但真实，temp=0.5 下 spearman≈0.08）。
> 本文以下章节记录该共识组件的设计与已核实公式，仍有效，但不再是核心卖点。主线奖励重设计待新一轮实验。

---

## 1. 一句话贡献

**在视频世界模型这种"可验证奖励本身带噪"的场景下，用组内共识对 GT 锚定的细粒度奖励做方差缩减，
得到共识塑形的 GRPO 优势估计——既保 RLVR 严谨性，又解决单一像素度量无法判别"运动/物理合理性"的痛点。**

三篇基线正交组合：

| 来源 | 贡献的部件 | 在本方法中的角色 |
|---|---|---|
| RLVR-World | GT 可验证 reward + GRPO 骨架 | 不动的地基 |
| ToolRL | 细粒度多分量 reward 设计法 | 重新设计 reward（加物理一致性分量） |
| TTRL | 投票/共识 | **改定义**：从"替代 GT 当标签"→"给优势做可靠性加权" |

---

## 2. 记号与 RLVR-World 骨架（已核实）

输入 $q=(s,a)$（context 帧 + 动作）。策略 $\pi_\theta$ 采样一组 $G$ 个候选
$\{o_i\}_{i=1}^G$，解码出预测下一状态 $\{\hat s'_i\}$。GT 下一状态 $s'$。

RLVR-World 原始 reward（论文 Eq.3）：
$$R_i=\mathrm{sign}(D)\cdot D(\hat s'_i,\,s'),\qquad \mathrm{sign}(D)=-1\ \text{若度量越小越好（MSE/感知损失）}$$

GRPO 优势（论文 Eq.1，标准组内归一化）：
$$A_i=\frac{R_i-\mathrm{mean}(\{R_j\})}{\mathrm{std}(\{R_j\})+\epsilon}$$

目标 $J_{GRPO}$ 为带 clip + KL 惩罚的策略梯度。**本方法只改 $R_i$ 的内部结构和 $A_i$ 的塑形，不动 $J_{GRPO}$ 的形式。**

---

## 3. 细粒度 GT 锚定 reward（ToolRL 思路）

**问题**：RLVR-World 视频版用单一度量 $D$（如 LPIPS）。单标量无法区分"静止退化帧"与"运动但合理帧"——
这正是用户的核心痛点。ToolRL 教训：**别用单一/二值 reward，拆成有界、对称归一化、可解释的多分量。**

设计 GT 锚定多分量 reward（全部对 GT 计算，保持可验证）：
$$R_i=\sum_{k}\alpha_k\,\tilde R^{(k)}_i,\qquad \tilde R^{(k)}_i\in[-1,1]\ \text{（对称归一化，仿 ToolRL 的 }[-R_{max},R_{max}]\text{）}$$

候选分量 $\tilde R^{(k)}$（均 vs GT，可验证）：
- **像素保真**：PSNR / −MSE
- **感知**：−LPIPS
- **结构**：SSIM
- **运动/物理一致性**：$\mathrm{sim}\big(\mathrm{flow}(\hat s'_i),\,\mathrm{flow}(s')\big)$ —— 光流/运动幅度对齐，**显式编码"是否运动、运动是否对"**

> 这一步把"物理合理性判别"从"祈祷单一度量捕捉"变成"GT 锚定的显式分量"，是回应用户最初质疑的关键，且仍 100% verifiable。
> 必做消融：reward scale、各 $\alpha_k$、有无运动分量（呼应 ToolRL 的 reward-design 消融）。

---

## 4. 共识塑形的优势估计（核心创新）

**动机（顶刊级论点）**：在 math/code，verifiable reward 是**精确**的（对/错）；
但视频度量是"物理合理性"的**带噪代理**：$R_i=R_i^\star+\eta_i$。
若组内共识与低噪相关，则按共识加权 = 一个**去噪估计器 / 控制变量**。

**共识信号**（leave-one-out 互一致性）：
$$c_i=\frac{1}{G-1}\sum_{j\ne i}\mathrm{sim}(o_i,o_j),\qquad \hat c_i=\frac{c_i-\mathrm{mean}(c)}{\mathrm{std}(c)+\epsilon}$$

**共识塑形优势**（reward 保持纯 GT，只调梯度权重）：
$$\boxed{\ \tilde A_i=A_i\cdot w(\hat c_i),\qquad w(\hat c_i)=1+\beta\,\hat c_i\ }$$

### 两条不可违反的设计不变量

1. **优雅退化到 RLVR-World**（Never break userspace）：候选坍缩/共识无信息时 $\hat c_i\to0\Rightarrow w\to1$，
   自动退回原版 GRPO。$w$ 组内均值≈1，禁止偷偷缩放有效学习率。
2. **GT 必须压住"集体幻觉"**：全体一致认同错误物理（高 $c_i$、低 GT）时，优势仍须为负。
   这就是为什么 $w$ 乘在**已含 GT 符号的 $A_i$** 上，而非加进 reward（否则高共识把 GT 判错的模式洗成正激励）。
   **专设一个 ablation 证明这条。**

---

## 5. "修改定义"的精确落点：TTRL vs Vote2World

| 维度 | TTRL（原始） | Vote2World（改定义） |
|---|---|---|
| 投票产物 | 共识标签 $y$ | 共识度 $c_i$（连续，不取 argmax） |
| 投票角色 | **替代 GT** 当伪标签 | **不替代 GT**，给优势做可靠性加权 |
| reward | $R(\hat y_i,y)=\mathbb{1}[\hat y_i=y]$（无 GT） | $R_i=\mathrm{sign}(D)D(\hat s'_i,s')$（GT 锚定，细粒度） |
| 共识失败后果 | 伪标签错→训练崩 | 退回纯 GRPO（不变量 1 兜底） |
| 适用前提 | 无 GT 可用 | 有 GT 但 GT 度量带噪 |

> 一句话定位：**TTRL 用投票"造标签"，我们用投票"判可信度"。** 同一个共识量，角色从 label-estimator 改成 advantage-denoiser。

---

## 6. 实验骨架（顶刊门槛）

**数据**：训练 RT-1 (fractal20220817)；目标域 RECON（vid2world）。
**基线**：RLVR-World（原版 GRPO）、增大 $G$ 的 RLVR-World（控算力）、(A) reward-shaping 变体、纯 SFT/MLE。
**核心消融**（每个隔离一个部件）：
1. GRPO vs 共识塑形 GRPO（同 $G$、同 reward）→ 证明共识有增益。
2. vs 增大 $G$ → 证明增益不只是"多采样"。
3. (B) 优势调制 vs (A) reward 加项 → 证明 (B) 不漂移 GT 最优。
4. 集体幻觉 ablation → 证明不变量 2。
5. reward 分量/scale 消融（ToolRL 式）→ 运动分量的必要性。
6. $\beta$ 敏感性、共识度量空间（像素 vs 特征）。

**指标**：PSNR/SSIM/LPIPS/FVD（vs GT）+ 物理一致性（光流误差）+ 下游策略成功率（若接 policy）。
**可复现**：固定随机种子、记录 $G,\beta,\alpha_k$、报告多 seed 均值±std。

---

## 7. 待你拍板的开放决策

1. **reward 分量集合与权重 $\alpha_k$**：先全开等权 + 消融，还是先 LPIPS+运动两分量起步？
2. **共识度量 $\mathrm{sim}$ 的空间**：像素空间（和 reward 同空间，简单）还是 VAE/感知特征空间（更鲁棒）？
3. **$w$ 的形式**：线性 $1+\beta\hat c$ 起步，还是直接上有下界的 $\mathrm{softplus}$/截断防负权重？
4. **是否接下游 policy 评估**：顶刊很吃"world model 改进→策略成功率提升"的闭环，但工作量大。

---

## 已核实引用（均来自本地 PDF 实读，未编造）

- **RLVR-World: Training World Models with Reinforcement Learning** — reward Eq.3 = $\mathrm{sign}(D)\cdot D(\hat s',s')$；GRPO 优势 Eq.1。〔作者/arXiv 号待从 PDF 首页补全〕
- **TTRL: Test-Time Reinforcement Learning** — NeurIPS 2025（论文页脚确认"39th NeurIPS 2025"）；majority-voting reward Eq.3 = $\mathbb{1}[\hat y_i=y]$。
- **ToolRL: Reward is All Tool Learning Needs** — NeurIPS 2025；reward = $R_{format}+R_{correct}$，细粒度三分量 + 对称归一化 $[-R_{max},R_{max}]$，$R_{max}=3$。
- **Vid2World: Crafting Video [...]** — 目标域 RECON 来源〔细节待读〕。
- **GRPO** — Shao et al., 2024, *DeepSeekMath*, arXiv:2402.03300（RLVR-World 引为 [56]，高置信）。

> 注：上述 4 篇本地 PDF 的精确作者列表/arXiv 号/页码，下一步用 `nature-academic-search` 或读 PDF 首页补全，再写进正式 bib。
