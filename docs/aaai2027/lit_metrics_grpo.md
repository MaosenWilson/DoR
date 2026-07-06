# 文献核查：特征层评测指标 + GRPO 改进（导师方向调整，2026-06-26）

> 用途：响应导师三点改进意见的**文献落盘**。**所有条目均经联网检索核实**（标注 ✅核实），arXiv 号与标题来自检索结果，非凭记忆杜撰。未完全核实处标 ⚠️。
> 导师三点：① 保留 decoder 后图像指标 + 增加 encoder 后特征层指标，加权，重写 Evaluator；② 用改进版 GRPO 作第二创新点，小改；③ reward 计算对比原始 RLVR 做设计，与①相辅相成。

---

## 一、特征层 / 生成质量指标（导师说"17 年起有专门指标"——确认核心就是 FID 一族）

### A. 分布级（Inception/I3D 特征，比"一批 vs 一批"）—— 属于**评测(Evaluator)**，不是逐候选 reward

| 指标 | 出处（✅核实） | 一句话 |
|---|---|---|
| **FID** Fréchet Inception Distance | Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter, *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, **NeurIPS 2017**, [arXiv:1706.08500](https://arxiv.org/abs/1706.08500) | 把真/生成两批图像送进 InceptionV3，比两组**特征分布**（高斯近似的均值+协方差）的 Fréchet 距离。**这就是导师说的"17 年特征层指标"。** |
| **KID** Kernel Inception Distance | Bińkowski, Sutherland, Arbel, Gretton, *Demystifying MMD GANs*, **ICLR 2018**, [arXiv:1801.01401](https://arxiv.org/abs/1801.01401) | FID 的 MMD（多项式核）版，**无偏、小样本更稳**。 |
| **FVD** Fréchet Video Distance | Unterthiner, van Steenkiste, Kurach, Marinier, Michalski, Gelly, *Towards Accurate Generative Models of Video: A New Metric & Challenges*, **2018**, [arXiv:1812.01717](https://arxiv.org/abs/1812.01717) | FID 的视频版，把 Inception 换成 I3D，兼顾时序真实度。 |

**关键技术提醒（诚实，必须告诉导师）**：FID/KID/FVD 都是**分布级**——需要一**批**样本才能估均值/协方差，**无法给单个候选打分**。所以它们只能进 **Evaluator（评测）**，**不能直接当 GRPO 的逐候选 reward**。这正是为什么 RLVR-World 的 reward 用的是逐对的 LPIPS 而非 FID。

### A+. FID **之后**（2019–2024）的特征层分布指标（FID 有已知缺陷，这些是改进/替代）

FID 的公认缺陷：① 依赖 InceptionV3 特征（对现代生成内容表征差）；② 假设特征服从高斯；③ 样本复杂度高（小样本不稳）。后续工作分两条线：

| 指标 | 出处 | 改了什么 / 对我们的价值 |
|---|---|---|
| **Improved Precision & Recall** | Kynkäänniemi, Karras, Laine, Lehtinen, Aila, **NeurIPS 2019**, [arXiv:1904.06991](https://arxiv.org/abs/1904.06991) ⚠️号待核 | 把"质量"和"覆盖度"**拆成两个数**（precision=保真, recall=多样），都在特征空间近邻里算。 |
| **Density & Coverage** | Naeem, Oh, Choi, Uh, Yoo, *Reliable Fidelity and Diversity Metrics for Generative Models*, **ICML 2020**, [arXiv:2002.09797](https://arxiv.org/abs/2002.09797) ✅ | 修正上面对离群点不稳的问题，更可靠的保真/多样分解。 |
| **FD-DINOv2** ★对我们最顺手 | Stein 等, *Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models*, **NeurIPS 2023**, [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/0bc795afae289ed465a65a3b4b1f4eb7-Paper-Conference.pdf) ✅(arXiv 号待核) | 大规模人评发现 **FID(Inception) 与人类判断不符**；改用 **DINOv2-ViT-L/14 特征**算 Fréchet 距离(FD-DINOv2)显著更好。**我们 B0 早就用 DINOv2 当参照——FD-DINOv2 是天然衔接、且更现代的评测指标。** |
| **CMMD** (CLIP-MMD) | Jayasumana 等, *Rethinking FID: Towards a Better Evaluation Metric for Image Generation*, **CVPR 2024**, [arXiv:2401.09603](https://arxiv.org/abs/2401.09603) ✅ | 用 **CLIP 特征 + MMD(RBF 核)** 替 FID：**无偏、不假设高斯、样本高效**。现代 FID 替代品。 |

**给本项目的取舍**：Evaluator 想加分布级特征指标，**优先 FD-DINOv2（接我们已有的 DINOv2）或 CMMD（更稳）**，而不是默认的老 FID/Inception——既更现代、又能在论文里讲"我们用了对生成更可靠的评测"。Precision/Recall、Density/Coverage 适合做**保真 vs 多样**的拆解分析（附录）。但同提醒：**这些仍是分布级，只进 Evaluator，不是逐候选 reward。**

### B. 逐对（per-pair）的深度特征指标 —— 可作 reward 的特征层度量

| 指标 | 出处（✅核实） | 一句话 |
|---|---|---|
| **LPIPS** | Zhang, Isola, Efros, Shechtman, Wang, *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, **CVPR 2018**, [arXiv:1801.03924](https://arxiv.org/abs/1801.03924) | 在预训练 DNN 多层特征上算欧氏距离。**注意：它仍是在"解码后图像"上抽特征**，本质 post-decoder。 |
| **DISTS** | Ding, Ma, Wang, Simoncelli, *Image Quality Assessment: Unifying Structure and Texture Similarity*, **TPAMI 2020**, [arXiv:2004.07728](https://arxiv.org/abs/2004.07728) | 在 VGG 特征图上同时算**结构相似(特征图相关)** + **纹理相似(空间均值相关)**，比 LPIPS 更稳健。**可作"特征层结构+纹理"reward 的范本。** |
| （背景/前身）Perceptual/Feature loss | Johnson, Alahi, Fei-Fei, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, **ECCV 2016**, [arXiv:1603.08155](https://arxiv.org/abs/1603.08155) ⚠️编号凭记忆，待核 | 用 VGG 特征差作训练损失的鼻祖（2016，早于导师说的 17 年）。 |

### C. 对本项目的落地解读（区分 Evaluator 与 reward）
- **Evaluator（点①的"重写"）**：在现有 MAE/MSE/PSNR/SSIM/LPIPS-vgg 上**增加 FID / FVD**（分布级、特征层、17 年起的"专门指标"），这是干净的对标补强。
- **reward（点③）**：reward 必须逐候选，**FID 不能用**；可用的"特征层 reward"是——(i) 我们已有的 FSQ 码空间 RMS（tokenizer 自身特征，解码前）；(ii) DISTS 式的"特征图结构+纹理"距离；(iii) cosine 等。**导师"加权融合 decoder 前/后指标"= 把 post-decoder 的 LPIPS/SSIM 与 pre-decoder 的码空间/DISTS 距离按权重组合。**
- **"不同粒度"**：特征图有空间结构 → 可在**全局池化 / 逐空间格 / 多层**等粒度上算（我们之前 phi 的 GAP vs 全空间 bug 正是这个轴的一个点）。

---

## 二、GRPO 改进（导师点②：用改进版作第二创新点）

| 方法 | 出处（✅核实） | 改了什么 / 与我们的关系 |
|---|---|---|
| **GRPO（原始）** | Shao 等, *DeepSeekMath*, **2024**, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | 组内采样 G 个、减组均值除组标准差得优势，去掉 critic。RLVR-World 用的就是它。 |
| **Dr. GRPO** ★最契合 | Liu 等, *Understanding R1-Zero-Like Training: A Critical Perspective*, **2025**, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) | **指出 GRPO 的归一化有偏**（长度偏置 + 除以组内 std 的偏置），提出**去偏**版。**与我们 thesis 直接共振**——我们说"组内排序被地板腐蚀"，它说"组内归一化本身有偏"，两者可叠加。 |
| **DAPO** | ByteDance 等, *DAPO: An Open-Source LLM RL System at Scale*, **2025**, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) | Clip-Higher、**动态采样(过滤组内奖励全同的退化组)**、token 级损失、overlong 整形。**"动态采样"与我们最相关**：可改成"过滤组内信号低于地板的组"。 |
| （备选）SEED-GRPO | *Semantic Entropy Enhanced GRPO*, 2025, [arXiv:2505.12346](https://arxiv.org/abs/2505.12346) ⚠️ | 按不确定性加权优势。 |
| （备选）Segment Policy Optimization | 2025, [arXiv:2505.23564](https://arxiv.org/abs/2505.23564) ⚠️ | 段级信用分配。 |

**第二创新点的最优落点（建议）**：以 **Dr. GRPO（去偏）或 DAPO（动态采样）** 为基座，加一个**针对"奖励噪声地板"的小改**——
- 思路：DAPO 按"组内奖励是否全同"过滤组；**我们改成按"组内信号方差 σ²⋆ 是否低于地板 σ²η"过滤/降权**（即只在信号高于地板的组上更新，或按 σ²⋆/σ²η 给组加权）。这与点①的地板量化（进展4 的地板/信号比）**直接相辅相成**，且是"在他们改进基础上的无伤大雅小改"。

---

## 三、三点意见 → 落地设计（与现有工作的衔接）

1. **重写 Evaluator**：在五项帧指标上加 **FID + FVD**（点①的"特征层专门指标"，分布级、做评测）。reward 侧用 pre/post-decoder 逐对指标加权。
2. **第二创新点 = 改进 GRPO**：Dr.GRPO/DAPO 为基 + **地板感知的组过滤/加权**小改（点②）。
3. **reward 设计对比 RLVR**：RLVR 用单一 post-decoder −LPIPS；我们用 **"组内 × 解码前(码空间/DISTS式) + 解码后(LPIPS/SSIM) × 多粒度"的加权可验证奖励**（点③），并用进展4 的"地板/信号比"决定权重（地板高的度量降权）。

### 待核实/待办
- ⚠️ Johnson 2016、SEED-GRPO、SPO 的 arXiv 号需二次核验后再进正式 bib。
- 设计层面要确认：FID/FVD 只进 Evaluator；reward 的"特征层"具体用 DISTS 式还是码空间 RMS（或两者加权）需做小实验定权重。

> 来源（本页所有 ✅ 条目）：见各行 arXiv 链接。检索于 2026-06-26。

---

## 四、RLVR-World 奖励的**精确公式**（代码核实，纠正早前"R=sign(D)·D"与"单一 −LPIPS"的说法）

来源：`verl/verl/trainer/ppo/ray_trainer.py` 的 `RayVGPTPPOTrainer`（行 ~1215–1265），代码确认：

```python
# recon 项（trainer.reward_fn 选 mse 或 mae）
recon_loss = mean((real - pred)**2)        # 或 mean(|real - pred|)
perceptual_loss = LPIPS_vgg(real, pred)    # tokenizer_wg.perceptual_loss
loss = recon_loss + perceptual_loss        # 等权相加（有 trainer.loss_weight 旋钮）
reward_tensor[i, last] = -loss[i]          # 奖励 = -loss，放在最后一个 response token
```

$$\boxed{\,R^{\text{RLVR}}_i \;=\; -\Big(\,\underbrace{d_{\text{recon}}(\hat s'_i,\,s')}_{\text{MSE 或 MAE}}\;+\;\underbrace{\text{LPIPS}_{\text{vgg}}(\hat s'_i,\,s')}_{\text{感知}}\,\Big),\qquad \hat s'_i=\text{decode}(o_i)\,}$$

要点：**两项**（重建 + 感知）、**全在解码后**、**等权**、**全图粒度**、**无特征层**。`LOSS_KEYS=[lpips,mse,mae,ssim,psnr]` 只用于**日志**（`critic/{k}/mean`），不是各自一个 reward。
→ 影响：我们主表 A0=−LPIPS 是**简化**；忠实基线应为 **A0′ = −(MSE+LPIPS)**（待补）。

## 五、我们的奖励设计（DoR — 地板感知的多空间加权可验证奖励）vs RLVR

**记号**：组内 K 个候选，候选 $i$ 的解码帧 $\hat s'_i=\text{decode}(o_i)$，码空间向量 $z_i=\text{indices\_to\_codes}(o_i)$，真值 $s'$、$z'$。分量集合 $m$（每个都是"越大越好"的负距离）：

| 分量 $m$ | 公式 | 空间 | 地板 $\phi_m$ |
|---|---|---|---|
| code（本文核心） | $r^{\text{code}}_i=-\text{RMS}(z_i-z')$ | 解码前 | ≈ 0 |
| recon | $r^{\text{recon}}_i=-\text{MSE}(\hat s'_i,s')$ | 解码后 | 低（进展4：地板<信号）|
| perc | $r^{\text{perc}}_i=-\text{LPIPS}_{\text{vgg}}(\hat s'_i,s')$ | 解码后 | 高 |
| (可选) feat | $r^{\text{feat}}_i=-\,d_{\text{DINOv2}}(\hat s'_i,s')$ 逐对 | 解码后 | 中 |
| (可选) 多粒度 | code 的全局池化 + 逐空间格两档 | 解码前 | ≈ 0 |

**Step 1 组内 z-score**（消去各分量量纲差异，源自 ToolRL 多分量做法）：
$$\tilde r^{(m)}_i=\frac{r^{(m)}_i-\text{mean}_k\,r^{(m)}}{\text{std}_k\,r^{(m)}+\epsilon}$$

**Step 2 地板感知加权求和**（核心创新，把进展4 的"地板/信号比"落成权重；形如逆方差/维纳加权）：
$$\boxed{\,R^{\text{DoR}}_i=\sum_m w_m\,\tilde r^{(m)}_i,\qquad w_m=\frac{\sigma^{(m)2}_\star}{\sigma^{(m)2}_\star+\phi_m^2}\,}$$
其中 $\sigma^{(m)2}_\star$=该分量的组内信号方差、$\phi_m$=该度量的奖励噪声地板（由进展4 离线测）。**code 项 $\phi\approx0\Rightarrow w\approx1$（最高权）；LPIPS 等高地板项被自动降权。** 这正是"地板淹没信号则不可信"的可操作化。

**Step 3 进 GRPO**：$R^{\text{DoR}}_i$ 喂入**改进版 GRPO**（Dr.GRPO 去偏 / DAPO 动态采样）+ **地板感知组过滤**（丢弃组内 $\sigma_\star^2\!\lesssim\!\sigma_\eta^2$ 的退化组）—— 第二创新点。

### 对照表（四维升级）

| 维度 | RLVR-World | DoR（本文设计） |
|---|---|---|
| 分量 | 2 项：recon(MSE/MAE)+LPIPS | 多项：+ **code(解码前)**、可选 DISTS/DINOv2、多粒度 |
| 空间 | 全**解码后**（带地板） | 以**解码前码空间**（≈零地板）为主 + 解码后辅 |
| 权重 | **等权 1:1** | **地板感知 $w_m=\sigma_\star^2/(\sigma_\star^2+\phi_m^2)$** |
| 粒度 | 全图 | 全局/逐格/多层 |
| GRPO | 原始 | Dr.GRPO/DAPO + 地板感知组过滤 |

**与三点意见的对应**：分量/权重 = 点①③；解码前码空间 = 本文 DoR 核心；GRPO 改进 = 点②。**退化保证**：只留 perc+recon、等权、解码后 → 精确退回 RLVR-World（可作消融的连续过渡）。
