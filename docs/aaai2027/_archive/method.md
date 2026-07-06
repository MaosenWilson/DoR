# 方法与贡献（定稿结构，2026-06-29）

> **论文标题**：*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models*
>
> **方法名**：**DoR — Decode-off Rewards**（去掉解码器重建地板的可验证奖励；旧义 "Dynamics over Reconstruction" 已弃用）。两实例都"把解码地板拿掉"：code 完全不经解码器、pixel_tok 抵消解码地板。
>
> 经 V1–V6 实验收敛后的方法骨架。**2026-06-30 修订**：不再把"code 是更好的 reward"作为单独主张；code / pixel_tok 是地板抵消的两个实例。新的方法口径是 **reward reliability calibration**：先识别 reward floor 如何破坏 GRPO 组内排序，再用地板抵消、秩可靠性软加权、运动残差对齐来校准 reward/advantage。配套策略见 `strategy_20260630.md`。

---

## 〇、管线与奖励计算落点（Fig 1 设计）

**视频世界模型有两处 token，务必别混：**

![image-20260629150512539](/Users/wilson/Library/Application Support/typora-user-images/image-20260629150512539.png)

1. **输入 token 上图蓝色（context，模型之前）**：当前帧 → Visual Encoder → Quantization → 输入 token，喂进世界模型；动作也量化成 action token。
2. **输出 token 上图绿色（预测，模型之后）= sample group**：世界模型**自回归生成视觉 token**，吐出一组 $G$ 个"预测下一帧 token 序列"。→ **sample group 就是模型预测出的 token，不是"输入前的状态"。**

**为什么图里还要过 decoder**：RLVR-World 的奖励是**图像指标**(MSE/LPIPS/SSIM)，要和**真值图像**比，就必须把预测 token 经 **Visual Decoder 解码回图像** → "Decoded predictions"。**decoder 的唯一作用 = 把预测 token 变回像素以便用图像指标算奖励。**

**RLVR-World 在哪算奖励**：decoder **之后**，`Decoded predictions ↔ 原始 GT`（解码后像素）—— **这条路带地板**（decoder 有损）。

**DoR 把奖励计算点前移 / 换目标**：
- **code（Decode-off，核心）**：sample group 本就是 token → 直接取其码向量 $z_i$ 与 $z'=\text{indices\_to\_codes}(\text{encode}(s'))$ 比，**不过 decoder、无地板**。
- **pixel_tok（解码后实例）**：仍解码，但对**可达目标** $\tilde s'=\text{decode}(\text{encode}(s'))$ 算，解码地板两边相消。

> **Fig 1 设计**：在 RLVR-World 框架图上，把 reward 的虚线从 "Decoded predictions ↔ ground-truth"（解码后 / 原始 GT）**挪到 sample group 处**（解码前 / 码空间）= **DoR-code**；并标注 **pixel_tok** 的"对可达目标"变体。一图看出"**我们在更早、无地板的地方算奖励**"。

---

## 一、贡献（先列清楚）

**C1 — 问题：奖励噪声地板（reward noise floor）。**
RLVR 训练 tokenized 视频世界模型时，可验证奖励在**解码后**的像素上计算；有损 tokenizer 的 `decode∘encode` 引入一层**与预测质量无关的不可约误差** $\phi_{\text{tok}}$。我们刻画它**跨度量普遍**：MAE/MSE/SSIM/LPIPS 上都 >0，且在**感知/结构度量（LPIPS、SSIM）上地板 ≥ 帧间动态信号**（实测 LPIPS 地板 0.112 > 信号 0.082）。

**C2 — 机理：地板腐蚀 GRPO 消费的组内排序。**
GRPO 只用组内相对排序。把奖励写成 $R=R^\star+\eta$（$\eta$=地板噪声），地板把组内排序翻转。建模为高斯噪声，**每对候选的排序翻转概率服从闭式 $\arccos(\rho)/\pi$**（orthant 概率），实测与上界 **5 seed 全部吻合到 ~0.002**；弱信号窗口翻转率高达 ~30%，`corr(翻转率, 信号)=−0.79`。

**C3 — 方法：可靠性校准的可验证奖励（reliability-calibrated verifiable reward）。**
可验证奖励应对齐到 tokenizer 的**可达重建**，而非原始 GT，以抵消共享的解码器地板：

- **code（解码前实例，最优）**：在 FSQ 码空间比，不经解码器，地板 $\approx 0$。
- **pixel_tok（解码后实例）**：把目标从原始 GT 换成 $\text{decode}(\text{encode}(s'))$，解码器系统地板两边相消。
这部分是已验证的 floor-cancellation 实例；但它本身还不够支撑整篇论文，因此新增两个轻量、opt-in 的校准模块作为 pilot：

- **rankrel（秩可靠性软加权）**：把 C2 的翻转概率理论落到 advantage 侧，软降权 reward 间隔小、局部排序易翻转的候选，不硬删组。
- **code-dynamics（运动残差对齐）**：在解码前码空间对齐 \(\Delta z=z'-z_t\) 的方向与幅度，给 reward 增加与静态外观更正交的动态信号。

最终能否写成 C3 的子贡献，以多 seed 配对 + 独立评测为准；若 rankrel / dynamics 不赢，作为诚实消融。

**诚实范围（负结果即证据）**：已有多分量融合（dorw / hybrid_tok）与 GRPO 改动（Dr.GRPO、硬地板过滤）**均未稳超单分量**。这说明"多指标堆叠"本身不是贡献；只有能改善组内可靠排序或补充正交动态信号的模块才值得保留。

---

## 二、度量分类（钉死：特征层 ≠ 解码前）

两条正交轴：

| | **像素级** | **特征级（深度网络特征）** |
|---|---|---|
| **解码后**（输入是解码图像，带地板） | MSE、MAE、PSNR、SSIM | **LPIPS(VGG)**、DISTS、FID/FVD/FD-DINOv2 |
| **解码前**（直接在码/token 上，无解码器，无地板） | —— | **code（FSQ 码 RMS）**、phi 连续特征 |

- **LPIPS(VGG)**：用 VGG 卷积**特征**算距离 → 特征级；但输入是**解码后图像** → 解码后、**带地板**。故"特征层"不等于"解码前"。
- 仅 **code/phi** 是真正解码前（零地板）。FID/FD-DINOv2 是**解码后特征分布**度量 → 只进评测(Evaluator)，给不了单候选打分、不能当 reward。

---

## 三、方法形式化

记号：组内 $K$ 个候选，候选 $i$ 的 token $o_i$，解码帧 $\hat s'_i=\text{decode}(o_i)$，码向量 $z_i=\text{indices\_to\_codes}(o_i)$；真值 $s'$、$z'$。

### 3.1 核心概念：奖励噪声地板 $\phi_{\text{tok}}$（定义 / 公式 / 怎么用）

**是什么（一句话）**：tokenizer 有损 → **任何**预测帧解码后都自带一层与预测质量无关的不可约误差；它等于**原始 GT** 与**最佳可达重建**之间的差距。把它叫"地板"，借信号处理"低于噪声地板的信号无法被检测"。

**公式**。设度量 $d$、真值帧 $s'$，逐帧地板与期望地板：
$$
\phi_{\text{tok}}(s') \;=\; d\big(\text{decode}(\text{encode}(s')),\; s'\big),
\qquad
\phi_{\text{tok}} \;=\; \mathbb{E}_{s'}\big[\,\phi_{\text{tok}}(s')\,\big]
$$

**性质**：① 对任意 $d$ 都 $>0$（解码有损的直接后果）；② **由 GT 测得**——故方法仍 GT-anchored、仍"可验证"；③ 是否**淹没动态信号随度量而变**（进展4：LPIPS/SSIM 地板 ≥ 帧间信号，MSE/PSNR 反之）。

**怎么用——两种角色，别混：**

- **(诊断 / C1)** 把 $\phi_{\text{tok}}^{(d)}$ 在各度量上**测出来**（eval-only），判哪些度量被地板主导（→ 哪些 reward 不可信）。**这里地板是一个被估计的"数"。**
- **(方法 / C3) 不**去"估计并**减**"这个数（减常数地板对组内排序**秩不变**，无效，见 §四消融）；而是**换对比目标**——把奖励对齐到**可达重建** $\tilde s'=\text{decode}(\text{encode}(s'))$，让两边共享的解码地板**由构造抵消**：
  - **code**：完全不经解码器（解码前码空间），地板天然为 0；
  - **pixel_tok**：$d(\text{decode}(o_i),\,\tilde s')$，解码地板两边相消。
  完美 token 预测 $\Rightarrow$ 奖励 $=0$（应得满分）；对原始 $s'$ 则永远被扣 $\phi_{\text{tok}}$（不公平，因 $s'$ 本就不可达）。**方法侧不需要估计任何地板数值。**

### 3.2 地板抵消奖励（method）

**核心：把对比目标从原始 $s'$ 换成可达重建 $\tilde s' = \text{decode}(\text{encode}(s'))$。**

- 解码后实例（pixel_tok）：

$$
R^{\text{pix-tok}}_i \;=\; -\,\text{LPIPS}\big(\hat s'_i,\; \underbrace{\text{decode}(\text{encode}(s'))}_{\text{可达目标}}\big)
$$

完美 token 预测 $\Rightarrow \hat s'_i=\tilde s' \Rightarrow R=0$（地板被去掉）；对比"$-\text{LPIPS}(\hat s'_i,s')$"，后者完美预测仍被扣 $\phi_{\text{tok}}$。

- 解码前实例（code，地板天然为 0）：

$$
R^{\text{code}}_i \;=\; -\,\text{RMS}(z_i - z')
$$

### 3.3 进 GRPO（与 RLVR 相同；不是 argmax）

奖励逐候选算出一个标量 $R_i$，组内归一化由 GRPO 完成：

$$
A_i \;=\; \frac{R_i-\text{mean}_k R}{\text{std}_k R+\epsilon}
$$

> 注：实验证明此处**不需要改 GRPO**——标准 `/std` 本就降权地板主导组；去掉它(Dr.GRPO)中性、硬过滤有害。

GRPO **不是**只挑组内最大奖励候选做 SFT；所有候选都参与更新：

$$
\mathcal L_{\mathrm{PG}}=-\frac1K\sum_i A_i\log\pi_\theta(o_i|q).
$$

因此方法设计的核心不是让某个候选分数绝对最高，而是让组内优势符号和排序更可信。

---

## 四、不是方法的部分（写成消融，强化主张）

- **多分量加权融合**（地板感知权重 $w_m=\sigma_\star^2/(\sigma_\star^2+\phi_m^2)$）：权重把高地板分量清零 → 融合塌成 code；即使分量都干净（code+pixel_tok）融合仍 ≤ code。→ "组合无增益"。
- **GRPO 改动**：Dr.GRPO 中性、地板硬过滤有害（删数据 + 破坏 std 的天然降权）。→ "GRPO 不需改"。
- 结论统一：**关键是奖励在哪个空间/对齐哪个目标，不是堆 gadget。**

---

## 五、与 RLVR-World 的一句话差异
RLVR-World：$R=-(\text{MSE}+\text{LPIPS})$，**全解码后、对原始 GT、带地板**。
DoR：**对齐可达重建去地板**（code 解码前 / pixel_tok 解码后），$R$ 更干净 → 组内排序更可信 → 动态预测更好。

---

## 六、度量规格（每个的 输入 / 公式 / 输出）

**公共记号与张量形状**（RT-1 single-step，$H{=}256,W{=}320$，每帧 $16{\times}20{=}320$ 个视觉 token，FSQ 码维 5）：
- 候选 token $o_i$：`cand` $[K,320]$ long；组大小 $K$。
- 解码帧 $\hat s'_i=\text{decode}(o_i)$：$[K,3,256,320]$，值域 $[0,1]$。
- 真值帧 $s'$：$[3,256,320]\in[0,1]$；上一帧 $s$（context 末帧）：$[3,256,320]$。
- 码向量 $z_i=\text{indices\_to\_codes}(o_i)$：$[K,16,20,5]\to$ flatten $[K,1600]$；$z'=\text{indices\_to\_codes}(\text{encode}(s'))$：$[1,1600]$。
- 可达目标 $\tilde s'=\text{decode}(\text{encode}(s'))$：$[3,256,320]$。

> 约定：所有 reward 都"**越大越好**"（取负距离 / 正相似度）；输出均为 $[K]$（逐候选一个标量），喂 GRPO 前不做组内归一化。

---

### 0. tokenizer 基元
| | 输入 | 输出 |
|---|---|---|
| `encode` | 帧 $[3,256,320]$ | token idx $[16,20]$ long |
| `decode` | token $[\cdot,320]$ | 帧 $[\cdot,3,256,320]\in[0,1]$ |
| `indices_to_codes` | idx $[\cdot,16,20]$ | 码 $[\cdot,16,20,5]$ float |

### 1. code（解码前，DoR 核心）
- **输入**：$z_i\,[K,1600]$，$z'\,[1,1600]$（解码前码向量）。
- **公式**：

$$
R^{\text{code}}_i \;=\; -\,\text{RMS}(z_i-z') \;=\; -\sqrt{\tfrac{1}{1600}\textstyle\sum_{d=1}^{1600}(z_{i,d}-z'_d)^2}
$$

- **输出**：$[K]$，单位=码空间 RMS 距离的负值；地板 $\approx 0$。越大越好。

### 2. pixel（解码后，RLVR 基线分量）
- **输入**：$\hat s'_i\,[K,3,256,320]$，$s'\,[3,256,320]$。
- **公式**（LPIPS 用 VGG 骨干，输入先线性映到 $[-1,1]$）：

$$
R^{\text{pixel}}_i \;=\; -\,\text{LPIPS}_{\text{vgg}}(\hat s'_i,\; s')
$$

- **输出**：$[K]$，感知距离负值；**带地板 $\phi_{\text{tok}}$**。

### 3. pixel_tok（解码后 + 地板抵消，C3 解码后实例）
- **输入**：$\hat s'_i\,[K,...]$，**可达目标** $\tilde s'=\text{decode}(\text{encode}(s'))\,[3,256,320]$。
- **公式**：

$$
R^{\text{pix-tok}}_i \;=\; -\,\text{LPIPS}_{\text{vgg}}\big(\hat s'_i,\; \tilde s'\big)
$$

- **输出**：$[K]$；token-最优候选 $\Rightarrow 0$（**地板被抵消**）。

### 4. mse / mae（解码后，像素级）
- **输入**：$\hat s'_i$，$s'$。
- **公式**：

$$
R^{\text{mse}}_i=-\tfrac{1}{3HW}\!\sum(\hat s'_i-s')^2,\qquad R^{\text{mae}}_i=-\tfrac{1}{3HW}\!\sum|\hat s'_i-s'|
$$

- **输出**：$[K]$。

### 5. ssim / ssim_tok（解码后，结构级；piqa，窗 11、$\sigma{=}1.5$）
- **输入**：$\hat s'_i$ 与 $s'$（ssim）或 $\tilde s'$（ssim_tok），均 clip 到 $[0,1]$。
- **公式**：$R^{\text{ssim}}_i=\text{SSIM}(\hat s'_i,s')$；$R^{\text{ssim-tok}}_i=\text{SSIM}(\hat s'_i,\tilde s')$。
- **输出**：$[K]\in[-1,1]$，越大越好（相似度，不取负）。

### 6. phi（解码前连续特征，A5/消融）
- **输入**：$\hat s'_i$ 经冻结 encoder+quant_linear 得**全空间**特征图 $\to$ flatten $[K,C'hw]$；$s'$ 同理 $[1,C'hw]$。
- **公式**：$R^{\phi}_i=-\,\text{RMS}\big(\text{feat}(\hat s'_i)-\text{feat}(s')\big)$。**注意：feat 在解码图上提取 → 仍解码后、带地板**（与 code 区别）。
- **输出**：$[K]$。

### 7. 地板 $\phi_{\text{tok}}$（problem 量化，eval-only）
- **输入**：真值帧集合 $\{s'\}$。
- **公式**（任意度量 $d$）：

$$
\phi_{\text{tok}}^{(d)} \;=\; \mathbb{E}_{s'}\big[\, d(\text{decode}(\text{encode}(s')),\, s') \,\big]
$$

- **输出**：标量/度量（如 LPIPS 下 0.112）。

### 8. 融合（消融，均不优于 code）
- **dorw（地板加权）**：输入各分量距离 $d_m(i)$、离线 $s_m,w_m$；
  $R^{\text{dorw}}_i=-\sum_m w_m\,d_m(i)/s_m$；输出 $[K]$。
- **hybrid_tok（z 融合 code+pixel_tok）**：
  $R^{\text{h-tok}}_i=\alpha\,z(R^{\text{pix-tok}})_i+(1-\alpha)\,z(R^{\text{code}})_i$，$z(\cdot)$=组内标准化，$\alpha{=}0.5$；输出 $[K]$。

### 9. GRPO 优势（奖励之后，组内归一化；与 RLVR 相同）
- **输入**：组内奖励 $R\,[K]$（上面任一）。
- **公式**：

$$
A_i=\frac{R_i-\text{mean}_k R}{\text{std}_k R+\epsilon}
$$

- **输出**：$[K]$ 优势，进策略梯度 $-(A_i\cdot\log\pi_\theta(o_i))$。

---

### 评测指标（Evaluator，对**原始** $s'$，不进 reward）
每窗对 $K$ 候选取均值、再对窗口取均值，复用 RLVR-World `Evaluator`：
- **MAE / MSE**（像素，↓）、**PSNR**（↑）、**SSIM**（piqa，↑）、**LPIPS-vgg**（特征级/解码后，↓）。
- **flow**（动态，↑）：RAFT 光流，$\text{cos}\big(\text{flow}(s\!\to\!\hat s'_i),\,\text{flow}(s\!\to\!s')\big)$ 按 GT 光流幅值加权。
- **dmotion**（兜底，↑）：$\text{cos}(\hat s'_i-s,\; s'-s)$。
> 训练目标(code/pixel_tok 等)与评测分离；FD-DINOv2/CMMD 为可选分布级评测，不进 reward。
