# 说明：编码后的 token 是不是"特征"？——为什么 DoR 的轴是 pre/post-decode 而非 pixel/feature

> 2026-06-29。面向导师的概念澄清。结论先行，后给推导与对论文的影响。
> 关联：`docs/aaai2027/method.md`（§〇 pipeline / 指标 taxonomy）、`results_v1v6.md`、`exp_v7_mse_tok_20260629.md`。

## 结论（一句话）

**编码后的 token（确切说是 FSQ 码向量）确实是一种"特征"——导师没说错；但这不是论文的轴。**
LPIPS、DINOv2 也是特征级度量，区别不在"像素 vs 特征"，而在 **奖励的计算路径过不过那层有损解码器（pre-decode vs post-decode）**。
奖励噪声地板 $\phi_{\text{tok}}$ 住在解码器里：LPIPS/DINO 是 feature-level 但 **post-decode**，照样吃地板；DoR 的 code 是 feature-level 但 **pre-decode**，所以零地板。**这两条轴正交。**

---

## 0. 什么是"地板"？是不是官方说法？

### 0.1 它到底是什么（精确定义）

$$\phi_{\text{tok}}=\mathbb{E}\big[\,d\big(\mathrm{decode}(\mathrm{encode}(s')),\,s'\big)\,\big].$$

取一帧真值 $s'$，用 tokenizer **编码再解码（往返一圈）**，它回不到原样——这层**不可约残差**在度量 $d$ 下的期望，就是地板。三个性质：

- **与预测质量无关**：连真值自己往返都还原不了，跟模型预测好坏无关。
- **是任何 token 预测能达到的下界**：你最好也只能吐出"真值的 token"，解码后顶多到 $\mathrm{decode}(\mathrm{encode}(s'))$，到不了 $s'$。
- **度量相关**：感知/结构度量（LPIPS/SSIM）地板大，像素 MSE 地板小。

**大白话**：解码器是台**带模糊的复印机**。你只能量复印件来估真值——哪怕复印得"完美"，复印机的模糊就给测量加了去不掉的下限。地板 = 复印机的模糊。
**要命之处**：RLVR 的奖励正是在复印件（decode 输出）上算的；当要学的**动态信号比这层模糊还小**（实测 LPIPS 下地板 $\approx 0.112$ > 帧间动态信号 $\approx 0.082$，见 `results_v1v6.md` 的 C1）时，奖励分不清候选好坏，GRPO 的组内排序被模糊主导。

### 0.2 是不是官方说法？——不是，是我们起的名（关乎学术诚信，必须讲清）

- **"reward noise floor / 奖励噪声地板" + 把它用到 tokenized RLVR 的可验证奖励上 = 本文为此造的术语，没有现成官方出处。** 这恰恰**就是贡献一（C1）**：首次命名并刻画这个现象。若它是现成概念，我们就没有 C1。
- 它**不是生造**，借了三个**已确立、可引**的概念（引作 background，绝不篡改其原意）：
  - **noise floor**（信号处理 / 电子学**标准术语**）：本底噪声水平，低于它的信号无法与噪声区分——借其"低于此不可检测"的语感。
  - **重建上界生成质量**（离散 tokenizer 常识）：VQ-VAE / VQGAN / FSQ 均以重建保真（如 rFID）衡量 tokenizer，生成不可能好过重建。
  - **率失真理论（Shannon）**：有损压缩存在失真下界。
- **写作纪律**：正文写 “we define / we term the *reward noise floor*”，引上述谱系作 background；**绝不**写成 “as defined by [某文献] 的 reward noise floor”——那是编造，踩引用红线。

---

## 1. 先把"token"拆成三个别混的对象

RLVR-World 的视频世界模型里，"token"在不同位置指不同东西。算 `code` 奖励时，经 `indices_to_codes(·)` 落到的是第 ③ 个：

| | 对象 | 是什么 | 是"特征"吗 |
|---|---|---|---|
| ① | 连续编码特征（Visual Encoder 输出，量化前） | CNN 的连续 latent feature map | 是，标准深度特征 |
| ② | 离散 token 索引（世界模型自回归预测的词表符号） | 整数类别标签（vocabulary 1index） | **不是**——类别符号，对它直接做 RMS 无意义 |
| ③ | **FSQ 码向量** $z=\mathrm{indices\_to\_codes}(②)$ | 把索引映回 FSQ 格点的连续向量（①的量化版） | **是**——与①同处 tokenizer 的 latent 空间，只是落在离散格点上 |

DoR 的码空间奖励：
$$R_i^{\text{code}}=-\,\mathrm{RMS}\big(z_i-z'\big),\qquad z=\mathrm{indices\_to\_codes}(o).$$
用的是 ③（量化后的连续 latent），不是 ② 的整数索引。**所以"code 空间是一种（量化的）特征/latent 空间"成立——导师对。** 但下面说明这不构成对论文的威胁，反而帮我们把轴说准。

---

## 2. 关键轴：解码前 vs 解码后（地板住在解码器里）

地板的定义就摆明了它的来源是**解码器**：
$$\phi_{\text{tok}}=\mathbb{E}\big[\,d\big(\mathrm{decode}(\mathrm{encode}(s')),\,s'\big)\,\big].$$
判断一个奖励有没有地板，只看一件事：**它的计算路径过不过 Visual Decoder。** 这条轴与"像素 / 特征"**正交**：

|  | 解码后 post-decode（过解码器，带地板 $\phi_{\text{tok}}>0$） | 解码前 pre-decode（不过解码器，零地板 $\phi_{\text{tok}}\approx 0$） |
|---|---|---|
| **像素层** | $-\mathrm{MSE}$ / $-\mathrm{LPIPS}$（在 decode 出的图像像素上） | —— |
| **特征层** | **LPIPS(VGG)、DINOv2 一致性**（也是特征，但在 decode 出的图像上提特征） | **DoR `code`（FSQ 码向量 RMS）** |

**为什么 post-decode 的特征也逃不掉地板（严格论证）：**
特征提取器 $F$（VGG/DINO）作用在 `decode(token)` 的图像上，看到的是 $F(\mathrm{decode}(\text{token}))$。即便候选 token 完美等于真值 token，它最好也只能达到 $F(\mathrm{decode}(\mathrm{encode}(s')))\neq F(s')$。
按**数据处理不等式 / 流水线顺序**：特征提取在有损解码器的**下游**，只能继承解码器已经丢掉的信息，不可能补回——于是特征空间里也存在地板
$$\phi^F_{\text{tok}}=\mathbb{E}\big[\,d_F\big(\mathrm{decode}(\mathrm{encode}(s')),\,s'\big)\,\big]>0.$$
而 `code` 在解码器**上游**计算，路径里根本没有 decode 这一步，$\phi_{\text{tok}}$ 的来源不产生 → 地板 $\approx 0$。

---

## 3. 给导师的一句话回复

> "码向量确实是特征，但 LPIPS/DINO 也是特征——区别不在特征还是像素，在解码器前还是后。地板在解码器里：LPIPS/DINO 是 feature-level 但 post-decode，照样吃地板；code 是 feature-level 但 pre-decode，所以零地板。我们的轴是 pre/post-decode，不是 pixel/feature。"

---

## 4. 对论文的两个直接后果

1. **Taxonomy 必须画成 §2 的 2×2，并明说贡献骑在 pre/post-decode 轴上**，而不是"我们发明了一种特征奖励"。否则审稿人一句"perceptual / feature loss 早有了（LPIPS、DINO 蒸馏…）"就把新意打没。`method.md` 现有 2×2 正好对上，把本说明的 §1–§2 并入。
2. **图 1 最省的改法**：RLVR 在 `Visual Decoder`**之后**接 Verifiable Reward（绿色输出 token → 解码 → 对 ground-truth）；DoR 从**绿色输出 token 直接**引一条线到 reward，**绕过 Visual Decoder**。一张图说清"同一组预测 token，在解码前 / 解码后两处取奖励"。

---

## 5. 必须记牢的两个追问（审稿人/导师会接着问）

- **"那就直接用 DINOv2 当奖励，它也是特征且更通用"** → DINO 需要图像输入 → 必须先 `decode` → 仍是 post-decode、仍带地板。`code` 是唯一**天然 pre-decode**的，因为世界模型本就直接吐 token；而且**零额外网络**（DINO/VGG 要外挂一个大模型），是"低地板"准则约束下**最省**的解。
- **同源循环嫌疑** → `code` 与训练目标同处一个空间，赢"code 指标"近乎平凡。**主结论只用训练目标之外的独立指标**（LPIPS / PSNR / FVD / 光流一致性），`code RMS` 只进附录。这条同时回应了"它不就是模型自己的特征吗"——正因为同源，才必须用外部独立指标验证。

---

## 6. 导师第二轮追问："特征有特征的评价标准（FID 等），不能简单靠 LPIPS/SSIM 差距说有地板"

### 6.1 必须认的部分

**"靠 LPIPS/SSIM 地板 > 信号 来证明地板存在"——这个论证方式作废。** LPIPS/SSIM 是 full-reference 逐样本保真度量，差距大可能只是它们对像素级偏差敏感的产物，度量依赖、弱。C1 不能骑在这上面。

### 6.2 关键澄清：per-sample **奖励** ≠ 分布级 **评测**（导师把两类混了）

| | full-reference 逐样本（MSE/SSIM/LPIPS/DINOv2-cos） | 分布级集合统计（FID/KID/FVD/FD-DINOv2/CMMD） |
|---|---|---|
| 能当 **reward** 吗 | **能**（单候选 vs 单 GT 有定义） | **不能**——单样本无定义，FID 是对一**批**样本算的 |
| 量什么 | 这个预测像不像**那一帧真值** | 生成的**分布**像不像真实分布 |

**RLVR 的可验证奖励，按定义就是 per-sample full-reference**（每个候选对那一帧 GT 打分）。**FID/FVD 当不了 reward**。所以"**奖励的地板**"只能在逐样本度量里量；地板不是我们选错度量的产物，**它是 RLVR 真实在用的那个奖励（LPIPS）自带的**。导师说的"用 FID"在 reward 这一层用不上。

### 6.3 更深一层：世界模型该用哪个标准？

- 世界模型 = **条件预测"那一帧真实的下一帧"**，不是无条件生成逼真帧。
- 一个 tokenizer 完全可以 **FID≈0（生成逼真）但逐样本地板大（复现不了这一帧）**。对动态预测，**逐样本保真才是对的标准**——所以 reward 用 per-sample 是对的，地板也理应在 per-sample 里量。FID 框架适合"生成器真不真实"，不适合"预测对没对上真实动态"。
- **并且**：就算用**特征**度量，只要它需要图像输入（DINOv2-cosine 这种 per-sample 特征一致性）→ 必须先 decode → **仍是 post-decode、仍带地板**。`code` 唯一天然 pre-decode。→ **决定地板的是解码前/后，不是特征/像素**（回到 §2 正交轴）。

### 6.4 据此修正（这条批评把论文推强）

1. **C1 改骑"度量无关"的论证**：地板的**危害不用 LPIPS 绝对值衡量，用秩腐蚀（C2）**。翻转率 $=\arccos(\rho)/\pi$，$\rho$ 是含噪奖励与干净奖励的**秩相关**——无量纲、跨度量。这是地板存在且有害的硬证据，不依赖"LPIPS gap"。
2. **地板诚实写成度量依赖**：$\phi_{\text{tok}}$ 是 (度量, tokenizer) 对的性质——感知度量大、MSE 小；而 RLVR 偏用感知度量，最脆。不夸大成"普遍大地板"。
3. **评测加分布/特征级独立指标**（正是导师一贯要的 + 06-27 会议方向）：**FVD + FD-DINOv2（+ KID/CMMD）**，验证 code 奖励训出的策略在**特征/分布层面也更好**——"用特征的标准评特征"。这些指标**只做评测、不做 reward**（当不了 per-sample 奖励）。

### 6.5 给导师一句话（升级版，直接覆盖 §3）

> "您说得对，特征不该用 LPIPS/SSIM 当**质量标准**——所以评测我们改用 FVD/FD-DINOv2 这些特征/分布级指标。但要分清：FID 这类是**集合统计、当不了 per-sample 奖励**；RLVR 的奖励按定义就是逐样本 full-reference，地板是**那个奖励**自带的，只能在逐样本度量里量。而且地板的**危害我们不靠 LPIPS 差距证，靠度量无关的秩翻转率** $\arccos(\rho)/\pi$ **证**。特征度量若需图像（DINOv2-cos）仍是解码后、仍带地板——决定地板的是解码前/后，不是特征/像素。"

---

## 7. 待办

- [ ] 把 §1（三对象）+ §2（2×2 正交轴 + 数据处理不等式论证）并入 `method.md` 的指标 taxonomy 与 §〇。
- [ ] **C1 论证重写**：`results_v1v6.md` 里把"LPIPS/SSIM 地板 > 信号"从主证据**降级为度量依赖的旁证**，主证据改用 C2 的秩翻转率 $\arccos(\rho)/\pi$（度量无关）。
- [ ] **评测补特征/分布级指标**：对现有 ckpt 离线跑 **FVD + FD-DINOv2（+ KID/CMMD）**（探针 `probe_phi_dino.py` 可复用；不重训），验证 code 在特征标准下也赢；写清"只评测不作 reward"。
- [ ] 按 §4.2 重画 Fig 1（pre/post-decode 两条引线）。
- [ ] 在 Related Work / §五 vs RLVR 处补一句"feature-level ≠ pre-decode"，并显式区分"per-sample 奖励 vs 分布级评测"两类度量。
