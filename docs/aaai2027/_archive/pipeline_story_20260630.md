# 当前 Pipeline 与论文故事线（2026-06-30）

> 论文：*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models*  
> 当前定位：不是证明“code 是更好的 reward”，而是证明 **tokenized video RLVR 里的 verifiable reward 需要做可靠性校准**。  
> 代码包名仍为 `dor`；RLVR-World 主架构不动，只改 reward 设计与轻量 GRPO 可靠性机制。

---

## 0. 一句话版本

RLVR-World 把 tokenized 视频世界模型的预测 token 解码成图像，再用 MSE/LPIPS 这类逐样本 full-reference 指标给每个候选打 reward，最后用 GRPO 根据组内优势更新策略。问题是：**解码器本身有不可约重建误差**，这层误差会污染 post-decode reward 的组内排序，而 GRPO 正是吃这个排序。我们的工作不是大改 RLVR，而是提出一套 **reward reliability calibration**：识别 reward floor、解释它如何翻转排序，并设计 floor-cancelled / motion-aware / rank-aware 的 reward 或 advantage 校准机制。

---

## 1. 原始 RLVR-World Pipeline

给定一个转移：

$$
q=(s,a),
$$

其中 \(s\) 是 context frames，\(a\) 是动作。视频世界模型作为策略：

$$
\pi_\theta(o|q)
$$

自回归采样一组 \(G\) 个候选下一帧 token 序列：

$$
o_1,\ldots,o_G.
$$

这些 token 经 tokenizer decoder 解码成图像：

$$
\hat s'_i=\mathrm{decode}(o_i).
$$

然后与真实下一帧 \(s'\) 逐候选计算 verifiable reward。RLVR-World 的忠实 reward 可写成：

$$
R_i^{\mathrm{RLVR}}
=
-\left[
\mathrm{MSE}(\hat s'_i,s')
+
\mathrm{LPIPS}_{\mathrm{vgg}}(\hat s'_i,s')
\right].
$$

GRPO 对同组 reward 做归一化：

$$
A_i=
\frac{R_i-\mathrm{mean}_{j}R_j}{\mathrm{std}_{j}R_j+\epsilon}.
$$

策略梯度为：

$$
\mathcal L_{\mathrm{PG}}
=
-\frac1G\sum_{i=1}^{G}A_i\log\pi_\theta(o_i|q).
$$

关键澄清：

- GRPO **不是**挑组内 reward 最大的候选做 SFT。
- 所有候选都参与更新。
- 高于均值的候选被提高概率，低于均值的候选被压低概率。
- 因而 reward 的**组内排序、优势符号、间隔可靠性**比绝对分数更关键。

---

## 2. Tokenized Video RLVR 的特殊问题

视频世界模型输出的是 token，不是图像。只要 reward 在图像上算，就必须经过：

$$
o_i
\xrightarrow{\mathrm{decode}}
\hat s'_i.
$$

但 tokenizer decoder 是有损的。即便模型完美预测了真实下一帧的 token：

$$
o_i=\mathrm{encode}(s'),
$$

它解码后最多也只能得到：

$$
\tilde s'=\mathrm{decode}(\mathrm{encode}(s')),
$$

而不是原始真值 \(s'\)。因此对任何 post-decode full-reference 度量 \(d\)，都存在：

$$
\phi_{\mathrm{tok}}^{(d)}
=
\mathbb E_{s'}\left[
d\left(
\mathrm{decode}(\mathrm{encode}(s')),
s'
\right)
\right].
$$

这就是本文定义的 **reward noise floor**。

它的三个性质：

1. **与预测质量无关**：真值自己 encode-decode 都回不到原图。
2. **由 decoder 引入**：LPIPS/DINO 这类特征指标只要输入是 decoded image，也仍然在 decoder 下游，仍带地板。
3. **度量依赖**：LPIPS/SSIM 等感知结构指标地板大；MSE 地板较小。

---

## 3. 为什么这个问题正好伤害 GRPO

把某个 post-decode reward 写成：

$$
R_i=R_i^\star+\eta_i,
$$

其中：

- \(R_i^\star\)：候选真实质量对应的干净 reward；
- \(\eta_i\)：decoder floor 诱发的候选级 reward 噪声。

GRPO 只关心同组内相对关系。如果 \(\eta_i\) 足够大，就会翻转候选排序。我们已有的理论和实证资产是：

$$
P_{\mathrm{flip}}
=
\frac{\arccos(\rho)}{\pi},
$$

其中 \(\rho\) 是含噪 reward 与干净 reward 的相关性。已验证：

- 5 seed 下实测翻转率约 \(0.185\)；
- 理论预测约 \(0.186\)；
- 二者高度吻合；
- 弱信号窗口翻转率更高；
- 翻转率与信号强度负相关。

这给了论文最硬的逻辑链：

$$
\text{decoder floor}
\rightarrow
\text{reward rank corruption}
\rightarrow
\text{GRPO advantage wrong sign / wrong order}
\rightarrow
\text{policy update learns noisy preference}.
$$

---

## 4. 当前方法总称：Reward Reliability Calibration

当前不再把方法讲成“code reward 更好”。code 只是一个实例。更准确的总方法是：

> **Reliability-calibrated verifiable reward for tokenized video world models.**

它包含三类校准。

---

## 5. 校准一：Floor-Cancelled Target Alignment

### 5.1 解码前实例：code reward

世界模型直接输出 token。我们可以不经过 decoder，在 FSQ 码向量空间比较候选与真值 token：

$$
z_i=\mathrm{indices\_to\_codes}(o_i),
\qquad
z'=\mathrm{indices\_to\_codes}(\mathrm{encode}(s')).
$$

reward：

$$
R_i^{\mathrm{code}}
=
-\mathrm{RMS}(z_i-z').
$$

优点：

- 不经过 decoder；
- reward floor 约为 0；
- 计算便宜；
- 与 tokenized world model 的输出空间一致。

但注意：不能再说“code 一定是更好的 reward”。原因：

- flow 优势不稳定；
- FD-DINO/KID 分布评测中 code 不赢 pixel/pixel_tok；
- code 与 tokenizer 同源，有 home advantage 嫌疑；
- 单靠 code 太单薄，撑不起整篇论文。

所以 code 现在的角色是：

> 一个干净的 floor-cancelled reward 实例，而不是全文唯一方法。

### 5.2 解码后实例：pixel_tok reward

另一种做法是不改变 post-decode metric，但把目标从原始 \(s'\) 换成 tokenizer 可达目标：

$$
\tilde s'
=
\mathrm{decode}(\mathrm{encode}(s')).
$$

reward：

$$
R_i^{\mathrm{pixel\_tok}}
=
-\mathrm{LPIPS}_{\mathrm{vgg}}(\hat s'_i,\tilde s').
$$

完美 token 预测时：

$$
\hat s'_i=\tilde s',
\qquad
R_i^{\mathrm{pixel\_tok}}=0.
$$

这比直接对原始 \(s'\) 更公平，因为原始 \(s'\) 本来就不是 tokenizer 可达的输出。

### 5.3 已有证据

`pixel_tok` 相比 `pixel` 的动态指标有稳定提升；`mse_tok` 作为负对照说明：

- LPIPS floor 大，`pixel -> pixel_tok` 动态增益大且 5/5 seed 一致；
- MSE floor 小，`mse -> mse_tok` 增益弱且不稳定。

这支持：

> floor-cancellation 的收益来自 floor，而不是任意换目标。

---

## 6. 校准二：Motion-Residual Reward

### 6.1 动机

单帧 full-reference reward 容易被静态外观主导。视频世界模型真正难的是动作条件下的状态变化。我们需要一个更正交的动态信号。

在 code 空间定义当前帧、候选下一帧、真值下一帧：

$$
z_t,\quad z_i,\quad z'.
$$

动态残差：

$$
\Delta z_i=z_i-z_t,
\qquad
\Delta z'=z'-z_t.
$$

不能使用：

$$
\|\Delta z_i-\Delta z'\|
$$

因为：

$$
\Delta z_i-\Delta z'
=(z_i-z_t)-(z'-z_t)
=z_i-z',
$$

它会退化回普通 code distance，没有新增动态信息。

因此使用方向与幅度：

$$
R_i^{\mathrm{dyn}}
=
\cos(\Delta z_i,\Delta z')
-
\gamma
\left|
\log
\frac{\|\Delta z_i\|+\epsilon}
{\|\Delta z'\|+\epsilon}
\right|.
$$

再与 base reward 融合：

$$
R_i
=
z(R_i^{\mathrm{base}})
+
\lambda z(R_i^{\mathrm{dyn}}),
$$

其中 \(z(\cdot)\) 是组内标准化。

当前实现：

- `pixel_tok_dyn`：

$$
z(R_i^{\mathrm{pixel\_tok}})
+
\lambda z(R_i^{\mathrm{dyn}})
$$

- `code_dyn`：

$$
z(R_i^{\mathrm{code}})
+
\lambda z(R_i^{\mathrm{dyn}})
$$

### 6.2 这条方法想证明什么

如果成立，它证明：

1. floor-cancelled reward 不是只能做静态保真；
2. 解码前 code 空间可以表达动作导致的动态残差；
3. 动态残差是与 LPIPS/code 距离不同的正交信号；
4. 在不改 RLVR 主架构下，可以通过 reward 设计提升动态预测。

### 6.3 当前 v9 初步结果如何解读

`dyn_lambda=0.25` 的 v9 中途结果显示：

- `pixel_tok_dyn` 相对 `pixel_tok`：
  - flow 平均提升约 \(+0.015\)，4/5 seed 提升；
  - dmotion 5/5 seed 提升；
  - 但 LPIPS/PSNR/SSIM 明显变差。

- `code_dyn` 相对 `code`：
  - flow 平均提升，3/4 已完成 seed 提升；
  - 但图像保真同样下降。

当前结论：

> motion residual 是有效动态信号，但 \(\lambda=0.25\) 太强。它目前能作为机制证明，不能直接作为最终主表 claim。

下一步应该跑更小权重：

$$
\lambda=0.10
$$

目标不是让 flow 暴涨，而是找 Pareto 点：

- flow / dmotion 多 seed 提升；
- LPIPS 恶化控制在很小范围；
- PSNR/SSIM 不明显塌。

---

## 7. 校准三：Rank-Reliability Advantage Weighting

### 7.1 动机

C2 已经证明 reward floor 会翻转组内排序。自然的算法问题是：

> 如果两个候选的 reward gap 小到接近噪声尺度，是否应该降低它们对 GRPO 更新的贡献？

这就是 `rankrel`。

### 7.2 公式

对候选 \(i,j\)，若 reward 噪声标准差为 \(\sigma_\eta\)，观察到 gap：

$$
|R_i-R_j|.
$$

局部排序可信度：

$$
c_{ij}
=
2\Phi
\left(
\frac{|R_i-R_j|}
{\sqrt{2}\sigma_\eta}
\right)
-1.
$$

实现中等价使用：

$$
c_{ij}
=
\mathrm{erf}
\left(
\frac{|R_i-R_j|}{2\sigma_\eta}
\right).
$$

每个候选取相邻排序中最不可靠的 gap：

$$
w_i
=
\min_{j\in\mathcal N(i)}c_{ij}.
$$

再作用到 advantage：

$$
\tilde A_i
=
w_iA_i,
\qquad
\tilde A_i
\leftarrow
\tilde A_i-\frac1G\sum_j\tilde A_j.
$$

加权后 re-center 是必要的，否则会引入整体 log-prob 偏置。

### 7.3 这条方法想证明什么

如果 `rankrel` 有效，它能把 C2 从诊断理论推进为优化方法：

$$
\text{rank flip theory}
\rightarrow
\text{rank confidence}
\rightarrow
\text{better GRPO update}.
$$

如果无效，也仍然能作为诚实消融：

> reward floor 的主要解决路径是 target/reward 设计，而不是改 GRPO。

---

## 8. 当前完整 Pipeline

当前推荐 pipeline 如下：

### Step 1. 固定 RLVR-World 前半部分

不改：

- RT-1 single-step 数据；
- CNNFSQ tokenizer；
- Llama world model；
- HF `.generate(top_k=100)` 采样；
- 每个 context/action 采样 \(G=16\) 个候选；
- GRPO 主训练框架。

### Step 2. 对每个候选计算多个候选 reward

已有/候选 reward：

| reward | 位置 | 目标 | 作用 |
|---|---|---|---|
| `pixel` | post-decode | raw GT | RLVR-style baseline |
| `a0faithful` | post-decode | raw GT | 忠实 RLVR baseline: MSE+LPIPS |
| `code` | pre-decode | GT token/code | floor-cancelled 实例 |
| `pixel_tok` | post-decode | decoded GT token | floor-cancelled 实例 |
| `code_dyn` | pre-decode | code + motion residual | 动态校准 |
| `pixel_tok_dyn` | mixed | pixel_tok + motion residual | 动态校准 |

### Step 3. 组内归一化成 advantage

默认：

$$
A_i=
\frac{R_i-\mathrm{mean}(R)}
{\mathrm{std}(R)+\epsilon}.
$$

可选 `rankrel`：

$$
\tilde A_i=w_iA_i.
$$

### Step 4. 用 GRPO 更新原世界模型

$$
\mathcal L
=
-\frac1G
\sum_i
\tilde A_i\log\pi_\theta(o_i|q).
$$

---

## 9. 论文故事怎么讲

### 9.1 Introduction 的故事

第一段：RLVR 让世界模型可以用 verifiable reward 自我改进。视频世界模型中，一个自然做法是把预测 token 解码成图像，再用 MSE/LPIPS 与真值帧比较。

第二段：但 tokenized video world model 有一个被忽视的问题：reward 并不直接作用在模型输出 token 上，而是作用在 decoded image 上。decoder 是有损的，因此 post-decode reward 带有与预测质量无关的 reward noise floor。

第三段：这个 floor 不是只影响绝对分数，而是破坏 GRPO 最关心的组内排序。我们用 \(R=R^\star+\eta\) 建模，并给出排序翻转概率 \(\arccos(\rho)/\pi\)，实测吻合。

第四段：因此，问题不只是“换一个更好的 metric”，而是“如何为 tokenized RLVR 设计可靠的 verifiable reward”。我们提出 reward reliability calibration，包括 floor-cancelled target alignment、motion-residual reward、rank-reliability advantage weighting。

第五段：实验证明：

- RL 训练本身显著优于 base；
- floor-cancelled reward 修复了 decoder floor；
- motion residual 能提供正交动态信号；
- rank flip 理论解释哪些 reward 更可靠；
- 多指标无脑融合和硬过滤并不稳定，说明可靠性校准比堆指标更关键。

### 9.2 Contributions 写法

建议贡献写成三条：

**C1. Reward noise floor.**  
We identify and quantify a reward noise floor in post-decode verifiable rewards for tokenized video world models.

**C2. Rank corruption mechanism.**  
We show that the floor corrupts within-group reward ranking, exactly the signal consumed by GRPO, and validate a closed-form flip-rate prediction.

**C3. Reliability-calibrated reward design.**  
We propose floor-cancelled target alignment and motion/rank reliability calibration, implemented as small opt-in modifications to the reward/advantage while leaving the RLVR pipeline unchanged.

### 9.3 不要这样讲

不要写：

- “code reward is better than pixel reward”；
- “token feature reward beats image reward”；
- “FID proves code is better”；
- “GRPO selects the best candidate”；
- “多指标融合一定更强”。

这些都容易被现有负结果或审稿人击穿。

应该写：

- “pre/post-decode is the relevant axis, not pixel/feature”；
- “reward reliability matters because GRPO consumes ranks”；
- “floor-cancelled and motion-aware rewards improve the reliability of the training signal”；
- “distribution metrics are evaluation-only, not per-sample rewards”。

---

## 10. 实验矩阵如何支撑故事

### 10.1 必备实验

| 实验 | 目的 | 成功判据 |
|---|---|---|
| floor metrics | 证明 post-decode reward 有不可约 floor | 多度量 \(\phi_{\mathrm{tok}}>0\)，感知指标更大 |
| rank flip analysis | 证明 floor 伤 GRPO 排序 | 实测 flip rate 贴合 \(\arccos(\rho)/\pi\) |
| pixel vs pixel_tok | 证明 floor-cancelled target 有效 | dynamic/flow 多 seed 提升 |
| mse vs mse_tok | 负对照 | MSE floor 小，对应增益弱 |
| v9 dyn lambda scan | 证明 motion residual 是正交动态信号 | flow/dmotion 提升，保真不明显崩 |
| rankrel | 验证 C2 能否变方法 | 对高 floor reward 有收益，或作为负结果 |
| FD-DINO/KID/PRDC | 回应导师“特征/分布评测” | 只做 evaluation，不作为 reward |

### 10.2 当前 v9 之后最该做的实验

现在 `dyn_lambda=0.25` 证明了机制，但牺牲画质。下一步应跑：

```bash
python scripts/train_grpo.py \
  --rewards pixel_tok,code,pixel_tok_dyn,code_dyn \
  --modes gt_only \
  --seeds 0,1,2,3,4 \
  --steps 150 --K 16 \
  --eval_every 10 \
  --dyn_lambda 0.10 --dyn_gamma 0.25 \
  --out_dir outputs/grpo_v9_dyn_lam010
```

若 \(\lambda=0.10\) 仍画质下降明显，再跑：

```bash
--dyn_lambda 0.05
```

最终主表应选 Pareto 最好的 \(\lambda\)，而不是当前机制最强的 \(0.25\)。

---

## 11. 如何写当前结果

如果只基于目前已有结果，建议这样写：

> Preliminary motion-residual results show that the auxiliary dynamic term consistently increases motion-sensitive metrics, improving dmotion in 5/5 seeds and flow in most seeds. However, a large weight (\(\lambda=0.25\)) trades off perceptual fidelity, indicating that the term should be treated as a calibration signal rather than a replacement objective. We therefore tune \(\lambda\) to identify a Pareto point that preserves image fidelity while improving dynamics.

中文口径：

> 当前动态残差项已经证明“方向是有效的”：它稳定推高 dmotion，并多数 seed 推高 flow。但 \(\lambda=0.25\) 过强，牺牲 LPIPS/PSNR/SSIM，因此不能直接作为最终主结果。下一步要做的是找 Pareto 权重，而不是否定这个方向。

---

## 12. 当前最稳的论文主张

截至 2026-06-30，最稳主张是：

1. **问题主张稳**：post-decode reward 有 tokenizer-induced reward floor。
2. **机理主张稳**：floor 通过 rank flip 伤害 GRPO，且闭式翻转率已验证。
3. **floor-cancellation 主张较稳**：`pixel_tok` 与 `mse_tok` 负对照支持“去地板收益随地板大小缩放”。
4. **motion residual 是正在形成的方法贡献**：当前证明有动态信号，但需要 \(\lambda\) scan 找到不牺牲保真的版本。
5. **code 单独作为最终方法不稳**：只能作为 floor-cancelled pre-decode 实例，而非全文核心 claim。

---

## 13. 最后给导师的一句话

> 我们现在不是在证明某个单一指标 code 一定比 LPIPS 好，而是在证明 tokenized video RLVR 的 reward 必须做可靠性校准：decoder floor 会污染 post-decode reward 的组内排序，而 GRPO 正依赖这个排序；因此 reward 应对齐 tokenizer 可达目标，并补充真正正交的动态残差信号，同时用 rank confidence 控制不可靠优势。现有结果已经支持 floor 与排序翻转机理，motion residual 初步证明能提升动态指标，下一步是调低权重找到不牺牲保真的 Pareto 点。
