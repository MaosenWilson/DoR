# DoR 奖励设计 vs RLVR-World（公式版）

> 本文所有公式用块级 `$$...$$`，Typora 默认即渲染（不需要开 Inline Math）。
> 记号：组内 $K$ 个候选；候选 $i$ 的视觉 token $o_i$，解码帧 $\hat s'_i=\mathrm{decode}(o_i)$，码空间向量 $z_i=\mathrm{indices\_to\_codes}(o_i)$；真值帧 $s'$、真值码 $z'$。

---

## 一、RLVR-World 的奖励（代码核实）

来源 `verl/verl/trainer/ppo/ray_trainer.py`（`RayVGPTPPOTrainer`, 行 ~1215–1265）：
`loss = recon_loss + perceptual_loss`，`reward = -loss`，其中 recon 由 `trainer.reward_fn` 选 MSE 或 MAE，perceptual 为 vgg-LPIPS。

$$
R^{\mathrm{RLVR}}_i \;=\; -\Big(\, d_{\mathrm{recon}}(\hat s'_i,\,s') \;+\; \mathrm{LPIPS}_{\mathrm{vgg}}(\hat s'_i,\,s') \,\Big)
$$

$$
d_{\mathrm{recon}}(\hat s',s') \;=\; \mathrm{MSE}\ \text{或}\ \mathrm{MAE}
$$

**特点**：两分量（重建 + 感知）、**全在解码后**、**等权 1:1**、全图粒度、无特征层。
（`LOSS_KEYS = [lpips, mse, mae, ssim, psnr]` 仅用于日志，不是各自一个奖励。）

> 推论：我们主表里 A0 = $-\mathrm{LPIPS}$ 是**简化**；忠实基线应为 $A0' = -(\mathrm{MSE}+\mathrm{LPIPS})$。

---

## 二、我们的奖励（DoR：地板感知的多空间加权可验证奖励）

### 2.1 分量（每个都是"越大越好"的负距离）

| 分量 $m$ | 公式 | 空间 | 地板 $\phi_m$ |
|---|---|---|---|
| **code**（核心） | $r^{\mathrm{code}}_i=-\,\mathrm{RMS}(z_i-z')$ | 解码前 | $\approx 0$ |
| recon | $r^{\mathrm{recon}}_i=-\,\mathrm{MSE}(\hat s'_i,s')$ | 解码后 | 低 |
| perc | $r^{\mathrm{perc}}_i=-\,\mathrm{LPIPS}_{\mathrm{vgg}}(\hat s'_i,s')$ | 解码后 | 高 |
| feat（可选） | $r^{\mathrm{feat}}_i=-\,d_{\mathrm{DINOv2}}(\hat s'_i,s')$（逐对） | 解码后 | 中 |
| 多粒度（可选） | code 的全局池化 + 逐空间格两档 | 解码前 | $\approx 0$ |

### 2.2 统一的逐候选奖励（一行闭式，与 RLVR 平行）

奖励是**纯逐候选函数**（组内归一化交给 GRPO，见 2.4）：

$$
R^{\mathrm{DoR}}_i \;=\; -\sum_{m\in\mathcal M}\, w_m\,\frac{d_m(i)}{s_m}
$$

展开（取 $\mathcal M=\{\mathrm{code},\mathrm{recon},\mathrm{perc}\}$）：

$$
R^{\mathrm{DoR}}_i \;=\; -\Big(\, w_{\mathrm{code}}\tfrac{\mathrm{RMS}(z_i-z')}{s_{\mathrm{code}}} \;+\; w_{\mathrm{recon}}\tfrac{\mathrm{MSE}(\hat s'_i,s')}{s_{\mathrm{recon}}} \;+\; w_{\mathrm{perc}}\tfrac{\mathrm{LPIPS}(\hat s'_i,s')}{s_{\mathrm{perc}}} \,\Big)
$$

- $d_m(i)$：分量 $m$ 的距离（见上表）；$\hat s'_i=\mathrm{decode}(o_i)$，$z_i=\mathrm{indices\_to\_codes}(o_i)$。
- $s_m$：度量 $m$ 的**离线尺度**（数据集 std），消量纲、使各项可加（**离线常数，非逐组**）。
- $w_m$：**地板感知权重**（核心创新，离线常数）：

$$
w_m \;=\; \frac{\sigma^{(m)2}_{\star}}{\,\sigma^{(m)2}_{\star} + \phi_m^{2}\,}
$$

$\sigma^{(m)2}_{\star}$=分量 $m$ 的组内信号方差，$\phi_m$=该度量地板（进展4 离线测）。code 项 $\phi\approx0\Rightarrow w\approx1$（满权）；LPIPS 等高地板 $\Rightarrow w$ 小（自动降权）。

### 2.3 RLVR-World 是它的特例（退化）

$$
\mathcal M=\{\mathrm{recon},\mathrm{perc}\},\;\; w_m=1,\;\; s_m=1 \;\;\Longrightarrow\;\; R^{\mathrm{DoR}}_i = -(\mathrm{MSE}+\mathrm{LPIPS}) = R^{\mathrm{RLVR}}_i
$$

### 2.4 组内归一化交给 GRPO（与 RLVR 相同，不属于奖励本身）

$$
A_i \;=\; \frac{\,R_i - \mathrm{mean}_{k}\,R\,}{\,\mathrm{std}_{k}\,R + \epsilon\,}
$$

进**改进版 GRPO**（Dr. GRPO 去偏 / DAPO 动态采样）+ **地板感知组过滤**：丢弃退化组

$$
\text{保留组当且仅当}\quad \sigma_{\star}^{2} \;\gtrsim\; \sigma_{\eta}^{2}\ (\approx \phi^2)
$$

> **职责分清**：奖励 $R_i$ = 逐候选（2.2，一行闭式，平行 RLVR）；优势 $A_i$ = 组内归一化（2.4，GRPO 的活，两者通用）。早前把组内 z-score 写进奖励是职责错配，已归位。

---

## 三、四维对照

| 维度 | RLVR-World | DoR（本文） |
|---|---|---|
| 分量 | recon + LPIPS（2 项） | **+ code（解码前）**、可选 DISTS/DINOv2、多粒度 |
| 空间 | 全**解码后**（带地板） | 以**解码前码空间（≈零地板）**为主 + 解码后辅 |
| 权重 | **等权 1:1** | **地板感知** $w_m=\dfrac{\sigma_\star^2}{\sigma_\star^2+\phi_m^2}$ |
| 粒度 | 全图 | 全局 / 逐格 / 多层 |
| GRPO | 原始 | Dr.GRPO / DAPO + 地板感知组过滤 |

---

## 四、退化保证（连续可消融路径）

只保留 perc + recon、令所有 $w_m$ 相等、且只用解码后分量时：

$$
R^{\mathrm{DoR}}_i \;\xrightarrow[\text{解码后、等权、仅 recon+perc}]{}\; R^{\mathrm{RLVR}}_i
$$

即**精确退回 RLVR-World**。于是"从他们 → 我们"是一条连续路径，每加一维（空间 → 权重 → 粒度 → GRPO）都是一个可单独验证的 ablation。

---

## 五、诚实标注 / 待定

1. $w_m$ 中的 $\sigma^{(m)2}_\star$ 需在 `cache_reward_spaces` 缓存数据上**离线估**（已存逐候选各指标，可直接算组内方差），非拍脑袋。
2. 本文为**设计稿**：要把 $R^{\mathrm{DoR}}$ 实现成新臂、与各单分量对比，验证"加权多空间 $>$ 等权两项"后才算坐实。
3. $w_m$ **固定（离线估）vs 自适应（逐组在线估）**：建议先固定（简单、可复现），自适应留 ablation。
4. 待补忠实基线 $A0' = -(\mathrm{MSE}+\mathrm{LPIPS})$。

---

## 六、权重 $w_m$ 的来源（推导 + 测量 + 验证，杜绝"凭空"）

审稿红线：$w_m$ 必须有出处和验证，不能硬编。完整链条三段：

### 6.1 推导（估计理论的标准结果，非拍脑袋）
把每个分量视为"真质量 $q_i$"的带噪观测（尺度归一后）：

$$
r^{(m)}_i = q_i + \eta^{(m)}_i,\qquad \mathrm{Var}(\eta^{(m)})=\phi_m^2,\qquad \mathrm{Var}_{\text{组内}}(q)=\sigma_\star^2
$$

分量 $m$ 的**可靠度（信号占比）**= 它与真质量的相关平方：

$$
\rho_m^2 \;=\; \frac{\sigma_\star^2}{\sigma_\star^2+\phi_m^2}
$$

合并多个带噪观测以最优估计 $q$ 的权 = **逆方差 / 可靠度加权**（Gauss–Markov BLUE、维纳收缩——标准结论）。取 $w_m=\rho_m^2$ 即此可靠度。**形式有出处。**

### 6.2 测量（输入是测出来的，非自由参数）
- $\phi_m$：进展4 的往返地板 $d_m(\mathrm{decode}(\mathrm{encode}(s')),s')$，**每个度量实测**。
- $\sigma^{(m)2}_\star$：`cache_reward_spaces` 逐候选各指标 → 组内方差，离线估。
- 二者确定后 $w_m$ 随之固定，**无可调旋钮**。

### 6.3 诚实弱环 + 验证（必须做，不能只靠推导）
推导假设"各度量是同一 $q$ 的带噪观测"，但实际它们测**不同**东西（LPIPS=感知、MSE=像素、code=动态）→ **不宣称"可证最优"**，须实验验证：
- **V1（主）**：floor-weighted $R^{\mathrm{DoR}}$ 作一个臂，对比**等权**与**单分量**——赢了才算权重有用。
- **V2（敏感性）**：在权重上加温度 $w_m^{\tau}$（$\tau:0\to\infty$，从等权到硬门控），画"性能 vs $\tau$"，看峰值是否落在地板推导权重附近——落在附近=推导被数据印证。
- **V3（退化/兜底）**：若连续式不灵，退到**硬门控**（丢弃 $\phi_m>\sigma_\star$ 的地板主导分量、其余等权），直接由进展4 的"谁淹没谁"判定，近乎无参数、最稳。

> 纪律：若 V1/V2 不成立，**如实改用等权或门控**，绝不硬编一个赢的权重。

> 配套文献与引用见 `docs/aaai2027/lit_metrics_grpo.md`。
