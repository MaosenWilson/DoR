# V9 — code-space motion residual reward（动态残差校准，2026-06-30）

> 目的：验证在 floor-cancelled reward 之外，是否可以加入一个解码前 code-space motion residual 信号，提高下一帧动态一致性，同时尽量不牺牲图像保真。  
> 结论先行：`pixel_tok_dyn` 的动态残差信号有效；`λ=0.25` 过强、牺牲画质；`λ=0.10` 当前是最佳 Pareto 点；`λ=0.05` 更保守、画质更稳但动态提升变弱。`code_dyn` 不适合作为主线。

---

## 1. 方法定义

基础 floor-cancelled perceptual reward：

$$
\tilde{s}'=\mathrm{decode}(\mathrm{encode}(s')),
\qquad
R_i^{\mathrm{pixel\_tok}}
=
-\mathrm{LPIPS}(\hat{s}'_i,\tilde{s}').
$$

code-space motion residual：

$$
\Delta z_i=z_i-z_t,\qquad
\Delta z'=z'-z_t,
$$

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

训练 reward：

$$
R_i^{\mathrm{pixel\_tok\_dyn}}
=
z(R_i^{\mathrm{pixel\_tok}})
+
\lambda z(R_i^{\mathrm{dyn}}),
$$

$$
R_i^{\mathrm{code\_dyn}}
=
z(R_i^{\mathrm{code}})
+
\lambda z(R_i^{\mathrm{dyn}}).
$$

其中 \(z(\cdot)\) 是组内标准化。当前 \(\gamma=0.25\)。

---

## 2. 实验设置

- 数据/任务：RT-1 single-step, `fractal20220817`
- 模型：base RT-1 world model + GRPO 微调
- 采样：`K=16`, HF `generate(top_k=100)`
- 训练：150 steps, 5 seeds
- 评测：RLVR Evaluator（LPIPS-vgg / PSNR / SSIM）+ 我们的 RAFT flow / dmotion
- 输出目录：
  - `outputs/grpo_v9_dyn`：`λ=0.25`, rewards = `pixel_tok,code,pixel_tok_dyn,code_dyn`
  - `outputs/grpo_v9_dyn_lam010`：`λ=0.10`, rewards = `pixel_tok_dyn,code_dyn`
  - `outputs/grpo_v9_dyn_lam005_pixelonly`：`λ=0.05`, reward = `pixel_tok_dyn`（当前记录为前 3 seed 中途结果）

---

## 3. 完整结果：λ=0.25

对比 `outputs/grpo_v9_dyn` 内同轮 base。

| reward | n | LPIPS↓ | PSNR↑ | SSIM↑ | flow↑ |
|---|---:|---:|---:|---:|---:|
| code | 5 | **0.1398±0.0016** | **25.27** | **0.7998** | 0.2447±0.0286 |
| code_dyn | 5 | 0.1437±0.0018 | 25.00 | 0.7936 | 0.2692±0.0460 |
| pixel_tok | 5 | 0.1414±0.0017 | 25.20 | 0.7979 | 0.2668±0.0319 |
| pixel_tok_dyn | 5 | 0.1449±0.0007 | 24.99 | 0.7936 | **0.2822±0.0240** |

### 配对差值：pixel_tok_dyn − pixel_tok

| 指标 | mean Δ | per-seed Δ | wins |
|---|---:|---|---:|
| flow↑ | +0.0154 | +0.0349,+0.0190,+0.0369,+0.0056,−0.0195 | 4/5 |
| dmotion↑ | +0.0168 | +0.0354,+0.0158,+0.0293,+0.0009,+0.0028 | 5/5 |
| LPIPS↓ | +0.0034 | +0.0030,+0.0037,+0.0050,+0.0016,+0.0040 | 0/5 |
| PSNR↑ | −0.2162 | −0.1117,−0.3053,−0.4663,+0.0079,−0.2055 | 1/5 |
| SSIM↑ | −0.0043 | −0.0028,−0.0049,−0.0082,−0.0018,−0.0037 | 0/5 |

### 解释

`λ=0.25` 证明 motion residual 不是噪声：它稳定提高 dmotion，并多数 seed 提高 flow。但权重过强，明显牺牲 LPIPS/PSNR/SSIM。因此它适合作为“机制证明”，不适合作为最终主结果。

---

## 4. 完整结果：λ=0.10

`outputs/grpo_v9_dyn_lam010`，与历史 `outputs/grpo_v5/floorcancel/pixel_tok` 对比。

| reward | n | LPIPS↓ | PSNR↑ | SSIM↑ | flow↑ | dmotion↑ |
|---|---:|---:|---:|---:|---:|---:|
| code_dyn | 5 | 0.1419±0.0023 | 25.13 | 0.7966 | 0.2411±0.0416 | 0.1572 |
| pixel_tok_dyn | 5 | 0.1428±0.0018 | 25.10 | 0.7961 | **0.2877±0.0257** | 0.1766 |

### 配对差值：pixel_tok_dyn(λ=0.10) − historical pixel_tok

| 指标 | mean Δ | per-seed Δ | wins |
|---|---:|---|---:|
| flow↑ | +0.0200 | +0.0061,+0.0039,+0.0899,+0.0304,−0.0304 | 4/5 |
| dmotion↑ | +0.0092 | +0.0001,+0.0012,+0.0348,+0.0118,−0.0020 | 4/5 |
| LPIPS↓ | +0.0004 | −0.0026,+0.0005,−0.0011,+0.0008,+0.0042 | 2/5 |
| PSNR↑ | −0.0173 | +0.1367,−0.0109,+0.0041,+0.0863,−0.3025 | 3/5 |
| SSIM↑ | −0.0002 | +0.0043,−0.0008,+0.0004,+0.0010,−0.0058 | 3/5 |

### 与 λ=0.25 的对比：pixel_tok_dyn(λ=0.10) − pixel_tok_dyn(λ=0.25)

| 指标 | mean Δ | wins |
|---|---:|---:|
| flow↑ | +0.0055 | 1/5 |
| dmotion↑ | −0.0047 | 1/5 |
| LPIPS↓ | **−0.0021** | **5/5** |
| PSNR↑ | **+0.1137** | 4/5 |
| SSIM↑ | **+0.0024** | 4/5 |

### 解释

`λ=0.10` 是目前最好的 Pareto 点：相对 `pixel_tok`，flow/dmotion 多数 seed 提升，而 LPIPS/PSNR/SSIM 基本不掉；相对 `λ=0.25`，显著修复画质损失。当前最可能写成主方法结果的是：

$$
R_i
=
z(R_i^{\mathrm{pixel\_tok}})
+
0.10\,z(R_i^{\mathrm{dyn}}).
$$

`code_dyn` 在 `λ=0.10` 下仍不理想：相对 code，flow 平均下降或不稳，LPIPS/PSNR/SSIM 也没有优势。因此 `code_dyn` 不进入主线，只作为负结果/消融。

---

## 5. 中途结果：λ=0.05（前 3 seed）

`outputs/grpo_v9_dyn_lam005_pixelonly`，当前完成 seed 0,1,2。

| seed | reward | LPIPS↓ | PSNR↑ | SSIM↑ | flow↑ | dmotion↑ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | pixel_tok_dyn | 0.1413 | 25.13 | 0.7961 | 0.2577 | 0.1796 |
| 1 | pixel_tok_dyn | 0.1420 | 25.22 | 0.7978 | 0.2640 | 0.1818 |
| 2 | pixel_tok_dyn | 0.1401 | 25.29 | 0.7992 | 0.2329 | 0.1632 |

### 配对差值：pixel_tok_dyn(λ=0.05) − historical pixel_tok（seed 0,1,2）

| 指标 | mean Δ | per-seed Δ | wins |
|---|---:|---|---:|
| flow↑ | −0.0101 | −0.0346,+0.0080,−0.0037 | 1/3 |
| dmotion↑ | +0.0051 | −0.0173,+0.0150,+0.0176 | 2/3 |
| LPIPS↓ | **−0.0016** | −0.0027,−0.0004,−0.0016 | **3/3** |
| PSNR↑ | **+0.0853** | +0.0649,+0.0867,+0.1041 | **3/3** |
| SSIM↑ | **+0.0015** | +0.0016,+0.0012,+0.0018 | **3/3** |

### 解释

`λ=0.05` 更保守，图像保真明显更稳，LPIPS/PSNR/SSIM 前 3 seed 全部改善；但动态提升基本消失，flow 平均下降。当前判断：`λ=0.05` 适合作为 sensitivity ablation，说明过小权重会保守到失去动态收益；主结果仍更倾向 `λ=0.10`。

---

## 6. 当前结论

1. **方法没有失效。** Motion residual 确实包含动态信号，`λ=0.25` 的 dmotion 5/5 与 flow 4/5 提升证明了这一点。
2. **直接强加动态项会伤保真。** `λ=0.25` 明显降低 LPIPS/PSNR/SSIM。
3. **`λ=0.10` 当前最佳。** 相对 `pixel_tok`，flow +0.0200、dmotion +0.0092，均为 4/5 seed 提升，同时 LPIPS/PSNR/SSIM 基本不变。
4. **`λ=0.05` 太保守。** 画质更稳，但 flow 提升消失。
5. **主线应是 `pixel_tok_dyn`，不是 `code_dyn`。** 动态残差更适合作为 floor-cancelled perceptual reward 的弱校准项，而不是直接叠在 code reward 上。

---

## 7. 写作口径

推荐写成：

> Motion residual calibration provides an orthogonal pre-decode dynamic signal. A large weight over-emphasizes motion and hurts perceptual fidelity, while a small weight is too conservative. At \(\lambda=0.10\), the calibrated reward improves flow/dmotion in most seeds while preserving LPIPS/PSNR/SSIM, yielding the best fidelity-motion Pareto trade-off.

中文口径：

> 动态残差校准不是替代 LPIPS/code 的主 reward，而是一个弱校准项。它在解码前 code 空间提供动作导致的变化信号。过强会牺牲保真，过弱动态收益消失；当前 \(\lambda=0.10\) 在动态提升和图像保真之间取得最佳折中。

---

## 8. 下一步

1. 等 `λ=0.05` 跑完 5 seed，补全 sensitivity 表。
2. 对 `pixel_tok_dyn(λ=0.10)` 跑 FD-DINO/KID/PRDC，确认分布级特征质量不崩。
3. 若时间允许，对最终 `pixel_tok_dyn(λ=0.10)` 做 3-5 个可视化 case，重点展示动作/运动方向更接近 GT，且图像质量无明显下降。
