# Reward 设计与当前实验结论简表（2026-07-01）

> 当前结论：单独说 `code reward 更好` 不稳；更合理的主线是 **floor-cancelled post-decode reward + weak pre-decode motion residual calibration**。  
> 目前最优候选是 `pixel_tok_dyn` with \(\lambda=0.10\)：动态指标提升，LPIPS/PSNR/SSIM 基本不掉。

---

## 1. 最终推荐的 reward 形式

### 1.1 基础：tokenizer 可达目标

世界模型输出候选 token：

$$
o_i\sim \pi_\theta(\cdot|s,a).
$$

解码为预测图像：

$$
\hat{s}'_i=\mathrm{decode}(o_i).
$$

真值下一帧为 \(s'\)。由于 tokenizer 有损，原始 \(s'\) 不一定是 tokenized model 可达的图像。因此定义可达目标：

$$
\tilde{s}'=\mathrm{decode}(\mathrm{encode}(s')).
$$

---

### 1.2 Post-decode floor-cancelled perceptual reward

基础 reward 不再直接对原始 GT \(s'\)，而是对 tokenizer 可达目标 \(\tilde{s}'\)：

$$
R_i^{\mathrm{pix\_tok}}
=
-
\mathrm{LPIPS}_{\mathrm{vgg}}(\hat{s}'_i,\tilde{s}').
$$

这一步的作用是减少 decoder reconstruction floor 对 reward 的污染。完美 token 预测时：

$$
o_i=\mathrm{encode}(s')
\Rightarrow
\hat{s}'_i=\tilde{s}'
\Rightarrow
R_i^{\mathrm{pix\_tok}}=0.
$$

---

### 1.3 Pre-decode motion residual reward

将预测 token、当前帧 token、真值下一帧 token 映射到 tokenizer code/latent feature：

$$
z_i=\mathrm{indices\_to\_codes}(o_i),
$$

$$
z_t=\mathrm{indices\_to\_codes}(\mathrm{encode}(s_t)),
$$

$$
z'=\mathrm{indices\_to\_codes}(\mathrm{encode}(s')).
$$

定义 code-space motion residual：

$$
\Delta z_i=z_i-z_t,\qquad
\Delta z'=z'-z_t.
$$

不能使用 \(\|\Delta z_i-\Delta z'\|\)，因为：

$$
\Delta z_i-\Delta z'
=(z_i-z_t)-(z'-z_t)
=z_i-z',
$$

会退化成普通 code distance。当前使用方向一致性 + 幅度比例：

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

当前实验中：

$$
\gamma=0.25.
$$

---

### 1.4 最终推荐 reward：弱动态校准

最终推荐使用组内标准化后的加权和：

$$
R_i^{\mathrm{final}}
=
z_G(R_i^{\mathrm{pix\_tok}})
+
\lambda
z_G(R_i^{\mathrm{dyn}}),
$$

其中：

$$
z_G(x_i)
=
\frac{x_i-\mathrm{mean}_{j=1}^{G}x_j}
{\mathrm{std}_{j=1}^{G}x_j+\epsilon}.
$$

当前最佳权重：

$$
\lambda=0.10,\qquad \gamma=0.25.
$$

对应代码 reward 名：

```text
pixel_tok_dyn
```

---

## 2. GRPO 如何使用 reward

GRPO 不是挑组内 reward 最大的候选做 SFT，而是所有候选都参与更新。给定每组 \(G\) 个候选的 reward：

$$
R_1,\ldots,R_G,
$$

优势为：

$$
A_i
=
\frac{R_i-\mathrm{mean}_jR_j}
{\mathrm{std}_jR_j+\epsilon}.
$$

策略梯度：

$$
\mathcal{L}_{\mathrm{PG}}
=
-
\frac1G
\sum_{i=1}^G
A_i\log\pi_\theta(o_i|s,a).
$$

因此 reward 设计的核心不是绝对分数，而是组内排序和相对间隔是否可靠。

---

## 3. 实验结果总结

### 3.1 主表：不同 \(\lambda\) 的动态校准

| reward | \(\lambda\) | n | LPIPS↓ | PSNR↑ | SSIM↑ | flow↑ |
|---|---:|---:|---:|---:|---:|---:|
| `pixel_tok` | 0 | 5 | 0.1424±0.0015 | 25.1161 | 0.7963 | 0.2677±0.0290 |
| `pixel_tok_dyn` | 0.25 | 5 | 0.1449±0.0007 | 24.9852 | 0.7936 | 0.2822±0.0240 |
| `pixel_tok_dyn` | **0.10** | 5 | 0.1428±0.0018 | 25.0988 | 0.7961 | **0.2877±0.0257** |
| `pixel_tok_dyn` | 0.05 | 5 | 0.1428±0.0023 | 25.0656 | 0.7954 | 0.2488±0.0163 |

读法：

- \(\lambda=0.25\)：动态更强，但 LPIPS/PSNR/SSIM 明显变差。
- \(\lambda=0.05\)：过于保守，动态收益消失。
- \(\lambda=0.10\)：当前最佳 Pareto 点，动态提升且图像保真基本保持。

---

### 3.2 关键配对：\(\lambda=0.10\) vs `pixel_tok`

`pixel_tok_dyn(λ=0.10) - pixel_tok`：

| 指标 | mean Δ | per-seed Δ | wins |
|---|---:|---|---:|
| flow↑ | **+0.0200** | +0.0061,+0.0039,+0.0899,+0.0304,−0.0304 | 4/5 |
| dmotion↑ | **+0.0092** | +0.0001,+0.0012,+0.0348,+0.0118,−0.0020 | 4/5 |
| LPIPS↓ | +0.0004 | −0.0026,+0.0005,−0.0011,+0.0008,+0.0042 | 2/5 |
| PSNR↑ | −0.0173 | +0.1367,−0.0109,+0.0041,+0.0863,−0.3025 | 3/5 |
| SSIM↑ | −0.0002 | +0.0043,−0.0008,+0.0004,+0.0010,−0.0058 | 3/5 |

结论：

> \(\lambda=0.10\) 在 4/5 seed 上提升 flow/dmotion，同时 LPIPS/PSNR/SSIM 基本不变。因此它可以作为当前主方法结果。

---

### 3.3 \(\lambda=0.05\) vs `pixel_tok`

`pixel_tok_dyn(λ=0.05) - pixel_tok`：

| 指标 | mean Δ | per-seed Δ | wins |
|---|---:|---|---:|
| flow↑ | −0.0190 | −0.0346,+0.0080,−0.0037,−0.0202,−0.0443 | 1/5 |
| dmotion↑ | +0.0012 | −0.0173,+0.0150,+0.0176,−0.0110,+0.0019 | 3/5 |
| LPIPS↓ | +0.0003 | −0.0027,−0.0004,−0.0016,+0.0013,+0.0049 | 3/5 |
| PSNR↑ | −0.0505 | +0.0649,+0.0867,+0.1041,−0.1433,−0.3648 | 3/5 |
| SSIM↑ | −0.0008 | +0.0016,+0.0012,+0.0018,−0.0018,−0.0068 | 3/5 |

结论：

> \(\lambda=0.05\) 太弱，保真没有明显优势，动态收益消失。因此它适合作为 sensitivity ablation，不作为主结果。

---

## 4. 当前论文故事怎么讲

当前不建议继续讲：

> code reward is better.

更稳的说法是：

> tokenized video RLVR 的 verifiable reward 需要可靠性校准。Post-decode LPIPS/MSE reward 直观但受 decoder floor 影响；将目标换成 tokenizer 可达重建可以减少 floor；再加入一个弱的 pre-decode motion residual，可以在不牺牲图像保真的情况下提升动态一致性。

对应贡献可以写成：

1. **Reward noise floor**：定义并刻画 tokenized video RLVR 中 decoder-induced reward floor。
2. **Rank corruption mechanism**：说明 floor 如何破坏 GRPO 消费的组内排序，并用 flip-rate 公式验证。
3. **Reliability-calibrated reward**：提出 `pixel_tok + weak code-space motion residual`，即：

$$
R_i^{\mathrm{final}}
=
z_G\left(
-
\mathrm{LPIPS}
(\hat{s}'_i,\tilde{s}')
\right)
+
0.10
z_G
\left(
R_i^{\mathrm{dyn}}
\right).
$$

---

## 5. 现在的结论

1. `code` 不是最终主线，只作为 pre-decode reward 实例。
2. `pixel_tok` 是更稳的 floor-cancelled base reward。
3. `pixel_tok_dyn(λ=0.10)` 是当前最佳方法候选。
4. `λ=0.25` 证明 motion residual 有效但过强。
5. `λ=0.05` 证明 motion residual 太弱会失去动态收益。
6. 下一步应检查 `pixel_tok_dyn(λ=0.10)` 的 FD-DINO/KID/PRDC，确认分布级特征质量不崩。

