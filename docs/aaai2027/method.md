# Method — Temporal RC-GRPO

> **当前主线**：单步实验负责 verifier floor diagnosis 和 RC calibration；多步实验负责 GRPO 侧创新，即 temporal credit assignment 与 horizon-aware drift control。  
> **不再作为主方法**：GSPO、REAL-style VPO、segmental single-step GRPO、codec_fused/CAST 旧路线均作为 intervention-locus 负结果或历史记录，不再写成本文方法。

## 1. RLVR-World 骨架

给定 context/action：

$$
q=(s,a),
$$

视频世界模型对同一转移采样一组候选 token 序列：

$$
\hat{z}_{i,1:H}\sim\pi_{\theta}(\cdot\mid q),\qquad i=1,\ldots,G.
$$

候选经 decoder 得到预测帧：

$$
\hat{s}'_i = D(\hat{z}_i).
$$

RLVR-World 的 verifier 对每个候选给出可验证 reward：

$$
R_i = R(\hat{s}'_i,s'),
$$

然后 GRPO 使用组内归一化优势：

$$
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
$$

本文不改变 RLVR-World 的基本骨架：仍然是 sample group、verifiable reward、group-relative policy update。改动发生在两个位置：

1. single-step：校准 verifier target；
2. multi-step：把 rollout-level advantage 改成 temporal frame-block advantage。

## 2. Single-Step Verifier Calibration

### 2.1 Codec Reconstruction Floor

tokenizer 往返会产生不可约重建残差：

$$
\phi_{\mathrm{tok}}^{(d)}
=
\mathbb{E}_{s'}[d(D(E(s')),s')].
$$

它不是模型预测错误，而是 decoder / tokenizer 对真实帧本身的不可达误差。post-decode reward 如 LPIPS、MSE、SSIM、DINO feature distance 都会看到这个 floor；pre-decode code-space reward 不经过 decoder，因此没有同一类 floor。

GRPO 只消费组内排序。若含 floor 的 reward 与无 floor 参照的组内相关为 $\rho$，排序翻转概率满足：

$$
P_{\mathrm{flip}}
=
\frac{\arccos(\rho)}{\pi}.
$$

这一定律已在多个 reward 上离线验证，因此 single-step 的主要作用是诊断 verifier-side rank corruption。

### 2.2 Reachable-Target Calibration

真实帧 $s'$ 不一定是 tokenizer 可达的 decoded frame。我们把 verifier target 改为：

$$
\tilde{s}' = D(E(s')).
$$

RC reward：

$$
R_i^{\mathrm{RC}}
=
z_G\big(-\mathrm{LPIPS}(\hat{s}'_i,\tilde{s}')\big)
+
\lambda_{\mathrm{dyn}}z_G(R_i^{\mathrm{dyn}}),
$$

其中当前冻结：

$$
\lambda_{\mathrm{dyn}}=0.10.
$$

动态残差：

$$
R_i^{\mathrm{dyn}}
=
\cos(\Delta z_i,\Delta z')
-
\gamma_{\mathrm{dyn}}
\left|
\log
\frac{\|\Delta z_i\|+\epsilon}{\|\Delta z'\|+\epsilon}
\right|,
\qquad
\gamma_{\mathrm{dyn}}=0.25.
$$

这里 $z_i$ 是候选 code，$z'$ 是 GT code，$\Delta z_i=z_i-z_t$，$\Delta z'=z'-z_t$。

### 2.3 Single-Step Loss

单步仍使用 vanilla GRPO：

$$
\mathcal{L}_{\mathrm{1step}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
A_i
\sum_{\tau}
\log\pi_\theta(o_{i,\tau}\mid q,o_{i,<\tau}).
$$

其中：

$$
A_i=
\frac{R_i^{\mathrm{RC}}-\mu_R}{\sigma_R+\epsilon}.
$$

单步不是本文最终算法创新的主战场；它证明 verifier 校准必要，并提供多步训练的 calibrated reward substrate。

## 3. Multi-Step Temporal GRPO

### 3.1 问题：Sequence-Level Advantage 太粗

multi-step rollout 生成 $F$ 个 future frames，每帧有一组 dynamics tokens：

$$
o_i=\{o_{i,t,\tau}:t=1,\ldots,F,\tau=1,\ldots,N\}.
$$

当前 lean multi-step GRPO 把所有帧的 reward 平均成一个 scalar：

$$
R_i=\frac{1}{F-1}\sum_{t=2}^{F} r_{i,t},
$$

再广播到整条 rollout：

$$
\mathcal{L}_{\mathrm{seq}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
A_i
\sum_{t=1}^{F}
\sum_{\tau=1}^{N}
\log\pi_\theta(o_{i,t,\tau}).
$$

这忽略了视频 rollout 的自条件结构：早期 token 会影响后续帧，后期错误也不应反向等价地惩罚所有 token。

### 3.2 Per-Frame Verifiable Reward

对每个候选、每个 future frame 计算 frame-level reward：

raw：

$$
r_{i,t}^{\mathrm{raw}}
=
-\left[
\mathrm{MSE}(\hat{s}_{i,t},s_t)
+
\mathrm{LPIPS}(\hat{s}_{i,t},s_t)
\right].
$$

RC：

$$
r_{i,t}^{\mathrm{RC}}
=
-\left[
\mathrm{MSE}(\hat{s}_{i,t},\tilde{s}_t)
+
\mathrm{LPIPS}(\hat{s}_{i,t},\tilde{s}_t)
\right],
\qquad
\tilde{s}_t=D(E(s_t)).
$$

沿用 RLVR-World multi-step convention：第一个 future frame 不作为直接 reward frame，因为它没有对应的 action-conditioned transition reward。但在 temporal-return 版本中，它仍可通过后续 rewards 获得信用。

### 3.3 Temporal Advantage

三种模式：

**seq baseline**：

$$
A_i^{\mathrm{seq}}
=
\frac{\bar r_i-\mu}{\sigma+\epsilon},
\qquad
\bar r_i=\frac{1}{F-1}\sum_{t=2}^{F}r_{i,t}.
$$

**frame advantage**：

$$
A_{i,t}^{\mathrm{frame}}
=
\frac{r_{i,t}-\mu_t}{\sigma_t+\epsilon}.
$$

第一个 future frame 的 direct frame advantage 置零。

**temporal-return advantage**：

$$
G_{i,t}
=
\sum_{u=\max(t,2)}^{F}
\beta^{u-t}r_{i,u},
$$

$$
A_{i,t}^{\mathrm{return}}
=
\frac{G_{i,t}-\mu_t}{\sigma_t+\epsilon}.
$$

其中 $\beta$ 是 temporal discount。return 模式允许早期生成 token 因后续 rollout 好坏获得 credit，更适合自回归视频世界模型。

### 3.4 Temporal GRPO Objective

Temporal GRPO loss：

$$
\mathcal{L}_{\mathrm{temp}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
\sum_{t=1}^{F}
A_{i,t}
\sum_{\tau=1}^{N}
\log\pi_\theta(o_{i,t,\tau}\mid h_{i,t,\tau}).
$$

当 `adv_temporal=seq` 时，退化为旧 multi-step GRPO。  
当 `adv_temporal=frame` 时，每个 frame 独立归一化。  
当 `adv_temporal=return` 时，用后续 reward 的折扣回报做 frame-block credit assignment。

### 3.5 Horizon-Aware KL

多步 v3 显示：模型早期学得好，但长训 final 漂移。为抑制自回归 rollout 后期漂移，引入 horizon-aware KL：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{temp}}
+
\lambda_{\mathrm{KL}}
\frac{1}{GFN}
\sum_{i,t,\tau}
w_t
\left[
\log\pi_\theta(o_{i,t,\tau})
-
\log\pi_{\mathrm{ref}}(o_{i,t,\tau})
\right].
$$

其中：

$$
w_t = 1 + \alpha_{\mathrm{horizon}}\frac{t-1}{F-1}.
$$

若 $\alpha_{\mathrm{horizon}}=0$，退化为普通均匀 KL；若 $\lambda_{\mathrm{KL}}=0$，关闭 KL。

注意：这里使用 sampled-token log-prob difference 作为 lightweight KL surrogate，与当前 harness 的实现风格一致。正式论文表述中应写作 KL regularization / reference-policy anchor，并在实现细节说明为 sampled-token estimate。

## 4. 负结果的定位

以下方法不再作为主方法：

| 类别 | 方法 | 当前判决 |
|---|---|---|
| scalar advantage | Dr.GRPO / hard floor-filter | 中性或更差 |
| reward fusion | dorw / codec_fused | 塌缩或伤保真 |
| single-step credit assignment | segmental / GP-SegGRPO | 无稳定收益 |
| objective | REAL-style rank-label VPO | 系统性变差 |
| optimization geometry | GSPO | 显著有害 |

它们作为 intervention-locus study：说明通用 GRPO 侧替换不能解决 tokenized video RLVR 的 verifier-floor 和 temporal-drift 问题。

## 5. 最小实验协议

### 5.1 Step-30 Multi-Step Sanity

不改算法，确认 early training 是否可稳定作为 final-at-30：

```bash
python scripts/train_grpo_msp.py \
  --rewards raw,rc \
  --seeds 0,1,2 \
  --steps 30 --K 16 --T 8 \
  --batch_windows 2 \
  --kl 0.001 \
  --eval_windows 8 \
  --eval_every 10 \
  --deterministic \
  --out_dir outputs/msp_pilot_step30
```

### 5.2 Temporal GRPO Pilot

三臂：

```bash
# old baseline
--adv_temporal seq --rewards raw,rc --seeds 0,1,2

# proposed
--adv_temporal return --rewards rc --seeds 0,1,2

# ablation
--adv_temporal frame --rewards rc --seeds 0,1,2
```

主判据：

- final LPIPS / LPIPS-last 不发散；
- final 优于 official multi-step RLVR held-out baseline；
- `return > seq`；
- `rc + return > raw + return`，证明 verifier calibration 与 temporal credit assignment 互补。
