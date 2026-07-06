# Temporal RC-GRPO — AAAI-2027 当前论文故事线

> **单一叙事事实源。** 方法细节见 `method.md`，实验状态见 `experiments.md`。日期型草稿统一放 `tmp/experiments/` 或 `tmp/notes/`，不再把一次性 md 散落在 `docs/experiments/`。

## 0. 当前判决

论文主线从“提出更好的 reward”改为：

$$
\boxed{\text{single-step verifier diagnosis/calibration} \rightarrow \text{multi-step temporal GRPO}}
$$

单步不再作为主战场。它的角色是证明：tokenized video world model 的 post-decode verifier 会被 codec reconstruction floor 腐蚀，原始 RLVR reward 给 GRPO 的组内排序并不干净；RC reward calibration 是后续多步 GRPO 的校准输入。

多步才是主菜。当前 multi-step pilot 已经显示：模型在 step 20--30 可以学到明显优于 held-out official RLVR checkpoint 的预测，但 100-step final 会漂移或发散。因此真正需要创新的地方不是继续换通用 GRPO 变体，而是针对视频自回归 rollout 做 temporal credit assignment 和 horizon-aware drift control。

已判负或降级的通用 GRPO 侧方案：

- Dr.GRPO / hard floor-filter：中性或变差。
- Segmental / GP-SegGRPO：单步干净 residual 设计下无稳定收益。
- Rank-label REAL-style VPO：5 seed 系统性低于 baseline。
- GSPO：三臂同 sweep 中在 RC reward 上显著有害，flow $\Delta=-0.021,t=-2.96,0/5$，保真全线变差。

因此本文不再讲“我们找到一个更好的通用 GRPO 变体”。新的可写结论是：

> In tokenized video RLVR, single-step experiments identify verifier-side rank corruption, while multi-step rollouts expose a temporal credit-assignment and drift-control problem. Effective improvement must be video-structured: calibrated verifiable rewards plus temporal GRPO for frame-block credit assignment.

## 1. 贡献重排

### C1. Verifier floor diagnosis + RC calibration

这条把原来的“地板理论”和“reward 设计”合并成一个贡献，避免看起来像两条 reward engineering。

定义 codec reconstruction floor：

$$
\phi_{\mathrm{tok}}^{(d)}
=
\mathbb E[d(D(E(s')),s')].
$$

它是 tokenizer 往返造成的不可约残差，住在 decoder 后，不是预测误差本身。

核心机制：

$$
P_{\mathrm{flip}}
=
\frac{\arccos(\rho)}{\pi},
$$

其中 $\rho$ 是含地板 reward 与无地板参照之间的组内相关。已有证据：

- LPIPS 上实测翻转率约 0.185，理论约 0.186。
- 最新离线分析扩展到 pixel / MSE / faithful reward，920 windows × 16 candidates：
  - pixel: $\rho=0.758$，实测 flip 0.186，理论 0.186。
  - MSE: $\rho=0.770$，实测 flip 0.181，理论 0.184。
  - a0faithful: $\rho=0.766$，实测 flip 0.182，理论 0.181。

因此旧说法“增益与地板绝对幅度成正比”已经废弃。正确解释是：增益由组内排序腐蚀程度 $\rho$ / flip rate 决定。

RC reward calibration 把 verifier target 从真实帧 $s'$ 改成 tokenizer 可达目标：

$$
\tilde{s}' = D(E(s')).
$$

单步主 reward：

$$
R_i^{\mathrm{RC}}
=
z_G\big(-\mathrm{LPIPS}(\hat{s}'_i,\tilde{s}')\big)
+
0.10\,z_G(R_i^{\mathrm{dyn}}),
$$

动态残差：

$$
R_i^{\mathrm{dyn}}
=
\cos(\Delta z_i,\Delta z')
-0.25
\left|
\log
\frac{\|\Delta z_i\|+\epsilon}{\|\Delta z'\|+\epsilon}
\right|.
$$

已经验证的单步结论：

- `pixel -> pixel_tok` 稳定修复 post-decode floor 带来的目标错位。
- `pixel_tok_dyn` 是当前 fidelity-motion Pareto 点。
- RC 显著减少 raw-GT perceptual reward 的灾难性发散。

写作定位：这不是论文主菜，而是后续 temporal GRPO 的 verifier calibration substrate。

### C2. Temporal GRPO for multi-step video rollouts

多步视频预测不是一条普通序列。当前 GRPO 把整个 rollout 压成一个 scalar reward / scalar advantage：

$$
\mathcal L_{\mathrm{seq}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
A_i
\sum_{t=1}^{T}
\sum_{\tau\in\mathrm{frame}(t)}
\log \pi_\theta(o_{i,t,\tau}).
$$

这会把第 2 帧、第 5 帧、第 8 帧的错误混到同一个优势里。视频 world model 是自条件的：

$$
\hat{s}_{t+1}\rightarrow \hat{s}_{t+2}\rightarrow \cdots \rightarrow \hat{s}_{t+T},
$$

早期小错误会通过后续自回归历史放大。因此 multi-step RLVR 需要 frame-block 级信用分配。

Temporal GRPO 将优势单位从 rollout 改为 future frame：

$$
\mathcal L_{\mathrm{temp}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
\sum_{t=1}^{T}
A_{i,t}
\sum_{\tau\in\mathrm{frame}(t)}
\log \pi_\theta(o_{i,t,\tau}).
$$

两种候选优势：

帧独立：

$$
A_{i,t}
=
\frac{r_{i,t}-\mu_t}{\sigma_t+\epsilon}.
$$

Temporal return：

$$
G_{i,t}
=
\sum_{u=t}^{T}\beta^{u-t}r_{i,u},
\qquad
A_{i,t}
=
\frac{G_{i,t}-\mu_t}{\sigma_t+\epsilon}.
$$

其中第一帧可没有直接 action-conditioned reward，但它会影响后续 rollout，因此在 return 版本里仍可通过未来 reward 得到信用。

写作定位：这是本文真正的 GRPO 侧方法贡献，针对 tokenized multi-step video world model 的 temporal credit assignment。

### C3. Horizon-aware KL for autoregressive drift control

Multi-step v3 的核心现象：

| readout | raw | RC | official RLVR |
|---|---:|---:|---:|
| final LPIPS | 0.3307 | 0.2797 | 0.2115 |
| best LPIPS | 0.1992 | 0.1955 | 0.2115 |

判读：

- step 20--30 已经学到有效预测。
- best checkpoint 3/3 seed 优于 official held-out RLVR。
- 100-step final 仍漂移或发散。

这说明多步问题不是“学不到”，而是“后期自回归漂移失控”。因此 KL 不能只作为普通稳定项，而应随 horizon 加权：

$$
\mathcal L
=
\mathcal L_{\mathrm{temp}}
+
\lambda_{\mathrm{KL}}
\sum_{t=1}^{T}
w_t
\mathrm{KL}\big(
\pi_\theta(\cdot|h_t)
\Vert
\pi_{\mathrm{ref}}(\cdot|h_t)
\big),
$$

其中：

$$
w_t = 1 + \alpha\frac{t-1}{T-1}.
$$

写作定位：KL 本身不是新东西，创新点是 horizon-aware drift control 被用于 tokenized video rollout 的后期漂移问题，并由 v3 的发散曲线直接动机化。

## 2. 当前不能主张的内容

不能写：

- code reward 更好。
- feature reward 更好。
- 我们提出了更好的通用 GRPO。
- GSPO / REAL / segmental 只是没调好。
- 增益与地板绝对幅度成正比。
- 多步完整训练设置击败官方 RLVR。

可以写：

- 单步实验揭示 verifier-side rank corruption，并验证 RC 能修复目标错位。
- 原始 vanilla GRPO 在单步上已经是强基座，继续换通用优化器无稳定收益。
- 多步 under held-out protocol 的 early checkpoint 明显优于 official RLVR，但 final 稳定性需要 temporal credit assignment 和 horizon-aware KL。

## 3. 下一步实验顺序

### P0. 固定早停多步 sanity check

先不改算法，确认 step 30 final 是否稳定：

```bash
steps=30, eval_every=10, kl=0.001, raw/rc × seeds 0,1,2
```

读数：final-at-30，不再用 best checkpoint。

### P1. Temporal GRPO pilot

在 `scripts/train_grpo_msp.py` 中新增：

- `--adv_temporal seq|frame|return`
- `--temporal_gamma`
- `--horizon_kl_alpha`

最小三臂：

```bash
# baseline
--adv_temporal seq --rewards raw,rc --seeds 0,1,2

# method
--adv_temporal return --rewards rc --seeds 0,1,2

# ablation
--adv_temporal frame --rewards rc --seeds 0,1,2
```

判据：

- final 不发散。
- final LPIPS / LPIPS-last 优于 official RLVR。
- `return > seq`，最好 `return > frame > seq`。
- `rc + return` 优于 `raw + return`，说明 RC 是必要 substrate。

### P2. 若 P1 成功

论文主结果改为：

1. single-step verifier diagnosis；
2. multi-step temporal GRPO final improvement；
3. intervention-locus negative results 解释为什么通用 GRPO 变体无效。

### P3. 若 P1 失败

多步降级为 limitation，主线回到：

1. verifier floor mechanism；
2. RC calibration；
3. GRPO-side systematic negative evidence。

但这种形态的顶会风险会明显更高。

## 4. 后续工作流程规则

1. 任何新实验或代码改动：先改总文档，再改代码。
2. 日期型草稿和一次性分析放 `tmp/experiments/` 或 `tmp/notes/`。
3. `docs/aaai2027/story.md`、`docs/aaai2027/method.md`、`docs/aaai2027/experiments.md` 只保留当前事实源。
4. 每次代码改动后必须做 code review：
   - 看 diff 是否只改目标区域。
   - 做 `py_compile` 或轻量 smoke。
   - 写清楚是否同步服务器。
