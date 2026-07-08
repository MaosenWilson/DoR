# RC-GRPO — AAAI-2027 当前论文故事线

> **单一叙事事实源。** 方法细节见 `method.md`，实验状态见 `experiments.md`。日期型草稿统一放 `tmp/experiments/` 或 `tmp/notes/`，不再把一次性 md 散落在 `docs/experiments/`。

## Title

**RC-GRPO: Reconstruction-Calibrated Temporal Credit Assignment for Tokenized Video World Models**

中文暂译：**RC-GRPO：面向 Tokenized 视频世界模型的重建校准式时间信用分配**。

标题含义：

- **RC** = Reconstruction-Calibrated，指 single-step 诊断得到的 reachable-target / reconstruction-floor 校准。
- **GRPO** = 沿用 group-relative policy optimization 框架，而不是提出通用 RL 优化器。
- **Temporal Credit Assignment** = 多步视频 rollout 中 frame-block 级优势分配，是本文主方法贡献。
- **Tokenized Video World Models** = 限定适用场景，避免被审成通用 GRPO 论文。

## 0. 当前判决

论文主线从“提出更好的 reward”改为：

$$
\boxed{\text{single-step verifier diagnosis/calibration} \rightarrow \text{multi-step temporal GRPO}}
$$

单步不再作为主战场。它的角色是证明：tokenized video world model 的 post-decode verifier 会被 codec reconstruction floor 腐蚀，原始 RLVR reward 给 GRPO 的组内排序并不干净；RC reward calibration 是后续多步 GRPO 的校准输入。

多步才是主菜。当前 multi-step pilot 显示：100-step final 会漂移或发散，但固定 step30 协议已经能给出稳定 final 读数。`RC + seq GRPO` 在 3/3 seed 上优于 raw 与 held-out official RLVR；temporal credit assignment 进一步改善 final-at-30。5-seed 终局显示 plain temporal-return 最稳定，temporal-gain 对 `seq_rc` 的 0–2 seed 有正向信号，但 5-seed head-to-head 不如 plain return。最新消融显示 horizon-aware KL 单独中性/略差，叠加 return 也无额外收益，因此 KL 不再作为贡献。

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

Temporal-gain return 是下一步待验证候选，借鉴 stepwise/gain credit assignment 的思想：视频 rollout 不只关心每一帧是否好，也关心误差是否继续放大。定义相邻帧 improvement：

$$
g_{i,t}=r_{i,t}-r_{i,t-1},
$$

并构造 shaped frame reward：

$$
\tilde r_{i,t}
=
r_{i,t}+\alpha_{\mathrm{gain}}g_{i,t}.
$$

然后用同样的 temporal return：

$$
G^{\mathrm{gain}}_{i,t}
=
\sum_{u=t}^{T}\beta^{u-t}\tilde r_{i,u},
\qquad
A^{\mathrm{gain}}_{i,t}
=
\frac{G^{\mathrm{gain}}_{i,t}-\mu_t}{\sigma_t+\epsilon}.
$$

实现上第一帧直接 reward 仍按 RLVR-World multi-step convention 置零；gain 从有直接 reward 的帧开始计算，避免把人工零 reward 当作真实前一帧评价。

写作定位：这是本文真正的 GRPO 侧方法贡献，针对 tokenized multi-step video world model 的 temporal credit assignment。

### C2.1 与已有 GRPO credit-assignment 工作的关系

本文不应声称“第一个发现 GRPO 需要细粒度 credit assignment”。更稳的定位是：

> Existing GRPO variants have recognized that uniform sequence-level credit can be too coarse. We instantiate this principle in tokenized video world models, where the natural credit unit is a predicted future frame and the reward is a calibrated full-reference verifier.

可借鉴但不能照搬的相近方向：

- GRPO-$\lambda$ / eligibility-trace style work：关注 LLM reasoning 中 terminal reward 的 token-level credit assignment。
- Stepwise / flow-model GRPO：关注生成轨迹中不同 generation steps 的 credit assignment 或 improvement shaping。
- DAPO / token-level policy-gradient 系列：强调长序列 RL 中 token-level loss reduction 和采样效率。

我们的区别：

1. reward 是逐帧 full-reference verifier：
   $$
   r_{i,t}=R(\hat{s}_{i,t},s_t).
   $$
2. credit 单元是 video frame block，而不是 LLM token 或 diffusion denoising step：
   $$
   A_{i,t}\sum_{\tau\in\mathrm{frame}(t)}\log\pi_\theta(o_{i,t,\tau}).
   $$
3. temporal GRPO 之前先接入 single-step 已验证的 reachable-target calibration：
   $$
   \tilde{s}_t=D(E(s_t)).
   $$

因此写作时使用 “inspired by / adapts fine-grained GRPO credit assignment to tokenized video rollouts”，不要写 “first temporal credit assignment for GRPO”。

### C2.2 参考文献使用策略

参考文献的目标不是把所有 GRPO 变体都铺开，而是支撑三件事：

1. **RLVR-world-model 基座**：用 RLVR-World、GRPO/DeepSeekMath、iVideoGPT、FSQ、RT-1/Open X-Embodiment 说明本文沿用的是 tokenized video world model + verifiable reward + group-relative policy optimization 的既有框架。
2. **single-step diagnosis 的理论背景**：用 VQ-VAE/VQGAN/FSQ 和 perception-distortion tradeoff 支撑“tokenizer 往返重建误差是生成质量上界/地板”的直觉；用 LPIPS/SSIM/RAFT/FD-DINO/KID/PRDC 等只支撑评价指标，不把这些指标包装成方法贡献。
3. **multi-step temporal GRPO 的定位**：引用 GRPO-$\lambda$、SPO、SD-GRPO、Stepwise-Flow-GRPO、OAR 等说明 coarse sequence-level advantage 的 credit assignment 问题已经被不同领域观察到；我们的贡献是把这个原则落到 tokenized video rollout，信用单元是 future-frame block，并且 reward 先经过 reachable-target calibration。

写作时相关工作可以分三段：

- **World models and tokenized video prediction**：`ivideogpt`, `genie`, `dreamerv3`, `rt1`, `openxembodiment`。
- **RLVR / GRPO post-training**：`rlvrworld`, `grpo`, `deepseekr1`, `ppo`, `rloo`, `drgrpo`, `dapo`, `gspo`, `realrlvr`。
- **Fine-grained credit assignment**：`grpolambda`, `spo`, `sdgrpo`, `stepwiseflowgrpo`, `oar`, `noiseawaregrpo`。

审稿风险控制：

- 不把 REAL/GSPO/SAPO/CISPO 写成我们必须全面对比的强 baseline。它们是 LLM/RLVR 优化器背景；我们只在负结果或相关工作中说明“直接替换通用 GRPO 目标没有解决 video reward/temporal structure 的问题”。
- 不把 `flow` / `dmotion` 写成官方 RLVR metric。它们是我们为了动态一致性补充的 proxy；主表仍以 full-reference fidelity metrics 对齐 RLVR-World。
- 不把 FD-DINO/KID/Precision-Recall/Density-Coverage 写成 per-sample reward。它们是集合/分布级 evaluator，只能服务评测或附录分析。

### C3. Temporal stabilization evidence and failed generic stabilizers

Multi-step v3 的核心现象：

| readout | raw | RC | official RLVR |
|---|---:|---:|---:|
| final LPIPS | 0.3307 | 0.2797 | 0.2115 |
| best LPIPS | 0.1992 | 0.1955 | 0.2115 |

判读：

- step 20--30 已经学到有效预测。
- best checkpoint 3/3 seed 优于 official held-out RLVR。
- 100-step final 仍漂移或发散。

这说明多步问题不是“学不到”，而是“后期自回归漂移失控”。我们测试了 horizon-aware KL：

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

step30 fixed protocol 的最新结果：

| arm | final LPIPS | LPIPS-last | official final win |
|---|---:|---:|---:|
| seq raw | 0.2115 ± 0.0060 | 0.2309 ± 0.0196 | 2/3 |
| seq RC | 0.2009 ± 0.0042 | 0.2096 ± 0.0046 | 5/5 |
| return RC, uniform KL | 0.1980 ± 0.0034 | 0.2055 ± 0.0026 | 5/5 |
| gain-return RC, uniform KL | 0.1989 ± 0.0070 | 0.2072 ± 0.0068 | 5/5 |
| seq RC + horizon KL | 0.2005 ± 0.0059 | 0.2117 ± 0.0087 | 3/3 |
| return RC + horizon KL | 0.1977 ± 0.0012 | 0.2064 ± 0.0014 | 3/3 |

写作定位：horizon-aware KL 不是正贡献。真正保留的是 temporal credit assignment：它把 rollout-level scalar advantage 改成 frame-block return advantage。Plain temporal-return 是最终多步 GRPO 主线，final LPIPS 0.1980 ± 0.0034，5/5 优于 official；相对 `seq_rc` 的 5-seed 配对 final LPIPS $\Delta=-0.0029$、LPIPS-last $\Delta=-0.0041$，均为 4/5 wins。per-horizon 分析显示该优势从 h=2 的 -0.0016 增长到 h=7 的 -0.0041，并且在 MSE/MAE/PSNR/SSIM 上方向一致；但 `dmotion` 不支持动态改进，因此本文只主张多步 full-reference fidelity 改善，不主张动态指标提升。

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
- 多步 under held-out protocol 的 step30 final 明显优于 official RLVR；100-step 长训仍存在 drift limitation。

## 3. 下一步实验顺序

### P0. 固定早停多步 sanity check

已完成，step30 final 稳定。历史命令：

```bash
steps=30, eval_every=10, kl=0.001, raw/rc × seeds 0,1,2
```

读数：final-at-30，不再用 best checkpoint。判决：`seq RC` final LPIPS 0.1997 ± 0.0051，3/3 seed 优于 official RLVR。

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

当前结果：

- `return_hkl00` 相比 `seq RC`：final LPIPS 均值 $\Delta=-0.0029$，4/5 wins；LPIPS-last $\Delta=-0.0041$，4/5 wins。
- per-horizon full-reference gains grow with horizon: LPIPS h=2:-0.0016 → h=7:-0.0041; MSE h=2:+0.00001 → h=7:-0.00057; PSNR h=2:+0.044 → h=7:+0.235; SSIM h=2:+0.0012 → h=7:+0.0050.
- `dmotion` is not improved (aggregated $\Delta=-0.00111$, 14/30 wins), so dynamic improvement is not claimed.
- `gain_return` 相比 `seq RC` on shared seeds 0–2：final LPIPS $\Delta=-0.0036$，3/3 wins；LPIPS-last $\Delta=-0.0046$，3/3 wins。
- `gain_return` 相比 `return_hkl00` on seeds 0–4：final LPIPS $\Delta=+0.0009$，1/5 wins；LPIPS-last $\Delta=+0.0017$，2/5 wins。不能升级为最终冠军。
- `seq_hkl05` 相比 `seq RC`：final LPIPS $\Delta=+0.0008$，1/3 wins，horizon KL 单独不成立。
- `return_hkl05` 相比 `return_hkl00`：final LPIPS $\Delta=+0.0008$，1/3 wins，叠加 KL 也无收益。
- n=3 下不能作显著性主张。
- `gain_return` 是正向但不稳定候选，当前主线保留 plain temporal-return 作为最稳定配置。

后续若要把 C2 写成强贡献，判据：

- final 不发散。
- final LPIPS / LPIPS-last 优于 official RLVR。
- `gain_return >= return > seq`，或至少 `return > seq`。
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

## 5. 审稿人风险与下一轮设计

### R1. “Reward calibration 只是 reward engineering”

应对：C1+C2 合并成一个贡献，不拆开讲。单步部分只承担 verifier diagnosis：

$$
\text{codec floor}\rightarrow \text{rank flip}\rightarrow \text{reachable-target calibration}.
$$

它不是主菜；主菜放在 multi-step temporal GRPO。

### R2. “Temporal-return GRPO 是不是只因为 early stopping”

当前 step30 protocol 是合理但敏感的。必须明确：

- 100-step long training 暴露 drift limitation；
- step30 是固定、预先锁定的 final-at-step readout，不使用 best checkpoint；
- 所有比较使用相同 held-out windows 和相同 step30。

下一轮最关键的补强不是继续调 step，而是给出 longer-horizon / longer-training stability 证据。

### R3. “Temporal credit assignment 的归因还不够”

当前已证：

- `seq RC` 优于 raw，说明 verifier calibration transfers to multi-step；
- `return RC` 5/5 优于 official，且比 gain 更稳；
- gain/horizon-KL 不支持升级为主方法。

缺口：

- `seq_rc` 只有 seeds 0--2；若要对 `return_hkl00` 做 5-seed 配对显著性，需补 `seq_rc` seeds 3,4。
- 缺少 per-horizon 读数表，不能证明 temporal-return 特别改善后期帧。

### R4. “只看 LPIPS/MSE，不够视频世界模型”

必须补至少一个 temporal / distribution readout：

- per-horizon LPIPS/MSE 曲线：$h=2,\ldots,T$；
- final-frame LPIPS-last 已有，但需要整理成曲线；
- 若时间允许，补 FD-DINO/KID 或 flow/dmotion on multi-step final checkpoints。

### 下一轮最小实验

P0：补 `seq_rc` seeds 3,4，形成 `seq_rc` vs `return_hkl00` 的 5-seed 配对。

P1：离线分析已有 checkpoints 的 per-horizon metrics，不训练。目标是证明 return GRPO 的收益集中在中后段 horizon，而不是随机均值波动。

P2：若 P0/P1 成立，停止方法搜索，进入写作和图表阶段。若 P0 不成立，则 temporal-return 降级为 stability refinement，主线改成 RC multi-step transfer + limitations。
