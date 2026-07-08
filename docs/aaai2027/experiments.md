# Experiments — 当前实验状态与判据（2026-07-06）

> **当前事实源更新**：07-01 的 CAST-GRPO 实验矩阵已经过时。保留下方旧矩阵作为历史索引，但当前论文主线以 `story.md` 为准：**single-step verifier diagnosis / RC calibration + multi-step temporal GRPO**。已跑过的通用 GRPO 侧替换(segmental / REAL / GSPO 等)均已判负或降级。

## Current Summary

| 模块 | 状态 | 判决 |
|---|---|---|
| 单步 verifier floor + RC | 已验证 | 作为诊断与校准 substrate；不再单独包装成主菜 |
| 离线 $\rho$/flip 诊断 | 已有首批结果 | pixel/MSE/a0faithful 的 flip 理论与实测吻合，解释 `mse_tok` 反常增益 |
| GRPO 侧改造 | 已系统性判负 | Dr.GRPO/filter/segmental/REAL/GSPO 均无稳定正收益 |
| 多步 MSP v3 (`kl=0.001`) | 完成 3 seed | best checkpoint 强，但 final checkpoint 不稳；动机化 temporal GRPO |
| Multi-step step30 seq | 已完成 | RC + seq final-at-30 稳定优于 raw 与 official RLVR；`seq_rc` 已补到 5 seed |
| Temporal-return GRPO | 已完成 5 seed | 最终主线：`return_hkl00` final LPIPS 0.1980 ± 0.0034，5/5 优于 official |
| Temporal-gain GRPO | 已完成 5 seed | 对 `seq_rc` 的 0–2 seed 正向，但 5-seed 下不如 plain return；降级为正向但不稳定消融 |
| Horizon-aware KL | 已完成 3 seed 消融 | 单独中性/略差，叠加 return 也无额外收益；降级为负/中性消融 |

## Multi-step MSP v3 Verdict

协议：`T=8,K=16,steps=100,batch_windows=2,kl=0.001,deterministic,raw/rc × seeds 0,1,2`。

Eval-only baselines:

| checkpoint | LPIPS | LPIPS-last | MSE |
|---|---:|---:|---:|
| base | 0.2157 | 0.2260 | 0.01484 |
| official multi-step RLVR | 0.2115 | 0.2155 | 0.01378 |

Summary:

| arm | final LPIPS | best LPIPS | tail-3 LPIPS | final MSE | best MSE |
|---|---:|---:|---:|---:|---:|
| raw | 0.3307 ± 0.1224 | 0.1992 ± 0.0059 | 0.3361 ± 0.0417 | 0.02250 ± 0.01121 | 0.01251 ± 0.00105 |
| rc | 0.2797 ± 0.1348 | 0.1955 ± 0.0074 | 0.2653 ± 0.0732 | 0.01841 ± 0.01143 | 0.01185 ± 0.00059 |

Verdict:

- Positive signal: both raw and RC reach best checkpoints below official RLVR in all 3 seeds.
- Negative/stability: final checkpoints are unstable; `kl=0.001` does not fully prevent policy drift.
- Conservative use: multi-step can support “early learning signal transfers,” but not a final headline unless a fixed early-stop protocol passes.

P0 step30 result:

| arm | final LPIPS | LPIPS-last | MSE | official final win |
|---|---:|---:|---:|---:|
| seq raw | 0.2115 ± 0.0060 | 0.2309 ± 0.0196 | 0.01431 ± 0.00056 | 2/3 |
| seq RC | 0.1997 ± 0.0051 | 0.2095 ± 0.0059 | 0.01279 ± 0.00082 | 3/3 |
| return RC, uniform KL | 0.1980 ± 0.0034 | 0.2055 ± 0.0026 | 0.01276 ± 0.00069 | 5/5 |
| gain-return RC, uniform KL | 0.1989 ± 0.0070 | 0.2072 ± 0.0068 | 0.01271 ± 0.00101 | 5/5 |
| seq RC + horizon KL | 0.2005 ± 0.0059 | 0.2117 ± 0.0087 | 0.01297 ± 0.00095 | 3/3 |
| return RC + horizon KL | 0.1977 ± 0.0012 | 0.2064 ± 0.0014 | 0.01262 ± 0.00007 | 3/3 |

Paired deltas:

- `seq_rc - seq_raw`: final LPIPS $\Delta=-0.0118$, 3/3 wins; LPIPS-last $\Delta=-0.0215$, 3/3 wins.
- `return_hkl00 - seq_rc`: final LPIPS $\Delta=-0.0029$, 4/5 wins; LPIPS-last $\Delta=-0.0041$, 4/5 wins. This is the final 5-seed evidence for temporal-return GRPO.
- `gain_a05 - seq_rc` on shared seeds 0–2: final LPIPS $\Delta=-0.0036$, 3/3 wins; LPIPS-last $\Delta=-0.0046$, 3/3 wins.
- `gain_a05 - return_hkl00` on seeds 0–4: final LPIPS $\Delta=+0.0009$, 1/5 wins; LPIPS-last $\Delta=+0.0017$, 2/5 wins. Gain-return is not better than plain temporal-return and has larger variance.
- `seq_hkl05 - seq_rc`: final LPIPS $\Delta=+0.0008$, 1/3 wins; horizon-aware KL alone is neutral/slightly worse.
- `return_hkl05 - return_hkl00`: final LPIPS $\Delta=+0.0008$, 1/3 wins; horizon-aware KL does not improve temporal-return.

Verdict:

- Step30 fixed protocol passes: multi-step no longer只能用 best checkpoint，final-at-30 已可作为读数。
- Strong result is RC vs raw under the same seq GRPO: this proves verifier calibration transfers to multi-step.
- Temporal-return GRPO is the final GRPO-side mainline: it is more stable than gain-return and 5/5 beats official held-out RLVR.
- Temporal-gain GRPO gives a positive signal versus `seq_rc` on seeds 0–2 but fails the 5-seed head-to-head against plain temporal-return; report it as a tested but unstable variant, not as the main method.
- Horizon-aware KL is not supported by the ablation. It should be written as a tested-but-unhelpful stabilization attempt, not as a contribution.

Immediate missing evidence:

- Completed: `seq_rc` seeds 3,4 have been added; `return_hkl00` vs `seq_rc` is now a 5-seed paired comparison.
- Completed: per-horizon evaluation confirms return GRPO's advantage grows with horizon.

Per-horizon readout (`return_rc - seq_rc`, negative is better):

| horizon | LPIPS delta | MSE delta | PSNR delta | SSIM delta |
|---:|---:|---:|---:|---:|
| 2 | -0.00160 | +0.00001 | +0.04396 | +0.00120 |
| 3 | -0.00221 | -0.00018 | +0.11132 | +0.00284 |
| 4 | -0.00241 | -0.00025 | +0.15766 | +0.00337 |
| 5 | -0.00341 | -0.00040 | +0.20745 | +0.00520 |
| 6 | -0.00354 | -0.00046 | +0.23973 | +0.00421 |
| 7 | -0.00408 | -0.00057 | +0.23454 | +0.00502 |

Aggregated over seed-horizon cells (`return_rc - seq_rc`):

- LPIPS: $\Delta=-0.00287$, 22/30 wins.
- MSE: $\Delta=-0.00031$, 20/30 wins.
- MAE: $\Delta=-0.00139$, 20/30 wins.
- PSNR: $\Delta=+0.16578$, 21/30 wins.
- SSIM: $\Delta=+0.00364$, 21/30 wins.
- dmotion: $\Delta=-0.00111$, 14/30 wins, not supportive.

Verdict: temporal-return GRPO's advantage is not a single-LPIPS artifact; it is consistent across full-reference fidelity metrics (LPIPS/MSE/MAE/PSNR/SSIM), and it grows toward later rollout frames. It does **not** improve the cheap dynamic-direction proxy `dmotion`, so dynamic improvement should not be claimed from this experiment.

Historical recommended clean follow-up P0:

```bash
steps=30, eval_every=10, kl=0.001, raw/rc × seeds 0,1,2
```

Readout must be final-at-30, not best checkpoint.

Recommended clean follow-up P1:

```bash
--adv_temporal seq --rewards raw,rc --seeds 0,1,2
--adv_temporal return --rewards rc --seeds 0,1,2
--adv_temporal gain_return --gain_alpha 0.5 --rewards rc --seeds 0,1,2
--adv_temporal frame --rewards rc --seeds 0,1,2
```

Main readout: final LPIPS / LPIPS-last stability, not best checkpoint.

---

# Historical Matrix — 矩阵、状态与判据（重构版，2026-07-01）

> 叙事 `story.md`,方法 `method.md`。已证数字详见 `reward_summary_20260701.md`、`results_v1v6.md`、`docs/experiments/exp_*.md`(运行记录,勿删,但不等于都要写进 Results)。
> 全部在 `dor` harness + 5090(thuml 现成基座,只做 GRPO 微调);多步用现成多步基座,不碰 verl/vllm。

---

## 0. Results 蓝图(按论文要展示什么排,不按跑的先后;2026-07-01 CAST-GRPO 转正后重排)

CAST-GRPO 变主贡献后,一批曾经的"头牌结果"降级成背景一句话,不再需要独立图表。**下表是从最终论文 Results section 倒推排出来的优先级**,不是执行日志。

| 优先级 | Results 条目 | 承载图表 | 支撑贡献 | 状态 |
|---|---|---|---|---|
| **P0** | **段级 vs 全局,最优配置,配对多 seed 显著** | **Table1(全文核心)** | A(主) | 🔴 阻塞在 §3.0 全局锚点未验证 |
| P0 | λ/K/γ 消融 + 全部退化检查通过 | Fig3 | A(主) | 🟡 部分完成(见 §3.1) |
| P1 | 翻转率 = $\arccos(\rho)/\pi$(度量无关,机理最硬证据) | Fig2 | B(地基) | ✅ 已完成,无需再补 |
| P1 | 段级信用 on 裸 code(负对照:弱信号上无效/负) | Fig3 附属 panel(非独立图) | A(佐证 A 的前提) | ✅ 已完成 |
| P2 | reward 校准三件套:pixel→pixel_tok(+14%)、mse_tok 负对照、dyn_lambda 扫→λ=0.10 | Fig4(合并成一张,不铺开) | C(配菜) | ✅ 已完成 |
| P2 | 绿灯 seg 配置的分布面板(FD-DINOv2/KID/PRDC,确认段级信用不伤多样性) | Fig6 | A 的补充证据 | ⬜ 待 P0 出赢家后补跑 |
| P3(可选/附录) | 全局 GRPO 花招失败(Dr.GRPO/floor_filter v3) | 正文一句话 + 附录小表 | 动机(Step2 的"负结果变动机") | ✅ 已完成,**不再单独出图** |
| P3(可选/附录) | 地板跨度量特征(C1 旁证) | 附录小表 | B 的旁证 | ✅ 已完成,**已从主证据降级** |
| P4(future work,若时间不够可砍) | 多步时空段级 GRPO | Fig5 | A(扩展) | ⬜ 未开始 |

### 已明确砍掉 / 不再进 Results 的(跑过,但现在不必要)

- **A0–A6 全臂多 seed 扫描表**(grpo_full 的完整版本):降为 P2 里"我们筛过候选"的一句话背景,**不铺开成表**。
- **SSIM 臂跨 seed 发散**:曾支撑已废弃的"code 更稳"主张,**彻底不用**。
- **hybrid/dorw 融合负结果**:降为 P3 一句话("naive 融合坍缩回单臂,无增益"),不单独出图。
- **确定性/GPU 非确定性排查**:不是 Results,是 **Methods/Limitations 的一条方法论说明**(为什么全程 `--deterministic` + 配对协议),不占实验篇幅。
- **旧奖励臂(pixel/code/mse/pixel_tok/mse_tok)的分布面板**:被 P2 的"测赢家 ckpt"取代,旧版本降为可选附录,不必补跑 Inception 空间剩余部分。
- **rank-reliability 独立加权(rankrel)作为第四贡献**:取消独立地位,**并入** method.md §3.2 的 $w_{g,m}$ 阻尼公式讨论,只在 P0/P1 有余力时才作为一个额外消融点,不单开实验线。

**下面 §1 起的详细矩阵按此优先级理解:只有 P0/P1/P2 是当前必须做完的,P3 已完成且封存,P4 是延伸。**

---

## 0.5 全流程清单(按单步/多步分,勾选进度)

> **当前在跑(2026-07-01)**:`outputs/segpd_2x2_l1.0`(pixtok_dyn substrate,K=4,λ=1.0,纯段级不混全局),3/5 seed 完成,PID 存活中。

### 单步(single-step)

**B. 地基机理**
- [x] B1 地板存在 + 跨度量特征刻画(感知度量地板>信号,MSE 相反)
- [x] B2 翻转率 $\arccos(\rho)/\pi$ 验证(5 seed,实测 0.185 vs 理论 0.186)

**C. Reward substrate 校准(已冻结为 `pixel_tok_dyn`,不再扩)**
- [x] C1 全臂扫描(A0-A6,定位候选;已降级为背景一句话,不铺表)
- [x] C2 floor-cancellation 主测:pixel vs pixel_tok(+14% flow,5/5 seed)
- [x] C3 mse_tok 负对照(floor-cancel 增益 ∝ 地板)
- [x] C4 dyn_lambda 扫描 {0.05,0.10,0.25} → 选定 $\lambda_{dyn}=0.10,\gamma_{dyn}=0.25$
- [x] C5 全局 GRPO 花招失败(Dr.GRPO 中性、floor_filter v3 变差)→ 段级信用的动机
- [x] C6a 分布面板 DINOv2 空间(旧奖励臂,5/7 完成;已降级为可选附录)
- [ ] C6b 分布面板 Inception 空间(旧奖励臂;**不再补,已砍**)

**A. 单步空间段级 GRPO(主实验)**

负对照(裸 code,验证段级信用需要信息丰富的 reward):
- [x] A1a seg(code) 2x2, λ=0.0(5 seed)
- [x] A1b seg(code) 2x2, λ=0.7(5 seed,flow 反而变差,负对照成立)
- [x] A1c seg(code) 2x2, λ=1.0(2 seed,方向仍负)

遗留路径(`pixtok_dyn` substrate,**2026-07-01 起非 CAST-GRPO 主线,已被 `codec_fused` 取代,保留数据供对比,不再补新配置**)K=4:
- [x] A2a segpd 2x2, λ=0.0(5 seed,内部基线)
- [x] A2b segpd 2x2, λ=0.7(5 seed,flow +0.054 t=+2.35,dmot +0.028 t=+2.69,保真不显著变)
- [ ] A2c segpd 2x2, λ=1.0(**4/5 seed,快跑完**;不再补齐第 5 个的紧迫性已降低,因主线已转 codec_fused)
- [ ] A2d/e/f/g(λ=0.3、4x4 各档)——**冻结,不再补跑**,除非 `codec_fused` 效果不如它需要回退对比

**主线(`codec_fused` substrate,method.md §3.2,2026-07-01 重构:段级在线 Wiener 权重,零自由超参,取代已删除的 $\exp(-\alpha\tilde b)$ 启发式)**:
- [x] A9a v1($\exp(-\alpha\tilde b)$ 启发式)——**废案①**,无推导+自由超参
- [x] A9b v2(方差分解 Wiener,$\hat\sigma^{*2}/(\hat\sigma^{*2}+b^2)$)——**废案②**:把地板**均值**当噪声标准差;组内常数部分本会被 GRPO 归一化减掉(C2 自己的论点),真噪声是候选间波动 $\sigma_\eta\ll b$
- [x] A9c v2 阶段一 pilot(segcf_1x1_anchor / 2x2_l0.0 / 2x2_l0.7,各 5 seed)——**已跑但按"段级 code reward"解读归档**:真实数据 q 探针实测 **q≡0**(K=1/4/16 全部),v2 实际训的是纯 code;anchor flow 0.2712≈历史全局 code 0.2751 交叉印证;此发现同时回溯解释了 `dorw` 当年塌缩成 code 的根因
- [x] A9d v3 代码重写:$q_{h,k}=[\max(0,\rho_{h,k})]^2$ + 段级动态残差项
- [x] A9e v3 sanity(合成 q=0.884≈Wiener 0.917)+ 真实数据 q 探针(K=1 q≈.85 / K=4 q≈.56 / K=16 q≈.41,不再塌 0)
- [x] A10 v3/v3.1 冒烟(v3.1 = r_dec 对齐可达目标,恢复 C3b 技巧)
- [x] A9f 修 `analyze_seg.py` 正则
- [x] **A11 v3.1 阶段一 pilot 已跑完并判决(2026-07-03,segcf31_*,各 5 seed)——未过绿灯**:λ=0.7 vs λ=0 配对:flow +0.006(3/5,t=0.50,噪声)、dmot +0.014(4/5,t=2.47,勉强)、**保真三项一致变差**(LPIPS t=−2.02/PSNR t=−2.21/SSIM t=−1.92)。v3.1 的目标对齐把 v3 的保真惩罚减半(t 从 −4 级降到 −2 级)但不足以翻盘。**按赛前判决规则执行:codec_fused 停止迭代,降级为诚实消融;主方法回退 segpd(pixel_tok_dyn 段级)**——它仍是唯一过绿灯的配置(flow +0.054 t=2.35、dmot +0.028 t=2.69,保真 |t|<1)。
- [x] A13 诊断确认(2026-07-03,外部分析 + 人工复核):段池化基线(segpd λ=0 flow 0.2142)远低于真全局(grpo_v9_lam010 flow 0.2871)——**旧信用结构的两个缺陷**:①"全局"是段池化伪全局,损伤 reward;②混合公式按 (1−λ) 缩减全局信号(λ=0.7 时只剩 30%)。→ **信用结构重构为 v4 residual 式(method.md §3.4)**

**当前主线(v4 GP-SegGRPO:Global-Preserving Segment Residual,2026-07-03 起)**
$$\hat A_{i,k}=A^{\text{global}}_i+\lambda\,\Delta A^{\text{seg}}_{i,k},\quad \Delta A^{\text{seg}}\text{ 零均值残差},\ \gamma=0\text{(阶段一)}$$
- [ ] G1 代码实现:`--adv_estimator gpseg`;全局项与标准 grpo 路径逐表达式相同;损失分解 $\mathcal L=\mathcal L_{\text{global}}+\lambda\mathcal L_{\text{residual}}$
- [ ] G2 Sanity(离线,CPU):①残差逐候选均值 ≡0;②K=1 ⇒ 残差 ≡0;③λ=0 ⇒ 损失表达式与全局路径一致
- [ ] G3 GPU 冒烟(λ=0.3,6 步)+ K=1 λ=0.7 冒烟(应≈全局)
- [x] G4 主实验已跑完(2026-07-04,westd,5 臂 × 8 seed 同 sweep 配对)
- [x] **G5 判决:红灯(2026-07-04)**。任何 λ 都没有配对显著超过真全局:λ=0.1 flow 反而变差(t=−1.84,1/8);λ=0.3/0.5/0.7 全部指标 |t|<1.4(纯噪声;dmot 在 λ≥0.3 有微弱一致正向 t=0.8-1.3、wins 5-6/8,但远不显著)。真全局 A0 flow=0.2745±0.0212(8 seed),与历史 grpo_v9(0.2871±0.0153,5 seed)跨批一致。**A13 假设被证实:旧 segpd 的"绿灯"确实只是弥补段池化损伤;在保留 100% 全局信号的干净 residual 设计下,段级再分配没有可测收益。** 按赛前 G5 规则执行:**段级信用分配(任何粒度、任何结构)降级为系统性负结果;论文主线回退 B+C。**
- ~~G6~~(已取消,G5 红灯)

**G5 红灯后的证据全景(GRPO 侧干预空间已系统性证伪,三个层级、各有干净对照)**:
| 层级 | 尝试 | 结果 |
|---|---|---|
| 全局标量 | Dr.GRPO 去偏 / 硬过滤地板组(F2) | 中性 / 变差 |
| 段级替代式 | segpd 混合公式(v1-v3.1) | "增益"实为弥补段池化自伤(A13) |
| 段级残差式 | GP-SegGRPO(v4,干净设计+8 seed 配对) | 与真全局无差(本判决) |

→ **正面结论反而更硬了:所有真实增益都来自 reward 侧校准(floor-cancel +14%、dyn 残差 Pareto),credit assignment 粒度不是瓶颈。**"在哪里修 reward 有效、在哪里修 GRPO 无效"本身是一个系统性的、有对照的发现。

**明确不再做**(外部分析 §11 建议,采纳):救 codec_fused / 加新 reward / SD-GRPO 替换 / 补旧 segpd 配置 / 跨批次引用历史全局数据当基线。

---

## 终局收口计划(2026-07-04,RC-GRPO 定稿后;取代上面所有未完成计划)

### P0a — Table 1 同 sweep 复测(单步收口的唯一缺口,~28h GPU)

**动机**:现有主表 arm 来自 ≥5 个不同批次(v1/v5/v7/v9/g4)且跨服务器迁移;自家方法论红线=跨批 flow 噪声 ±0.03-0.08。一次同 sweep 复测彻底关掉这个攻击面。

```bash
cd /root/autodl-tmp/vote2world; P=/root/miniconda3/bin/python
$P scripts/train_grpo.py \
   --rewards pixel,a0faithful,mse,mse_tok,pixel_tok,pixel_tok_dyn,code \
   --modes gt_only --seeds 0,1,2,3,4,5,6,7 \
   --dyn_lambda 0.10 --dyn_gamma 0.25 \
   --steps 150 --K 16 --deterministic --out_dir outputs/table1_final
```
(7 arm × 8 seed,sweep 模式自带断点续跑;base 行数字从各 run 的 step-0 eval 提取,无需单独跑。)

### P0b — 秩可靠性预测效度分析(选项 D / GPT §7.4,~2 天,零训练)

**设计要点(吸取历史地雷教训)**:目标变量=**实际训练结局**(P0a 各 arm 的最终 flow/dmot),不是任何带地板的参照;解释变量=各 arm reward 的离线组内秩指标(Spearman-vs-code 参照、pairwise 翻转率、弱信号窗口翻转率)。需先扩 `cache_reward_spaces.py` 补缓存 `pixel_tok`/`dyn` 分量(现缓存只有 lpips/mse/code_rms/phi/dino)。统计:arm 数~8 → 用 Spearman 秩相关 + bootstrap 置信区间,按窗口分层做稳健性;**预期效度成立则升格 C3 的正向补强,不成立则只作 C1 的附录分析,不影响主线**。

### P0c — 写作立即启动(与 P0a 并行)

Intro/Method(C1+C2)与 Fig1-3 不依赖任何未跑数据,今天就能写。图表清单沿用 GPT 文档 §8(Fig5=intervention ladder,图题用 "Where should we intervene?")。

### P1 — 收尾图表(P0a 跑完后各 ~0.5 天)
- 分布面板(FD-DINOv2/KID/PRDC)只测 table1_final 的赢家与 baseline ckpt。
- 主观图:随机样本 + 运动明显样本,pixel vs pixel_tok vs RC(pixel_tok_dyn) vs GT。

### P2 — 多步扩展(仅在选项 A/C 路线被拍板时启动,硬止损 7/10)
缺口精确清单:①ctx_msp rollout 实现+对权重冒烟(1.5-2.5 天,解码不能是乱码);②压缩 tokenizer 地板测量(0.5 天,eval-only,顺带回答"floor-cancel 多步是否成立"=多步版 B1);③80 动态 token 布局实测(0.5 天)。止损日未冒烟则放弃,回 P0/P1 写作。

### 多步(multi-step)

- [x] M0 下载现成权重(`rt1-world-model-multi-step-base/-rlvr` + `rt1-compressive-tokenizer`)
- [x] M1 时空段 id 机制实现 + 单测(`build_seg_ids_st`,纯 CPU,与权重无关,`MULTISTEP_SEGIDS_OK`)
- [ ] M2 多步 rollout 实现(复用 RLVR-World `ctx_msp` processor + 压缩 tokenizer,忠实构造 BOS/EOS 序列)
- [ ] M3 对下载权重冒烟:确认自回归生成 + 解码出的帧不是乱码(阻塞 M4 前必须过)
- [ ] M4 实测 80 个动态 token 的空间布局(决定走真空间分段还是退化 token_chunk)
- [ ] M5 多步段级 GRPO vs 多步全局 GRPO vs 官方 `multi-step-rlvr`(现成基线)
- [ ] M6 分步误差累积 $h\in\{1,3,5,8\}$ 的 LPIPS/MSE 曲线

---

## 1. 证据状态总览

| 类别 | 结论 | 状态 | 出处 |
|---|---|---|---|
| 地板 φ_tok 存在、度量依赖 | 感知度量地板大、MSE 小(旁证) | ✅ 已证 | results_v1v6 C1 |
| 翻转率 = $\arccos(\rho)/\pi$ | 5 seed 0.185≈0.186,弱信号翻转更多 | ✅ 已证(最硬) | results_v1v6 C2 |
| floor-cancel 收益 ∝ 地板 | pixel_tok +14% flow(5/5);mse_tok 弱(3/5,负对照) | ✅ 已证 | exp_v7 |
| 弱动态残差 pixel_tok_dyn | λ=0.10:flow/dmotion 4-5/5 seed↑,保真不掉(Pareto) | ✅ 已证 | reward_summary, exp_v9 |
| code 普遍更好 | flow 不复现、FD/KID/PRDC 各臂≈、全臂 recall 坍塌 | ❌ 已证否(红线) | grpo_full, dist_panel |
| 全局 GRPO 改造 | Dr.GRPO 中性、硬过滤 v3 变差 | ❌ 已证否(→段级动机) | results_v1v6 |
| 段级信用 on 裸 code(负对照) | 弱信号上段级信用无效/负(λ 越大越差) | ✅ 已证(负对照,支持"信号丰富度是前提") | seg_2x2_* |
| **段级信用 on pixtok_dyn(真正方法)** | λ=0.7:flow +0.054(t=2.35)、dmot +0.028(t=2.69),保真不明显掉,n=5 | 🟡 **中期正信号,退化检查未过前不可拍板** | segpd_2x2_* |
| 全局锚点对齐(§阻塞检查) | seg_grpo λ=0 是否≈真实全局 GRPO+pixel_tok_dyn | ⬜ **未验证,阻塞判绿灯** | 待补跑 |
| 多步时空段级 GRPO | 误差累积↓ | ⬜ 待扩 dor | — |

---

## 2. 主实验:单步空间段级 GRPO on pixtok_dyn(正在跑,真正的方法)

**假设**:固定 reward substrate 为已证最优的 `pixel_tok_dyn`(λ_dyn=0.10,γ_dyn=0.25),把 rollout 级标量优势换成空间段级信用分配,动态指标稳超全局 GRPO 且保真不掉。

**设计**(唯一变量 = 优势估计器):`--seg_reward pixtok_dyn`、同协议(lr1e-5/K16/150步/tw24/ew12)、5 seed、`--deterministic`。
- **内部基线** = 同网格 `seg_grpo λ=0`(段级 reward 池化后取全局)——**⚠️ 尚未验证等价于真正的全局 GRPO 路径,见 §3.0 阻塞检查,判绿灯前必须先过**。
- **处理**(已跑/在跑) = seg_grpo × {2x2(K=4), 4x4(K=16)} × {λ=0.7, 1.0} × γ=0.2。

**命令**(前台):
```bash
cd /root/autodl-tmp/vote2world; P=/root/miniconda3/bin/python
for cfg in "2x2 0.0" "2x2 0.7" "2x2 1.0" "4x4 0.0" "4x4 0.7"; do set -- $cfg
  $P scripts/train_grpo.py --rewards code --modes gt_only --seeds 0,1,2,3,4 \
     --adv_estimator seg_grpo --seg_grid $1 --seg_reward pixtok_dyn \
     --seg_lambda $2 --seg_gamma 0.2 --dyn_lambda 0.10 --dyn_gamma 0.25 \
     --steps 150 --K 16 --deterministic --out_dir outputs/segpd_${1}_l$2; done
```
**判绿灯**:`python scripts/analyze_seg.py outputs` → 配对 Δ(treatment − 同网格 λ=0)+ 配对 t 检验。
- **绿**(前提:§3.0 全部通过) = flow/dmotion wins≥4/5 或配对 t 显著,且 LPIPS/PSNR/SSIM 不明显掉。
- **红** = 段级信用也洗没 → 诚实消融,回退"reward 校准 + 机理"双贡献。

**中期读数(2x2, λ=0.7 vs λ=0, n=5, last-3-eval 均值)**:flow meanΔ=+0.0538(paired t=+2.35,df=4)、dmot meanΔ=+0.0280(t=+2.69);LPIPS/PSNR/SSIM 均不显著(|t|<1)。**方向正确,但 §3.0 锚点检查未过前不作为定论**。

---

## 3. 单步消融与阻塞性退化检查(导师纪要 §11,判绿灯前必须完成)

### 3.0 阻塞性检查(优先级最高,必须先做)

| 检查 | 命令 | 目的 |
|---|---|---|
| **全局锚点对齐** | `--adv_estimator grpo --rewards pixel_tok_dyn --dyn_lambda 0.10 --dyn_gamma 0.25 --seeds 0,1,2,3,4 --steps 150 --K 16 --deterministic --out_dir outputs/global_pixtokdyn` | 验证 `segpd_2x2_l0.0` 是否≈这个真正的全局路径(不是靠段级机制凑出来的近似基线) |
| **K=1 退化**(pixtok_dyn, λ=0.7) | `--seg_grid 1x1 --seg_reward pixtok_dyn --seg_lambda 0.7 --seg_gamma 0.2 ...` | 段机制本身(K=1 时只有 1 段)是否引入失真;应≈全局锚点 |
| **γ=0 消融**(pixtok_dyn, λ=0.7, 2x2) | `--seg_gamma 0.0`(其余同 §2 命令) | 隔离"codec-aware 阻尼"与"纯空间信用"各自的贡献 |

若任一检查不成立,先修实现,再判 §2 的结果是否可信。

### 3.1 参数消融(需补齐)

| 消融 | 目的 | 判据 | 状态 |
|---|---|---|---|
| λ 扫 {0,**0.3**,0.7,1.0} | 段级 vs 全局融合;λ=0 退化对齐 | λ=0≈锚点;某 λ 最优 | **λ=0.3 缺,需补** |
| K 扫 {1,4,16} | 粒度;K=1 退化对齐 | K=1≈锚点;中间粒度最优 | K=1 缺,见 §3.0 |
| γ 消融(pixtok_dyn, γ=0 vs 0.2,固定 λ=0.7) | codec-aware 阻尼是否有独立贡献 | γ>0 更好 | 缺,见 §3.0 |
| code reward 上的段级信用(负对照,已做) | 验证段级信用收益依赖 reward 信息量 | 弱信号上无效/负 | ✅ 已证(seg_2x2_*) |
| 段级可靠性软加权(rankrel) | 把 C2 变方法 | 有增益或诚实负结果 | ⬜ 待做 |

### 3.2 最终数字的样本量(导师纪要 §11.B)

绿灯配置确定后,seed 从 5 扩到 8–10 再定稿主表数字。

---

## 4. 多步时空段级 GRPO(权重已备,待扩 dor)

**前置**:下载现成权重(网络,不抢 GPU):
```bash
cd /root/autodl-tmp/vote2world; export HF_ENDPOINT=https://hf-mirror.com
for r in rt1-world-model-multi-step-base rt1-world-model-multi-step-rlvr rt1-compressive-tokenizer; do
  /root/miniconda3/bin/huggingface-cli download thuml/$r --local-dir checkpoints/$r; done
```
**dor 扩展**:加载多步基座 + 压缩 tokenizer;H 步自回归 rollout;实测 80 动态 token 空间布局(网格→空间段,否则 token_chunk);复用 `seg_advantage`(段索引扩成 $H\times K$)。
**对照**:现成 `multi-step-rlvr` = RLVR 官方多步 RL 结果,直接当基线。
**主指标**:分步 $h\in\{1,3,5,8\}$ 的 LPIPS/MSE + flow;看误差累积是否随时间信用分配下降。
**判据**:多步段级 > 多步全局 GRPO,且 $\ge$ 官方 multi-step-rlvr。5090 先小规模,不够再 H800(数据可无损拷)。

---

## 5. 评测(独立指标,只评测不作 reward)

- 逐样本:flow/dmotion(动态)、LPIPS/PSNR/SSIM(保真)——训练目标之外。
- 分布/特征级面板:`scripts/eval_fd_dino.py`(FD/KID/PRDC × DINOv2+Inception,离线不重训,n≥512 稳)。口径:各训练臂≈、均>base;近零 recall = 多样性坍塌(动机性观察,非铁证)。
- code_rms 同源 → 只进附录。
- **待办**:§2/§3 选出绿灯配置(`segpd_*`)后,对其 ckpt 补跑 `eval_fd_dino.py`,确认段级信用没有让分布/多样性变得更差(尤其 recall)。

---

## 6. 论文图表(对应贡献)

| 图/表 | 内容 | 贡献 |
|---|---|---|
| Fig1 | pipeline:预测 token → (解码后 reward | 段级信用分配) 两条线 | 总览 |
| Fig2 | 翻转率 vs $\arccos(\rho)/\pi$(5 seed)+ σ⋆ 分层 | B(机理) |
| **Table1** | **单步:seg_grpo(K,λ) vs 全局,flow/保真配对多 seed** | **A(主)** |
| Fig3 | λ/K/γ 消融 + λ=0/K=1 退化对齐 | A(信用分配起效) |
| Fig4 | reward 校准:pixel_tok/pixel、pixel_tok_dyn λ 扫、mse_tok 负对照 | C(配菜) |
| Fig5 | 多步:分步误差累积 seg vs 全局 vs 官方 rlvr | A(多步) |
| Fig6 | 分布面板(各臂≈base↑,recall 坍塌) | 评测/动机 |
