# 引言修订建议 + 消融设计（回应 M1/M2/M3）

> 2026-07-19。基于 experiments.md §3.1 的 episode-disjoint 5-seed 主表，所有 margin 已算准。
> 目的:堵住审稿人对"C2 被过度包装""增益只在一个数据集""优于 RLVR 无数字"三枪。

## 0. 关键数字(全部来自 §3.1，可直接引用)

主表(episode-disjoint, 5 seeds, final step, LPIPS/MSE):

| arm | LPIPS | MSE | LPIPS-last | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| base | 0.21273 | 0.015480 | 0.22783 | 18.827 | 0.73749 |
| official RLVR | 0.20403 | 0.014390 | 0.21206 | 19.106 | 0.75323 |
| seq+raw | 0.19915 | — | 0.21180 | — | — |
| seq+RC | 0.19933 | 0.013820 | 0.21176 | — | — |
| TR+raw | 0.20118 | 0.013941 | 0.21487 | — | — |
| **TR+RC (full)** | **0.19657** | **0.013556** | **0.20767** | **19.447** | **0.75908** |

算出的 margin:
- **full vs base**:LPIPS **−7.6%**、MSE **−12.4%**（= 摘要现有数字，正确）。
- **full vs 官方 RLVR**:LPIPS **−3.7%**、MSE **−5.8%**、LPIPS-last **−2.1%**、PSNR +0.34、SSIM +0.006。
- **C2 隔离 = TR+RC − seq+RC**（§3.1 配对）:LPIPS **−1.39%**(4/5, t=−1.94, p=0.125)、
  LPIPS-last −1.93%(4/5, p=0.149)、**MSE −1.91%(5/5, t=−2.40, p=0.074, sign-flip p=0.031)**。

⚠️ **"5/5 vs 官方 RLVR"未验证**:§3.1 只有 means。写"相对官方 RLVR 降 3.7%/5.8%"是安全的（表支持）;
若要写 seed 胜率，先从 per-seed 数据核对 TR+RC 各 seed vs 官方，别凭记忆写 5/5。

## 1. M1 —— C2 措辞降档 + 正文隔离消融

### 1a. 摘要/贡献(2)的 C2 措辞:从"强增益"改为"结构对齐"

C2 现在写成"以完整帧 token 块为责任单元…分配逐帧回报"——**这已经是结构性表述,没 over-claim 数字,保留即可**。唯一要确保的是:**别让贡献(3)的合并 7.6% 替 C2 背书**。做法 = 贡献(3)里把 C1 机制与 C2 增益拆开陈述（见 §3）。

### 1b. 正文必须有的 C2 隔离消融表（诚实版）

**Table (C2 isolation)** —— 直接放 TR+RC − seq+RC 的配对统计（上面的数字），并如实写:
> "隔离 Temporal Return（相对 sequence-level，同为 RC verifier、同超参）:五个指标平均方向一致,
> MSE 在 5/5 seeds 改善;但 n=5 下无任一双侧 paired t 达 0.05,故 C2 表述为**具有时间结构的稳定
> 方向性趋势**,而非显著增益。"

**并给完整 2×2**（base/official + seq×{raw,RC} × TR×{raw,RC}），诚实标注:
> "单独 RC（seq+RC）与单独 Temporal Return（TR+raw）均未超过 sequence+raw 基线;只有二者组合
> （TR+RC）取得最优。difference-in-differences 交互为 −0.00479（4/5, p=0.309），受 seed-0 的
> raw-return 退化影响,**不作为统计超加和主张,不列第三项贡献**。"

这样审稿人做 2×2 时看到的,正是我们主动摊开的——**抢在他前面诚实,是防"过度包装"最好的盾**。

### 1c. C1 的贡献靠"审计",不靠训练分解

**关键框架**:不要试图从训练 2×2 里"隔离 C1 的增益"(会得到"seq+RC 不如 seq+raw"的尴尬)。
C1 的证据是**同候选审计**(Δρ>0/Δflip<0，跨 codec/平台)+ **三 codec 重建误差**,不是训练分解数字。
- C1 = 机制诊断 + 校准(审计表 + 地板表);
- C2 = 帧块信用对齐(隔离消融 + 逐 horizon 结构);
- full method = 打赢 base 和官方 RLVR(主表)。
三条证据线各管一段,不互相背书,也不硬凑加性分解。

## 2. M3 —— 摘要/贡献(3)加 RLVR-World margin

### 摘要最后一句改写

现:"…降低约 7.6% 与 12.4%，并优于同协议下的RLVR-World模型。"
→ **改**:
> "在 episode-disjoint 复评中,完整方法相对冻结 base 将 LPIPS 与 MSE 分别降低约 **7.6%** 与
> **12.4%**;相对同协议下公开的 RLVR-World checkpoint,进一步将 LPIPS 与 MSE 降低约 **3.7%** 与
> **5.8%**。"

（两个基线说清=顺带解决 m1;赢官方 RLVR 的硬数字=M3。若核对到 seed 胜率再补"(N/5 seeds)"。）

### 贡献(3)最后一句同样处理

现:"…完整 RC-GRPO 相对冻结 base 将 LPIPS 与 MSE 分别降低约 7.6% 与 12.4%，并优于同协议下的RLVR-World模型。"
→ **改**:
> "…完整 RC-GRPO 相对冻结 base 将 LPIPS/MSE 降低约 7.6%/12.4%,相对同协议官方 RLVR-World
> checkpoint 再降约 3.7%/5.8%;隔离 Temporal Return 相对 sequence-level RC 在五指标上方向一致
> (MSE 5/5 seeds),但在 n=5 下未达双侧显著,故作为具有时间结构的方向性结果报告。"

**这一句同时做完 M1(隔离 C2 且诚实标注)+ M3(RLVR margin),是性价比最高的一处改动。**

## 3. M2 —— 单平台增益的防御(scope + 强化)

贡献(3)已诚实区分"跨 codec/世界模型验证机制、固定 RT-1 报告质量",保留。**强化**两点:
1. 把"赢官方 RLVR"提到显眼位置(摘要+贡献都有数字)——在 SOTA 基线上赢,比"降 base 7.6%"更硬;
2. 明确写一句适用边界(可放 experiments/limitation):
   > "训练侧预测质量增益在 RT-1 压缩 tokenizer 协议上建立;VP2 等即时监督充分的近天花板平台上
   > Temporal Return 无可测增益,我们将其报告为适用边界而非普适性能提升。"

## 4. 建议新增/落实的消融与实验清单

| 编号 | 消融/实验 | 状态 | 作用 |
|---|---|---|---|
| A | 主 2×2 + base + official（6 arm 表） | ✅ 有数据 | headline;full 打赢两基线 |
| B | C2 隔离:TR+RC − seq+RC 配对统计 | ✅ 有数据 | M1 诚实隔离,标注不显著 |
| C | C1 机制:同候选审计 Δρ/Δflip（跨 codec/平台）+ 三 codec 地板 | ✅ 有数据 | C1 证据(不靠训练分解) |
| D | **时间对应控制**:L=1 / L=3 / full / candidate-shuffled return | 🔜 E2 待跑 | C2 因果:收益是否来自正确 future-credit |
| E | horizon 压力测试 T∈{4,6,8} | 🔜 E3 待跑 | C2 结构随长度 |
| F | GAE λ 扫描(K=8) | ✅ 有数据 | **仅附录**:降方差消融,gae0.7>return(3/3),不进 headline |

**最该补的是 D(shuffled control)**——它是 C2 唯一缺的因果证据,审稿人会直接问"你的时序收益是不是
随便打乱也一样"。E2 设计已在 experiments.md,paired seeds 0-2,主判据 full aligned vs shuffled 的
LPIPS 配对差。跑通=C2 从"方向性趋势"升级到"有因果对照的方向性结果",显著加分。

## 5. 一句话总结给你

- **M1**:C2 措辞已够稳(结构性),关键是正文放 B(隔离)+ 2×2 诚实标注,别让合并数字替 C2 背书;
- **M3**:摘要和贡献(3)各改一句,把"优于 RLVR"换成"再降 3.7%/5.8%"(seed 胜率待核);
- **M2**:把"赢官方 RLVR"提显眼 + 一句适用边界;
- **最该跑**:D(shuffled temporal-correspondence control),补 C2 唯一的因果缺口。
