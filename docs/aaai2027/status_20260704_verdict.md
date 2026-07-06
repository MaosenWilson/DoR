# 现状文档 — G5 终局判决后的完整证据状态（2026-07-04）

> **用途**：给多个外部智能体和人工做独立判别。自包含,不需要读对话历史或旧文档。
> **取代** `status_20260703_handoff.md`（该文写于 GP-SegGRPO 判决之前,已过时）。
> 所有数字来自实际跑出的结果（配对多 seed + t 检验）,无编造。
> 论文：*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models*。
> 目标 AAAI-2027：**摘要 2026-07-21(还有 17 天),全文 2026-07-28(还有 24 天)**。

---

## 0. TL;DR

1. **地基机理(B)与 reward 校准(C)稳固**:tokenizer 重建地板的定义与跨度量刻画、翻转率理论 $\arccos(\rho)/\pi$(实测 0.185 vs 理论 0.186)、floor-cancellation(+14% flow, 5/5 seed)、动态残差 Pareto 点(λ_dyn=0.10)——全部有干净消融链。
2. **GRPO 侧信用分配改造已被三个层级系统性证伪**(2026-07-04 终局),每一层都有干净对照:全局标量改造无效、段级替代式的"增益"实为弥补自伤、段级残差式(方法学上最干净的设计,8 seed 同 sweep 配对)与真全局无差。
3. **正面结论**:该任务里所有可测增益都来自 **reward 侧校准**,credit assignment 粒度**不是瓶颈**。
4. **论文处境**:两个稳固贡献(B+C)+ 一个系统性负结果。**第三贡献缺口是当前唯一待决策问题**,选项见配套文档 `next_steps_20260704_options.md`。

---

## 1. 任务与平台(固定,无争议)

- 任务:RT-1 机器人操作(fractal20220817)单步下一帧预测。世界模型 = Llama 式自回归(768d/12L),每帧 320 个视觉 token(16×20 网格,CNNFSQ tokenizer)。thuml 预训练基座,**只做 GRPO 微调**。
- RL:精简 GRPO(无 PPO clip;同批生成与梯度间权重不变,ratio≡1),G=16 候选/转移,HF `.generate(top_k=100)`。
- 协议(所有实验统一):lr=1e-5, steps=150, train_windows=24, eval_windows=12, batch_windows=2, `--deterministic`。
- 评测(独立于训练目标):flow(RAFT 光流一致性,动态)、dmotion(帧差余弦,动态)、LPIPS-vgg/PSNR/SSIM(保真)。readout=最后 3 次 eval 均值。
- 已知方法学限制:GPU 非确定性使同 seed 跨批次 flow 波动 ±0.03~0.08 → **只信同 sweep 配对比较**;跨批引用必须标注。

---

## 2. 稳固证明的结论(B+C,论文的两根柱子)

### B. 地板与翻转率机理

| 结论 | 证据 |
|---|---|
| B1 tokenizer 往返重建地板 $\phi_{\text{tok}}=\mathbb E[d(\text{decode}(\text{encode}(s')),s')]$ 存在、与预测质量无关、度量依赖(LPIPS 地板 0.112 > 帧间动态信号 0.082;MSE 反之) | eval-only 探针,跨 5 度量 |
| B2 地板腐蚀 GRPO 组内排序:候选对翻转概率 = $\arccos(\rho)/\pi$($\rho$=含噪奖励与干净参照的相关性) | 实测 0.185 vs 理论 0.186,5 seed 逐 seed 吻合;弱信号窗口翻转 ~30% vs 强信号 ~10%;corr(翻转率,信号强度)=−0.79 |

### C. Reward 校准(已冻结配方:`pixel_tok_dyn`)

$$R_i = z_G\big(-\text{LPIPS}(\hat s'_i,\ \tilde s')\big) + 0.10\cdot z_G\big(R^{\text{dyn}}_i\big),\quad \tilde s'=\text{decode}(\text{encode}(s'))$$
$$R^{\text{dyn}}_i=\cos(\Delta z_i,\Delta z')-0.25\big|\log(\|\Delta z_i\|/\|\Delta z'\|)\big|\ \ (\text{码空间运动残差},\ \Delta z=z-z_t)$$

| 消融 | 结果 |
|---|---|
| C1 floor-cancellation 主效应(比较目标:原始 GT → 可达目标 $\tilde s'$) | flow **+14%**,5/5 seed 一致 |
| C2 mse_tok 负对照(增益应∝该度量的地板大小) | LPIPS 地板大→5/5 一致;MSE 地板≈0→3/5、符号不稳(纯噪声) |
| C3 λ_dyn 扫描 {0.05, 0.10, 0.25} | 0.05 太弱(flow 0.2536);**0.10 Pareto(flow 0.2871±0.0153)**;0.25 动态更强但保真显著变差 |

**真全局 GRPO + pixel_tok_dyn 的 8-seed 复测(2026-07-04,g4_global)**:flow 0.2745±0.0212 / dmot 0.1732±0.0119 / LPIPS 0.1433±0.0019——与历史 5-seed(0.2871±0.0153)跨批一致,**这就是当前最强的、可复现的完整方法**。

---

## 3. 系统性证伪:GRPO 侧信用分配改造(三层级,全部有干净对照)

### 3.1 三层级证伪总表

| 层级 | 尝试 | 设计质量 | 结果 |
|---|---|---|---|
| **全局标量改造** | Dr.GRPO 去偏;硬过滤地板主导组 | 5 seed | Dr.GRPO 中性;硬过滤全面变差(丢数据+破坏 std 的天然降权) |
| **段级替代式** | $\hat A=\lambda A^{\text{seg}}+(1-\lambda)A^{\text{global,pooled}}$(把 reward 按 2×2 段池化,段级组内归一化后与"池化伪全局"混合) | 5 seed,配对 λ=0 | 曾现"绿灯"(flow +0.054 t=2.35),**但后被 A13 诊断证明是假象**:段池化基线(flow 0.2142)远低于真全局(0.2871),所谓增益只是弥补池化自伤;且混合公式按 (1−λ) 缩减全局信号 |
| **段级残差式** | **GP-SegGRPO**:$\hat A_{i,k}=A^{\text{global}}_i+\lambda\Delta A^{\text{seg}}_{i,k}$,残差零均值(只做 rollout 内再分配),全局项与标准 GRPO 逐表达式相同,λ=0/K=1 由构造严格退化 | **8 seed 同 sweep 配对,方法学上无可挑剔** | **与真全局无差**(见 3.2) |

### 3.2 GP-SegGRPO 终局数据(2026-07-04,g4_*,5 臂 × 8 seed 同 sweep)

| 配置 | flow | dmot | LPIPS | PSNR | SSIM |
|---|---|---|---|---|---|
| 真全局 A0 | 0.2745±0.0212 | 0.1732±0.0119 | 0.1433±0.0019 | 25.078±0.146 | 0.7952±0.0024 |
| gpseg λ=0.1 | 0.2550±0.0237 | 0.1705±0.0169 | 0.1435 | 25.048 | 0.7949 |
| gpseg λ=0.3 | 0.2716±0.0259 | 0.1791±0.0158 | 0.1433 | 25.087 | 0.7952 |
| gpseg λ=0.5 | 0.2795±0.0316 | 0.1801±0.0179 | 0.1435 | 25.059 | 0.7952 |
| gpseg λ=0.7 | 0.2668±0.0189 | 0.1800±0.0123 | 0.1440 | 25.037 | 0.7946 |

配对 t(vs A0,df=7,显著需 |t|≳2.36):
- λ=0.1:flow **t=−1.84**(1/8 wins,偏负);其余噪声。
- λ=0.3/0.5/0.7:全部指标 |t|<1.4。dmot 有微弱一致正向(+0.006~0.007,t=0.8~1.3,wins 5-6/8)但远不显著。

**判决(赛前锁定的 G5 规则)**:无任何 λ 过线 → 段级信用分配降级为系统性负结果。

### 3.3 其他已证伪(避免重复踩坑)

| 尝试 | 根因 |
|---|---|
| "code(解码前)reward 普遍更好" | flow 优势跨批不复现;分布面板(FD-DINOv2/KID/PRDC)各臂无差 |
| 段级信用用在裸 code reward 上 | λ 越大越差(信号太弱,切段只剩噪声) |
| 数据集级离线 Wiener 权重融合(dorw) | 塌缩成纯 code:地板均值被当噪声标准差,权重压到 ~0 |
| codec_fused v1($q=\exp(-\alpha\tilde b)$) | 拍脑袋启发式,无推导 |
| codec_fused v2(方差分解,地板均值当噪声) | 实测 q≡0:地板均值是组内常数,GRPO 归一化自动消掉,真噪声是候选间波动 $\sigma_\eta\ll b$(此诊断同时解释了 dorw 塌缩) |
| codec_fused v3/v3.1($q=\rho^2$ 相关性版) | q 分布健康(K=4 时 q≈0.56 有真实段间分化)但训练不占优且保真下降;目标对齐修复(v3.1)将保真惩罚减半仍不翻盘 |

---

## 4. 正面解读(这批负结果的价值)

1. **"reward 侧可修、GRPO 侧不可修"是一个有完整对照链的系统性发现**:同一任务、同一协议、同一评测,reward 侧的两个干预(floor-cancel、dyn 残差)都产生显著、可复现的增益;GRPO 侧的三个层级干预(含一个方法学上无可挑剔的 residual 设计)全部无效。这不是"我们没调好",是有对照的排除。
2. **翻转率机理(B2)为"为什么 GRPO 侧不可修"提供了解释框架**:GRPO 只消费组内排序;排序被地板腐蚀是 reward 的性质,不是 advantage 结构的性质——在被污染的排序上重新分配 credit(任何粒度)都无法恢复丢失的信息,**修复必须发生在 reward 生成处(去地板、对齐可达目标)**。机理与实验互相印证。
3. **codec_fused 三代演进留下一条可写的方法学线索**:可靠性加权的噪声量必须锚定候选间波动而非地板均值(v2 教训);即便噪声量对了(v3 的 q 分布健康),reward 融合也不是瓶颈(v3.1 仍不占优)——**一旦 reward 已被校准(pixel_tok_dyn),融合/加权/再分配的边际收益趋零**。

---

## 5. 论文当前形态(判决后)

| 贡献 | 内容 | 状态 |
|---|---|---|
| C1(稳) | codec reconstruction floor 的定义、跨度量刻画、翻转率机理 $\arccos(\rho)/\pi$ | ✅ 数据全齐 |
| C2(稳) | codec-calibrated reward:可达目标对齐 + 弱码空间动态残差,完整消融链 | ✅ 数据全齐 |
| C3(**缺口**) | 原计划 = 段级信用分配(CAST-GRPO/GP-SegGRPO),已判负 | 🔴 需要决策:用什么补位(见配套 options 文档) |

**候选 Table 1(如果今天就写)**:pixel(RLVR 基线)/ pixel_tok / pixel_tok_dyn(ours)× {flow, dmot, LPIPS, PSNR, SSIM},加 mse_tok 负对照与 λ_dyn 扫描为消融——reward 校准作为主方法完全成立,但"改良 GRPO"的第二创新点(导师 06-27 的要求之一)当前没有正结果承载。

---

## 6. 数据资产清单(全部在 westd:17223 `/root/autodl-tmp/vote2world/outputs/`)

| 目录 | 内容 | seed |
|---|---|---|
| `g4_global`, `g4_gpseg_2x2_l{0.1,0.3,0.5,0.7}` | G4 终局判决数据 | 8 |
| `grpo_v9_dyn_lam010/005/025` | λ_dyn 扫描(C3 消融) | 5 |
| `grpo_v1/singles`(pixel/code/mse/a0faithful), `grpo_v5/floorcancel`(pixel_tok/ssim_tok), `grpo_v7/mse_tok` | reward 校准链(C1/C2 消融) | 5 |
| `segpd_2x2_l{0,0.7,1.0}`, `seg_2x2_*`, `segcf*` | 段级信用三代(负结果素材) | 5 |
| `grpo_full`(8 臂全扫) | 早期 reward 空间扫描 | 5 |
| `outputs/analysis/reward_spaces_s*.npz`, `floor_metrics.json` | B1/B2 机理原始数据 | 5 |
| ckpt 全部保存 | 可离线补分布面板(`scripts/eval_fd_dino.py`) | — |
| 多步权重已下载(`rt1-world-model-multi-step-base/-rlvr`, `rt1-compressive-tokenizer`) | 多步扩展的全部前置 | — |

多步现状:时空段 id 机制已实现+单测过(`src/dor/multistep.py`);**rollout 未实现**(需忠实复刻 RLVR 的 ctx_msp processor + 压缩 tokenizer 序列格式,文件内有 7 步实现计划)。
