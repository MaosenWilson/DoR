# RC-GRPO 论文蓝图 — 现状 · 创新点 · 方法公式 · 实验设计（2026-07-04）

> **单一权威文档**:整合终局判决后的全部内容,自包含(不需读其他文档/对话历史)。所有数字来自实际实验,无编造。
> 论文:*Mind the Reconstruction Floor: Calibrating Verifiable Rewards for Tokenized Video World Models*
> 目标:AAAI-2027(摘要 7/21、全文 7/28)。方法名:**RC-GRPO(Reward-Calibrated GRPO)**。

---

## 0. 现状速览(2026-07-04 晚)

| 线程 | 位置 | 状态 |
|---|---|---|
| **Table 1 同 sweep 复测**(单步终局数据,7 arm × 8 seed) | westb:31055 `outputs/table1_final` | 🔄 在跑(~26h),跑完**单步训练实验全部收官** |
| **多步 pilot v2**(修稳定性后重跑,raw/rc × 3 seed) | westd:17223 `outputs/msp_pilot` | 🔄 待跑/在跑(~6h);v1 因 batch_windows=1 梯度噪声在 step 40-70 后 4/6 发散,已改 2 对齐单步协议 |
| 理论与消融(C1/C2/C3 所有支撑数据) | 已完成 | ✅ 不再需要新实验 |
| 写作 | — | 待启动(蓝图=本文档) |

**多步 v1 pilot 的关键信息(虽然发散,但信号极好)**:唯一稳定的 seed 上,base 0.2157 → **官方 multi-step-rlvr 0.2115** → 我们 raw 0.1953 → **我们 rc 0.1922**(LPIPS,同协议同窗口)——**100 步精简 GRPO 已超官方 RL checkpoint**,rc 最优。发散前各 run best 值(0.192-0.207)全部好于官方基线。修稳后若复现,这是论文的第四块拼图。

---

## 1. 论文定位(一句话)

> Tokenized video RLVR 的要害不在 GRPO 内部,而在喂给 GRPO 的 reward 排序。我们刻画 tokenizer 解码引入的 **codec reconstruction floor** 并证明它以可预测的概率($\arccos(\rho)/\pi$)翻转 GRPO 消费的组内排序;据此提出 **RC-GRPO**——一个 GRPO 前置的 reward 校准 prefix(可达目标对齐 + 弱解码前动态残差),优化器零改动;并用受控的 **intervention-locus 研究**证明:reward 侧校准产生稳定增益,而三类 GRPO 侧信用重分配均无法恢复已污染的排序——**有效干预点是 reward 构造,不是 advantage 重整**。

---

## 2. 创新点(三条,不膨胀)

### C1 — 问题与机理:codec reconstruction floor + 排序腐蚀理论
- 定义并测量 tokenizer 往返重建地板(与预测质量无关、度量依赖)。
- 证明其危害不是绝对分数偏移,而是**腐蚀 GRPO 唯一消费的组内排序**,翻转概率有闭式:$P_{\text{flip}}=\arccos(\rho)/\pi$,实测 0.185 vs 理论 0.186(5 seed 逐 seed 吻合)。

### C2 — 方法:RC-GRPO(GRPO 前置 reward 校准 prefix)
- **可达目标对齐**(floor-cancellation):比较目标从原始 GT 换成 tokenizer 可达重建,flow +14%(5/5 seed);
- **弱解码前动态残差**:码空间运动残差(方向+幅度),$\lambda_{\text{dyn}}=0.10$ 为 Pareto 点;
- GRPO 优化器**零改动**——不是新优化器,是让 vanilla GRPO 接到可靠排序。

### C3 — 干预点定位(intervention-locus study,系统性对照发现)
- 同任务同协议下:reward 侧两个干预**显著有效**;GRPO 侧三层级干预(全局标量改造 / 段级替代式 / 段级残差式,最后者 8-seed 同 sweep 配对、方法学无可挑剔)**全部无效**。
- 机理解释:排序腐蚀发生在 reward 生成处;在污染排序上重新分配 credit(任何粒度)无法恢复丢失信息。
- 给出实践准则:改 GRPO 之前,先检查喂进去的 reward 排序是否可靠。

**(潜在第四块,取决于多步 v2)**:RC 校准迁移到多步 rollout,在缓解误差累积上超过官方 multi-step RLVR checkpoint。成立则并入 C2 的泛化证据,不单列贡献。

---

## 3. Method(完整公式)

### 3.1 问题设定与记号

转移 $q=(s,a)$(context 帧 + 动作)。tokenized 视频世界模型作为策略 $\pi_\theta$,对同一转移采样 $G$ 个候选下一帧 token 序列 $\{o_i\}_{i=1}^G$,解码 $\hat s'_i=\text{decode}(o_i)$,与真值 $s'$ 逐候选算 verifiable reward。RLVR-World 忠实 reward:
$$R^{\text{RLVR}}_i=-\big[\text{MSE}(\hat s'_i,s')+\text{LPIPS}_{\text{vgg}}(\hat s'_i,s')\big]$$
GRPO 组内归一化优势与策略梯度(精简版,无 critic、无 PPO clip——同批生成与梯度间权重不变,ratio≡1):
$$A_i=\frac{R_i-\text{mean}_jR_j}{\text{std}_jR_j+\epsilon},\qquad \mathcal L_{\text{PG}}=-\frac1G\sum_{i=1}^G A_i\log\pi_\theta(o_i\mid q)$$
**关键结构事实**:GRPO 只消费**组内相对排序**(优势符号、间隔),不消费绝对分数。

### 3.2 C1:地板与排序腐蚀

**codec reconstruction floor**(确定性量;"噪声"一词仅在下方排序建模中使用):
$$\phi^{(d)}_{\text{tok}}=\mathbb E_{s'}\big[d\big(\text{decode}(\text{encode}(s')),\,s'\big)\big]$$
性质(已测):①与预测质量无关(真值自己往返都回不去);②是任何 token 预测的下界(最优也只能到 $\tilde s'=\text{decode}(\text{encode}(s'))$);③度量依赖——单步 CNNFSQ:LPIPS 地板 0.112 > 帧间动态信号 0.082,SSIM 同向,MSE 反之(地板 0.0019 < 信号 0.0034);多步压缩 tokenizer:LPIPS 地板 ≈0.077。

**排序腐蚀**:建模 $R_i=R^\star_i+\eta_i$($R^\star$ 干净奖励,$\eta$ 地板诱发的候选级扰动)。组内均值减法消不掉 $\eta_i$(逐候选随机量,非常数)。若含噪与干净奖励相关性为 $\rho$,联合高斯下候选对排序翻转概率:
$$\boxed{P_{\text{flip}}=\frac{\arccos(\rho)}{\pi}}$$
**验证**(5 seed,以解码前 code 空间为干净参照):实测 0.185 vs 理论 0.186(逐 seed 差 ~0.002);弱信号窗口翻转 ~30% vs 强信号 ~10%;corr(翻转率, 信号强度) = −0.79。

### 3.3 C2:RC-GRPO reward 校准 prefix

**组件一:可达目标对齐(floor-cancellation)**
$$R^{\text{pix\_tok}}_i=-\text{LPIPS}_{\text{vgg}}(\hat s'_i,\ \tilde s'),\qquad \tilde s'=\text{decode}(\text{encode}(s'))$$
完美 token 预测 ⇒ $\hat s'_i=\tilde s'$ ⇒ 满分 0(地板在源头被消掉)。

**组件二:解码前动态残差**(单帧感知奖励易被静态外观主导;操作任务的难点是动作导致的状态变化)。码空间 $z=\text{indices\_to\_codes}(\cdot)$,$z_t$=当前帧、$z'$=真值下帧、$z_i$=候选:
$$\Delta z_i=z_i-z_t,\quad \Delta z'=z'-z_t$$
不能用 $\|\Delta z_i-\Delta z'\|$(代数上塌回 $\|z_i-z'\|$,无新信息),用方向+幅度:
$$R^{\text{dyn}}_i=\cos(\Delta z_i,\Delta z')-\gamma_{\text{dyn}}\Big|\log\frac{\|\Delta z_i\|+\epsilon}{\|\Delta z'\|+\epsilon}\Big|,\qquad \gamma_{\text{dyn}}=0.25$$

**最终 RC reward**(喂给完全不动的 vanilla GRPO):
$$\boxed{R^{\text{RC}}_i=z_G\big(R^{\text{pix\_tok}}_i\big)+\lambda_{\text{dyn}}\,z_G\big(R^{\text{dyn}}_i\big)},\qquad \lambda_{\text{dyn}}=0.10,\quad z_G(x_i)=\frac{x_i-\text{mean}_j x_j}{\text{std}_j x_j+\epsilon}$$
**写作注意**:$z_G$ 是**分量聚合前的尺度归一**(LPIPS 与动态分数量纲差一个量级),与 GRPO 自身的 advantage 归一化是两个不同步骤,必须显式区分,避免"重复归一化"质疑。

**支撑消融(已完成)**:
| 消融 | 结果 |
|---|---|
| pixel → pixel_tok(组件一) | flow +14%,5/5 seed |
| mse → mse_tok(负对照:MSE 地板≈0,收益应≈0) | 3/5 seed、符号不稳(纯噪声)——**收益来自地板,不是任意换目标** |
| $\lambda_{\text{dyn}}\in\{0.05,0.10,0.25\}$ | 0.05 太弱;**0.10 Pareto**(flow 0.2871±0.0153);0.25 动态更强但保真显著变差 |
| 8-seed 复测(真全局+RC) | flow 0.2745±0.0212,与 5-seed 跨批一致 |

### 3.4 C3:intervention-locus(GRPO 侧三层级,公式与结果)

**层级一:全局标量改造**(Dr.GRPO 去偏 $A_i=R_i-\bar R$ 无 /std;硬过滤地板主导组)→ Dr.GRPO 中性;硬过滤全面变差(丢数据 + 破坏 /std 对低 spread 组的天然缩放)。

**层级二:段级替代式**。帧 token 网格 16×20 切 $K$ 段,段级组内优势混合池化伪全局:
$$\hat A_{i,k}=\lambda A^{\text{seg}}_{i,k}+(1-\lambda)A^{\text{global,pooled}}_i,\qquad A^{\text{seg}}_{i,k}=\frac{r_{i,k}-\mu_k}{\sigma_k+\gamma\tilde b_k+\epsilon}$$
表面"绿灯"(vs 内部 λ=0:flow +0.054, t=2.35),但**诊断证伪**:段池化基线(flow 0.2142)远低于真全局(0.2871)——增益只是弥补段池化自伤;且 $(1-\lambda)$ 缩减全局信号。

**层级三:段级残差式(GP-SegGRPO,方法学上最干净的设计)**:
$$\hat A_{i,k}=A^{\text{global}}_i+\lambda\,\Delta A^{\text{seg}}_{i,k},\qquad \Delta A^{\text{seg}}_{i,k}=A^{\text{seg}}_{i,k}-\tfrac1K\sum_{k'}A^{\text{seg}}_{i,k'}$$
全局项与标准 GRPO 逐表达式相同;残差零均值(只做 rollout 内再分配);$\lambda=0$ 或 $K=1$ 由构造严格退化。**8 seed 同 sweep 配对结果**:$\lambda\in\{0.1,0.3,0.5,0.7\}$ 无一显著超过真全局(flow 最好 t=+0.58;λ=0.1 反而 t=−1.84;保真全部噪声内)。

**结论**:$$\text{effective intervention point}=\text{reward construction before GRPO}\ \neq\ \text{advantage reshaping inside GRPO}$$

**(附:可靠性加权的三代演进,写 Discussion/附录)**:$q=\exp(-\alpha\tilde b)$(无推导,废)→ 方差分解 $q=\hat\sigma^{*2}/(\hat\sigma^{*2}+b^2)$(把地板**均值**当噪声标准差;组内常数被 GRPO 归一化消掉,真噪声是候选间波动 $\sigma_\eta\ll b$;实测 q≡0,同一病根解释了更早 dorw 融合塌缩)→ 相关性 $q=[\max(0,\rho)]^2$(= C2 理论量 ρ 的段级在线版;q 分布健康但训练仍不占优)。教训:**一旦 reward 已校准,融合/加权/再分配的边际收益趋零**。

### 3.5 多步扩展(ctx_msp,公式与实现)

**序列格式**(照 RLVR-World 官方 processor 逐行核实):压缩 tokenizer 把 context 帧编成 1280 个 token(32×40),每个未来帧只编 80 个动态 token(8×10);词表 $V=4375$,布局 [dyn 原值 | ctx $+V$ | action $+2V$ | BOS/EOS 9006/9007]:
$$\text{seq}=[\underbrace{c_{1:1280}+V}_{\text{ctx}}\ |\ \underbrace{d^{(1)}_{1:80}}_{\text{dyn}_1}\ \underbrace{a^{(1)}_{1:13}+2V}_{\text{act}_1}\ |\ d^{(2)}\ a^{(2)}\ |\cdots]$$
**rollout**(交替生成,忠实复刻 verl interact loop):每帧自回归采样 80 个 dyn token(temp 1.0, top-k 100),然后**强制注入**该帧动作的 13 个离散 token;首个未来帧无前置动作 ⇒ 不可预测 ⇒ reward 跳过(官方惯例)。

**多步 reward**(跳首帧、mean 聚合,$F=T-1$ 个未来帧):
$$R^{\text{raw}}_i=-\frac{1}{F-1}\sum_{h=2}^{F}\big[\text{MSE}+\text{LPIPS}\big](\hat x_{i,h},\,x_h)\qquad(\text{RLVR 忠实})$$
$$\boxed{R^{\text{RC-msp}}_i=-\frac{1}{F-1}\sum_{h=2}^{F}\big[\text{MSE}+\text{LPIPS}\big](\hat x_{i,h},\,\tilde x_h)},\qquad \tilde x_h=\text{detokenize}(c,\,d^{\text{GT}}_h)$$
即可达目标对齐直接迁移(多步地板实测 LPIPS≈0.077 > 0,故 floor-cancellation 有的放矢)。策略梯度只对 dyn token 位置的 logp 求和(action 是强制的,不进目标)。

---

## 4. 实验设计

### 4.1 平台与协议(所有实验统一)
- RT-1 机器人操作(fractal20220817);单步:thuml single-step base(Llama 768d/12L)+ CNNFSQ frame tokenizer(320 tok/帧);多步:thuml multi-step base + 压缩 tokenizer;**只做 GRPO 微调,不预训练**;官方 RLVR checkpoint(单步/多步)作为现成对比。
- 协议:lr=1e-5, G=K=16, steps=150(多步 100), batch_windows=2, train/eval windows=24/12(多步 24/8), `--deterministic`(pin cuBLAS/cuDNN;注意力 backward 仍非确定 ⇒ **只信同 sweep 配对**,配对 t 检验,readout=最后 3 次 eval 均值)。
- 评测(独立于训练目标):flow(RAFT 光流一致性)、dmotion(帧差余弦)=动态;LPIPS-vgg/PSNR/SSIM=保真;多步加分步 LPIPS/MSE(误差累积);分布面板 FD-DINOv2/KID/PRDC 只做评测(集合统计,当不了 per-sample reward)。

### 4.2 主表(Table 1,westb 在跑)
同 sweep 7 arm × 8 seed:`pixel / a0faithful(忠实 RLVR) / mse / mse_tok(负对照) / pixel_tok / pixel_tok_dyn(=RC,主方法) / code(对照)` + base 行(step-0 eval 提取)。判读:RC 对 baseline 的配对优势 + 组件链(pixel→pixel_tok→RC)每步必要性。

### 4.3 多步 pilot(westd)
raw vs rc × 3 seed(修稳:batch_windows 1→2;v1 有 4/6 在 step 40-70 后发散,两臂均等——稳定性问题非 reward 问题)+ base/官方 rlvr eval-only 基线。判据:**rc vs raw 同 seed 配对**(目标对齐是否迁移多步);超官方 rlvr 是 bonus(v1 已有强信号:稳定 seed 上 rc 0.1922 < raw 0.1953 < 官方 0.2115 < base 0.2157)。绿灯后扩 5-8 seed。

### 4.4 干预阶梯(C3 的数据,全部已有)
横轴:vanilla+raw → vanilla+RC → Dr.GRPO → 硬过滤 → 段级替代 → 段级残差;纵轴:flow/dmot 增益与保真变化。信息:reward 侧两步有增益,GRPO 侧全部平/负。

### 4.5 收尾(离线,各 ~0.5 天)
赢家 ckpt 分布面板;主观图(随机 + 运动明显 case:raw 偏静态外观 vs RC 抓住局部运动);(可选)秩可靠性→训练结局的预测效度分析。

---

## 5. 图表规划

| 图表 | 内容 | 支撑 |
|---|---|---|
| Fig 1 | pipeline + 地板来源(RLVR 对原始 GT vs RC 对可达目标 + 码空间残差旁路) | 总览 |
| Fig 2 | 跨度量地板 vs 帧间信号(单步 + 多步 0.077) | C1 |
| Fig 3 | 翻转率实测 vs $\arccos(\rho)/\pi$ + 弱/强信号分层 | C1 |
| **Table 1** | 7 arm × 8 seed 同 sweep(4.2) | C2 |
| Fig 4 | 组件链消融(pixel→pixel_tok→RC;mse_tok 负对照;λ_dyn 扫) | C2 |
| Fig 5 | 干预阶梯("Where should we intervene?") | C3 |
| Fig 6 | 多步误差累积:base/官方 rlvr/raw/rc 分步 LPIPS | C2 泛化 |
| Fig 7(可选) | 主观对比 | 佐证 |

---

## 6. 写作红线(证据已定,不许越线)

不写:"code 普遍更好"、"feature reward beats image"(轴是 pre/post-decode 不是 pixel/feature)、"FID/FVD 可作 reward"(集合统计)、"多指标融合是贡献"、"CAST-GRPO/GP-SegGRPO 是主方法"(已判负,只作 C3 素材)、"所有 GRPO 变体无效"(只声称本场景三类已测干预无法恢复污染排序)、"地板是随机噪声"(确定性量,噪声只在排序建模用)。
必写:$z_G$ 与 GRPO 归一化的区分;C3 的对照链完整性(每层级的设计与判据);多步的稳定性修正(batch_windows 对齐,诚实记录 v1 发散)。
