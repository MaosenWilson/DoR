# Experiments：正式证据账本

本文件只记录当前 C1/C2 的正式协议、可复核结果、必要边界和尚缺实验。历史开发过程由 Git
history 保存，不把 candidate pairs、horizons、windows 或 generation draws 伪装成独立训练 seeds。

## 1. 实验设置

### 1.1 数据集与世界模型

**RT-1 `fractal20220817`.** 主实验使用 RLVR-World 发布流程中处理后的 `fractal20220817`
机器人操作数据。我们固定使用 20 条轨迹；每帧为 $256\times320$ RGB 图像，每个时间步包含
13 维连续机器人动作。所有方法共享相同的数据缓存、动作离散化范围、视觉 tokenizer 和初始世界
模型权重。

单步实验采用公开的 CNN-FSQ tokenizer 与 Llama-style autoregressive world model。模型接收连续
4 帧上下文及其动作；每帧被编码为 $16\times20=320$ 个视觉 tokens，13 维动作分别量化到 256
个区间，模型随后生成下一帧的 320 个视觉 tokens。该设置用于验证 C1 的同候选排序变化及
raw-anchored RC 训练 pilot。

多步实验采用 RLVR-World 的 compressive FSQ tokenizer 和 multi-step base checkpoint。每个窗口
包含 $T=8$ 帧，即 1 帧初始观测和 7 个自回归 future-frame blocks；每个未来帧由 $8\times10=80$
个 dynamics tokens 表示。我们遵循原流程，不对第一个预测帧施加直接 action-conditioned reward，
而在其后的 horizons 2--7 计算逐帧 verifier；Temporal Return 仍可通过后续 return 为前面的 token
block 分配优势。所有训练臂均从同一个 multi-step base checkpoint 初始化，公开的
multi-step RLVR checkpoint 仅作为 evaluation-only 参照，不声称复现其完整训练过程。

原始训练协议使用 sampling seed 1 从 stride-8 window pool 中固定抽取 24 个训练窗口和 8 个在线
评测窗口。为排除同轨迹泄漏，最终结果另外在训练阶段从未出现的 8 条轨迹上评测其全部 32 个
stride-8 窗口；该 episode-disjoint manifest 在所有方法、训练种子和公开 checkpoint 间完全一致。

**VP2.** 跨平台实验使用公开的 action-conditioned iVideoGPT checkpoint
`thuml/ivideogpt-vp2-robosuite-64-act-cond`。已完成实验对应 `PushCenterMulti` 数据，共 5,000
条轨迹。iVideoGPT 使用两帧 $64\times64$ 上下文图像，并以 $4\times4$ dynamics-token block
自回归生成未来帧。训练、校准和测试 manifest 按 episode 划分，任何两个 split 均不共享轨迹。
该平台用于检验 C1 是否跨数据分布和 tokenizer 成立，并作为“即时帧监督较充分”的 C2 边界
场景；它不能代表 VP2 的全部 11 类任务。已完成的 H8 训练边界实验使用 24 条训练轨迹和 16 条
测试轨迹，每条轨迹固定一个窗口，并以 episode-macro 聚合。

为进一步检验具有接触后果和延迟状态变化的场景，我们已准备独立的
`thuml/ivideogpt-vp2-robodesk-64-act-cond` checkpoint 与 RoboDesk `open_drawer` 数据。
`open_drawer` 包含 2,500 条、每条 35 步的轨迹，共 87,500 个 transitions；原始图像为
$256\times256$，动作为 5 维，并提供 2,250/250 条 train/valid 官方划分。该资源当前只属于待完成
扩展协议，在获得正式 gate 和训练结果前不进入结果表。

**NVIDIA Cosmos DV-FSQ.** 为检验编解码重建误差是否只是本项目 tokenizer 的特例，我们使用完全
独立设计的 Cosmos 离散视频 tokenizer，在相同 RT-1 机器人帧上执行冻结的
encode--quantize--decode 审计。该 codec 采用因果时序、小波变换与 DV-FSQ；输入按其公开接口
缩放为 $256\times256$、9 帧 clip。Cosmos 当前没有与本文协议匹配的候选世界模型，因此只用于
检验“独立 codec 是否存在不可忽略的编解码重建误差”，不用于 same-candidate rank repair 或 GRPO 训练主张。

### 1.2 候选采样与可验证奖励

除特别说明外，每个输入条件采样 $G=16$ 个候选。RT-1 和 VP2 使用自回归采样，temperature 固定
为 1.0，top-$k$ 固定为 100。配对实验共享数据顺序、policy seed、candidate generation schedule
和 evaluator，以减小候选采样噪声对方法差异的影响。视觉 tokenizer 和 LPIPS 网络在全部实验中
保持冻结。

我们沿用 RLVR-World 的逐样本 full-reference verifier，基础距离为

$$
d(x,y)=\operatorname{MSE}(x,y)+\operatorname{LPIPS}_{\mathrm{VGG}}(x,y).
$$

raw verifier 使用真实未来帧 $s'_t$ 作为目标，
$r^{\mathrm{raw}}_{i,t}=-d(\hat s_{i,t},s'_t)$；RC verifier 只将目标替换为
$s'_{\mathrm{RC},t}=D(Q(E(s'_t)))$，即
$r^{\mathrm{RC}}_{i,t}=-d(\hat s_{i,t},s'_{\mathrm{RC},t})$。两个 verifier 使用同一组候选、相同
距离函数和相同标度，因此比较只隔离 target reachability 的影响。MSE 和 LPIPS 等权相加，不根据
测试集结果重新调权。

sequence-level GRPO 先对各帧 reward 求平均，再在 $G$ 个候选内进行标准化，并把同一个优势复制
到整段输出。Temporal Return 则对第 $t$ 个视觉 token block 计算
$G_{i,t}=\sum_{u=t}^{T}\gamma^{u-t}r_{i,u}$，随后在同一 horizon 的候选组内标准化。两种方法使用
相同的 token log-probability、优化器和 KL 项，只改变 advantage 与视觉 token block 的对应方式。

### 1.3 训练细节

所有训练均使用 AdamW，其余优化器参数采用 PyTorch 默认值，梯度范数裁剪为 1.0。主实验不在
同一 batch 上进行多轮 PPO 更新，也不根据测试指标选择中间 checkpoint。训练期间每 10 个
updates 记录一次内部评测曲线，headline 结果统一读取 final checkpoint。主要超参数如下：

| 设置 | 初始模型 | updates | batch windows | $G$ | learning rate | KL | $\gamma$ | seeds |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RT-1 single-step C1 pilot | single-step base | 150 | 2 | 16 | $1\times10^{-5}$ | 0 | -- | 0--2 |
| RT-1 multi-step 主实验 | multi-step base | 30 | 2 | 16 | $1\times10^{-5}$ | 0.001 | 0.95 | 0--4 |
| VP2 C1 training boundary | VP2 RoboSuite checkpoint | 20 | 2 | 16 | $3\times10^{-6}$ | 0.001 | -- | 0--2 |

RT-1 single-step pilot 使用 24 个训练窗口和 12 个内部评测窗口。RT-1 multi-step 主实验使用
24/8 个训练/在线评测窗口，并采用 coefficient 0.001 的 frozen-reference sampled KL；KL 估计器
固定为 RLVR-World/VERL-compatible `low_var_kl`。参考策略为冻结的 multi-step base model。
早期使用 linear sampled log-ratio 的探索结果不与该主协议混合，只能作为实现敏感性或历史记录。

确定性运行固定 Python、NumPy、PyTorch 和 CUDA 随机种子，设置 cuBLAS workspace，并关闭
cuDNN benchmark。由于 32 GB 显存下强制 math SDPA 会导致显存溢出，我们保留 memory-efficient
attention；其 backward 不保证逐位确定，因此统计推断始终以独立的 paired policy seeds 为单位，
而不把一次运行视为完全可复现的确定性轨迹。

### 1.4 测试指标与模型选择

无论训练使用 raw 还是 RC target，所有 headline fidelity 指标都相对原始真实未来帧计算。
我们报告 LPIPS-VGG、MSE、PSNR 和 SSIM；其中输入图像先缩放到 $[0,1]$，LPIPS 按其标准接口
映射到 $[-1,1]$。多步实验将 horizons 2--7 的均值作为 full-rollout 指标，并单独报告最后一个
horizon 的 LPIPS-last。post-quant latent RMS、token Hamming distance、RAFT flow、frame-delta
motion 以及 DINO/KID 只用于机制或边界分析，不能替代 raw-frame fidelity headline。

RT-1 最终复评对每个 checkpoint 使用相同的 32 个 episode-disjoint windows、相同 window order、
$G=16$、generation seed 999 和一次固定 generation draw。主结果采用 window-macro；另报告
episode-macro 作为不同轨迹窗口数的敏感性分析。base checkpoint 与官方 RLVR checkpoint 通过
同一评测脚本测量。除非实验名称明确标注为 checkpoint-selection ablation，主表一律使用训练终点，
不从 step 10/20/30 中择优。

跨平台 C1 审计不更新模型参数。VP2 的短期审计使用 horizon 2、64 个 contexts 和 2 个独立
generation draws；长期审计使用 horizon 8、8 个 contexts 和 2 个 draws，二者均采用 $G=16$。
所有 draws 先在所属 context 内聚合，再按 episode 构造置信区间。Cosmos 重建误差审计不生成候选组，
其统计单位为冻结 codec 重建的真实帧。

### 1.5 统计分析与计算环境

训练方法比较以 paired policy seed 为推断单位，报告各臂均值、逐 seed 配对差、改善 seed 数、
paired $t$ test、exact sign-flip test 和 seed-level bootstrap confidence interval。五个 seeds 内的
多个 windows、horizons、候选 pairs 或 generation draws 均为相关观测，不能扩充为额外训练样本。

C1 的冻结候选审计在同一 candidate group 上计算 raw/RC reward 与 decoder-input latent/code
reference 的 Spearman correlation，并统计 pairwise ranking flip。置信区间按 episode cluster
bootstrap 构造；VP2 的重复 generation draws 先在 context/episode 内聚合，不作为独立
样本。正式主结果同时保存数据划分、checkpoint selection、聚合规则及其 manifest/hash，后续分析
不得静默更改协议后与原结果混报。

实验使用 PyTorch 在单张 NVIDIA GeForce RTX 5090 32 GB GPU 上完成。训练采用 float32 policy
weights，并在 tokenizer/decoder 等显存密集模块中使用 bfloat16 autocast。代码、manifest、随机
种子、checkpoint 路径和评测聚合方式均写入每个运行的 JSON artifact；正式发布版本将同时提供
环境依赖锁定文件与 checkpoint/manifest hashes。

## 2. C1：Reachability-Constrained Rank Calibration

### 2.1 RT-1 Same-Candidate Audit

RC 只替换 target，不改变 candidates 或 metric。相对 decoder-input representation readout 的
episode-cluster结果为：

| tokenizer/protocol | groups | $\Delta\rho$ [95% CI] | $\Delta$flip [95% CI] | status |
|---|---:|---:|---:|---|
| single-step CNN-FSQ | 148 | +0.0168 [+0.0035,+0.0289] | -0.0082 | supported |
| multi-step compressive FSQ | 1,152 | +0.0145 [+0.0105,+0.0176] | -0.0087 | supported |

multi-step 的 horizons 2--7 均为 $\Delta\rho>0$、$\Delta$flip$<0$。联合高斯诊断
$\arccos(\rho)/\pi$ 与实测 flip 接近，但该关系是已有概率恒等式，不作为原创理论。

### 2.2 Cross-Codec Floor Audit

在同一 LPIPS-VGG 口径下，三个 FSQ codec 实例均产生非零 encode--decode 编解码重建误差：

| codec | architecture role | reconstruction-floor LPIPS |
|---|---|---:|
| RLVR-World CNN-FSQ | 单步图像 tokenizer | $\sim0.053$ |
| compressive FSQ | 多步视频 tokenizer | $\sim0.077$ |
| NVIDIA Cosmos DV-FSQ | 独立因果视频 tokenizer | $0.190$ |

该表支持“编解码重建误差不是单一 tokenizer 的偶然现象”。由于三个 codec 的空间分辨率、时间压缩和
训练数据不同，本文不据此声称重建误差随压缩率单调增大；Cosmos 也不承担候选排序结论。

### 2.3 Cross-World-Model Rank Audit

| platform | protocol | $\Delta\rho_s$ [95% CI] | $\Delta$flip [95% CI] | RC-top latent gain [95% CI] |
|---|---|---:|---:|---:|
| VP2 | H2, 64 contexts, 2 draws | +0.0932 [+0.0674,+0.1217] | -0.0409 [-0.0524,-0.0306] | +0.00203 [+0.00115,+0.00300] |
| VP2 | H8, 8 contexts, 2 draws | +0.0753 [+0.0408,+0.1104] | -0.0346 [-0.0488,-0.0207] | +0.00142 [+0.00013,+0.00238] |

VP2 使用 decoder 实际消费的 post-quant continuous latent，而不是把类别 token ID 当作欧氏距离；
该表支持 C1 rank mechanism 从 RT-1 迁移到另一 tokenizer/生成架构，不等价于跨平台 raw-pixel
training gain。

### 2.4 Training Conversion Boundary

RT-1 single-step 的 raw/RC/raw-anchored 三臂 pilot 完成 seeds 0--2。raw-anchored update 相对 raw
的 final LPIPS 为 $-0.00117$（3/3），MSE 为 $-6.0\times10^{-5}$（3/3），post-quant latent RMS
为 $-0.00739$（3/3）；相对 pure RC 的 LPIPS 仅 2/3 且区间跨零。因此它通过 RT-1 pilot，未证明
优于 RC。

VP2 的 RC/Temporal-Return 训练未改善 raw-GT LPIPS。结论是：**跨平台 rank repair
成立，跨平台 raw-GT training conversion 尚未成立。** 论文不能把 C1 写成普适性能提升。

## 3. C2：Frame-Block Temporal-Return GRPO

### 3.1 Episode-Disjoint Low-Variance-KL Factorial

正式四臂均使用 `low_var_kl`、paired seeds 0--4，并在训练完全未见的 8 episodes / 32 stride-8
windows 上统一复评。每个 checkpoint 使用相同 window order、generation seeds、$K=16$ 和 final
step；主表为 window macro，episode macro 作为不等窗口数敏感性分析。

| arm | LPIPS $\downarrow$ | LPIPS-last $\downarrow$ | MSE $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ |
|---|---:|---:|---:|---:|---:|
| base checkpoint | 0.21273 | 0.22783 | 0.015480 | 18.827 | 0.73749 |
| official RLVR checkpoint | 0.20403 | 0.21206 | 0.014390 | 19.106 | 0.75323 |
| sequence + raw | 0.19915 | 0.21180 | 0.013679 | 19.403 | 0.75555 |
| sequence + RC | 0.19933 | 0.21176 | 0.013820 | 19.382 | 0.75577 |
| Temporal Return + raw | 0.20118 | 0.21487 | 0.013941 | 19.341 | 0.75362 |
| Temporal Return + RC | **0.19657** | **0.20767** | **0.013556** | **19.447** | **0.75908** |

`Temporal Return + RC - sequence + RC` 的 paired training-seed 结果：

| metric | mean delta | relative change | wins | paired $t$ | two-sided $p$ | exact one-sided sign-flip $p$ |
|---|---:|---:|---:|---:|---:|---:|
| LPIPS | -0.002764 | -1.39% | 4/5 | -1.94 | 0.125 | 0.0938 |
| LPIPS-last | -0.004093 | -1.93% | 4/5 | -1.79 | 0.149 | 0.0938 |
| MSE | -0.000264 | -1.91% | 5/5 | -2.40 | 0.074 | 0.0313 |
| PSNR | +0.0647 | +0.33% | 4/5 | +1.75 | 0.155 | 0.0625 |
| SSIM | +0.00331 | +0.44% | 4/5 | +1.65 | 0.175 | 0.1250 |

该复评保留了五个指标的平均方向，MSE 在 5/5 seeds 上改善；但没有任一 paired $t$ test 达到
双侧 0.05。episode-macro 仍保持五指标平均方向，MSE 为 5/5、PSNR 为 5/5、LPIPS 为 3/5。
因此 C2 当前是跨指标且具有时间结构的稳定趋势，不能再沿用旧 8-window readout 中
“LPIPS-last 已显著、LPIPS 5/5”的表述。窗口、episode、horizon 或 generation candidates 均不充当
新增 training seeds。

### 3.2 Component Interaction

四臂中完整组合均值最低，但 episode-disjoint LPIPS difference-in-differences interaction 为
$-0.00479$，4/5 同向，paired $t=-1.16$，$p=0.309$。该均值受 seed 0 的 raw-return 退化影响，
不能解释为 RC 与 Temporal Return 的统计超加和，不列第三项贡献。完整组合相对原始
sequence+raw 的 LPIPS 虽平均改善 $0.00258$，但仅 2/5 seeds 同向，不能作为 headline paired gain。

### 3.3 Structural Evidence and Missing Control

episode-disjoint 复评中，Temporal Return 相对 sequence RC 的平均 LPIPS delta 从 horizon 2 的
$-0.00133$ 逐步扩大到 horizon 7 的 $-0.00409$，六个 horizons 均为负；MSE、PSNR 和 SSIM 的
逐-horizon均值也全部朝正确方向。由于每个 seed 内的 horizons 强相关，不能把 30 个
seed--horizon cells 当独立样本；按 seed 拟合的 LPIPS delta slope 为
$-5.34\times10^{-4}$/horizon，仅 3/5 同向，paired $p=0.179$。该结构签名支持“远期帧受益更大”的
解释，但仍不能替代正式时间对应控制。

投稿前必须比较 $L=1$、$L=3$、full 与 candidate-shuffled return。shuffled control 在每个 horizon
保持 reward multiset，只破坏 candidate identity 的跨时间对应；它直接检验收益是否来自正确的
future-credit assignment。

## 4. Boundary Readouts

- **Motion.** Temporal Return 未稳定改善 dmotion/flow，不声称更好的 dynamics fidelity。
- **Distribution.** 单 seed DINO/KID readout 显示 post-training 可能牺牲集合分布指标；该结果只作
  limitation，不推导普适 trade-off。
- **Official checkpoint.** 只能说在相同 held-out protocol 下比较，不能声称复现并击败
  RLVR-World 的完整训练 recipe。

## 5. Compact Negative Ledger

| rejected direction | retained conclusion |
|---|---|
| 多指标/多域静态融合、learned fusion、pre-decode code/gradient | 未超过最小 RC verifier |
| Rank-Guard、floor filtering、spatial weighting | 排序诊断量不能直接升格为在线权重 |
| MRRT/local target search | 更低 target reconstruction error 未转化为更好 policy training |
| RC Energy/distribution reward | 分布效用提高但违反 raw-fidelity non-inferiority |
| RCAV/action verifier | real transitions 可观测，generated candidates 上 command utility 不可靠 |
| Dr.GRPO、REAL-style VPO、GSPO、segmental variants | 未超过 vanilla GRPO，部分显著有害 |
| Rank-Reliable/RCTR/CATR temporal variants | 未超过 plain Temporal Return |
| VP2 delayed-influence adaptive return (UATR2) | 与 raw 基本持平且不优于 candidate-shuffled control，停止扩展 |

该表只界定方法选择，不作为贡献，也不声称穷尽 reward 或 GRPO 设计空间。

## 6. Pending Decisive Experiments

### E1. Episode-Disjoint Frozen-Checkpoint Reevaluation（完成）

20 个 final checkpoints、base 与 official checkpoint 已在训练完全未见的 8 episodes / 32 windows
上完成统一复评，manifest SHA-256 为
`fb22f86c309336debc49f4e798e1ad1c1a12bb7f630600e76eef282616f78012`。原始 JSON 与汇总已备份到
`tmp/msp_episode_disjoint_eval/`；正式判读见 3.1--3.3。

### E2. Temporal Correspondence Controls

固定 RC verifier 与所有训练超参，比较 $L=1$、$L=3$、full return 和 candidate-shuffled return，
paired seeds 0--2。主判据是 full aligned 相对 shuffled 的 LPIPS 配对差；同时检查 full 不差于
$L=1$，以及 $L=1\rightarrow L=3\rightarrow$ full 的方向。未通过则 C2 只能表述为在固定协议中
有效的 block-wise objective，不能声称正确 temporal correspondence 是原因。

### E3. Horizon-Length Stress Test

优先在 E2 有结论后再跑 $T\in\{4,6,8\}$ 的 sequence-RC/return-RC paired comparison。各长度
默认共享 learning rate、KL、group size、updates 和 seeds；若因算力只完成部分长度，主文报告
已完成长度并在附录说明，不把未跑长度写成已验证。

### E4. VP2 External-C2 Validation Program

#### E4.0 Task eligibility before method comparison

VP2 官方基准包含 RoboSuite tabletop pushing 与 RoboDesk 共 11 类任务，并明确指出 pixel/perceptual
metrics 未必对应下游规划成功。当前 `PushCenterMulti` 中，逐帧 raw reward 已提供很强的即时监督；
plain return、RCTR/CATR 和 UATR/UATR2 的阴性结果共同否定了“继续调整一组全局 horizon weights”
这条路线。后续不再在该任务上搜索折扣率、静态 mask 或新的全局时间系数。

对每个 RoboDesk 候选任务先运行冻结分支诊断。固定 candidate prefix 和后续 actions，重复采样
continuations，检验当前 block 是否对后续 utility 产生可复核、candidate-specific 的影响。任务准入
要求：(i) aligned prefix--continuation identity 优于 within-context shuffled control；(ii) 至少两个
horizons 的 episode-cluster 置信区间下界为正；(iii) split-half branch values 能在 held-out branches
上预测 candidate ordering。未通过的任务只作即时监督对照，不进入 C2 训练主张。

#### E4.1 Priority A: Branch-Value Temporal Credit

第一优先级不是再次给所有 candidates 共享一个时间权重，而是估计每个 candidate block 的边际
未来影响。令 \(\widehat V_{i,t}\) 为固定前缀 \(\hat s_{i,\le t}\) 与真实后续 actions 后，多次分支
续写得到的 future utility 均值，定义

$$
c_{i,t}=r_{i,t}+\lambda_t\left(\gamma\widehat V_{i,t}-\widehat V_{i,t-1}\right),
$$

再在同一 horizon 的 candidate group 内标准化。\(\widehat V_{i,t-1}\) 在采样 block \(t\) 前确定，
因此是 context-matched counterfactual baseline；\(\lambda_t\) 只由冻结校准集上的 split-half
reliability 决定。即时 reward 足够时 \(\lambda_t\to0\)，存在可靠延迟后果时才增加未来信用。
该设计借鉴 counterfactual credit assignment 的条件基线和 RUDDER 的 return decomposition，但
具体到 autoregressive video frame blocks 的 branch-value residual 是本文待验证的本地化设计，
不能在门控前写成贡献。

训练准入门：held-out branch selection 相对 immediate reward 为正，aligned credit 优于 candidate-
shuffled baseline，且至少两个任务通过。通过后先跑 3-seed、20-step 四臂
`sequence-RC / return-RC / branch-RC / shuffled-branch-RC`；primary 是 VP2 task classifier 或
MPC success，LPIPS/MSE/SSIM/last-frame fidelity 为 secondary non-inferiority。只有 branch-RC
同时优于 return-RC 和 shuffled control 才扩到 5 seeds。

#### E4.2 Backup B: Cross-Fitted Return Decomposition

若 branch value 有稳定信号但在线分支代价过高，则在冻结 rollouts 上训练 episode-disjoint 的 prefix
return predictor \(F(\hat s_{i,\le t},a_{\le t})\)，以
\(c_{i,t}=F_{i,t}-F_{i,t-1}\) 重新分配 rollout utility。该方案对应 RUDDER 的 prediction-difference
思想，在线成本低于分支采样；风险是 predictor 学到画面状态捷径。准入要求 OOF return prediction
优于 mean baseline、temporal shuffle 失效、contribution sum 近似恢复原 return，并在 held-out
candidate selection 上优于 immediate reward。任一条件失败即停止，不用训练结果补救 predictor。

#### E4.3 Backup C: Executability-Aware Credit

若 RoboDesk 的延迟后果主要表现为接触、抽屉/滑轨状态或物体离桌，而 raw LPIPS 不敏感，则用真实
transitions 训练 episode-disjoint inverse-dynamics/process verifier，评价生成 transition 是否执行了
给定 action。它只作为逐帧 process utility，仍由同一个 branch-value temporal formula 分配信用。
必须做 \(2\times2\) 对照：`raw vs raw+process` 乘 `sequence vs branch credit`；只有 temporal
interaction 为正，才能支持 C2，而不能把 reward 增益冒充信用分配增益。先验风险是 generated
frames 上的 inverse-dynamics calibration 失效，RT-1 上已有同类阴性，故该路线排在第三。

#### E4.4 Control-centric evaluation

VP2 的最终外部 readout 应加入官方 sampling-based planning success 或任务 classifier score。它不是
新方法，但可检验“LPIPS 几乎不变、控制后果改善”的可能性。只报告 perceptual metrics 会偏离 VP2
的设计目的；反过来，也不能只报告 planning success 而隐藏 raw-frame fidelity 退化。

### E5. RoboNet External Audit

下载完成后先建立 episode-disjoint manifests，再依次运行 checkpoint/data provenance、same-candidate
rank gate 和 delayed-signal diagnostic。RoboNet 若缺少可复核的 delayed-effect/task-success readout，
只承担 C1 外部机制与 C2 边界，不强行加入 C2 headline。

## 7. Claim-Evidence Contract

| claim | status |
|---|---|
| target-set mismatch causes measurable rank disagreement | supported across three platforms |
| RC reduces same-candidate rank corruption | supported across three platforms |
| raw-anchored update universally improves raw-GT fidelity | not supported |
| Temporal Return improves RT-1 tail fidelity | 4/5 direction; not conventionally significant |
| Temporal Return improves RT-1 full-rollout LPIPS | 4/5 direction; not conventionally significant |
| Temporal Return improves RT-1 MSE | 5/5; exact one-sided sign-flip $p=0.031$, paired $t$ $p=0.074$ |
| benefit requires correct candidate-time correspondence | pending E2 |
| benefit grows with rollout length | pending E3 |
| RC and Temporal Return are statistically super-additive | rejected |
