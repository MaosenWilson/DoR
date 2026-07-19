# 论文图表方案（C1 广 + C2 深 + 主观图 + 多场景）

> 2026-07-18。回答:C1 怎么展示、长序列怎么展示、主观图怎么做、多场景怎么给。
> 标注数据状态:✅ 已有 / 🔜 需生成。

## A. C1（重建残差改变 GRPO 组内排序）的展示

**Fig 1 — 机制:翻转律。** ✅
散点/曲线:实测 pairwise flip rate vs 预测 arccos(ρ)/π,跨多个奖励空间(LPIPS/MSE/MSE+LPIPS)。
一条 y=x 参考线,点全落其上 → 重建残差改变组内排序是可预测的定律。

**Fig 2 — 跨 tokenizer 的编解码重建误差与可达目标。** ✅ 标量已有；定性帧待导出
定量子图报告 CNN-FSQ、压缩 FSQ 与 Cosmos DV-FSQ 的编解码重建误差；定性子图对每个平台并排展示
`raw GT / decode(encode(GT)) / |residual|`。三个 tokenizer 的分辨率、空间/时间压缩方式不同，
因此主文只主张“不同离散 tokenizer 均存在不可忽略的编解码重建误差”，不把 LPIPS 数值直接解释成压缩强度的
单调函数。若要讨论压缩率，需要另外报告可比的 bits-per-pixel/token-rate 控制实验。

**Table 1 — 跨世界模型的同候选排序审计。** ✅
行 = RT-1 CNN-FSQ、RT-1 压缩 FSQ、iVideoGPT VP2；列 = Δρ [CI]、Δflip、状态。
这些设置具有可复用的候选生成器，因此可直接检验 RC 对 GRPO 组内排序的影响。Cosmos 当前只有
tokenizer、没有与本文协议匹配的候选世界模型，故只进入 Fig. 2 的重建误差审计，不伪造排序结果。

**Fig 3 — 诊断链闭环。** ✅
同一批候选,raw→RC 后 ρ↑、flip↓(配对 t)。证明校准确实修复排序,不是相关性。

## B. C2（长序列 GRPO 时序信用）的展示

**Fig 4 — 长序列退化与平台 headroom(核心长序列图)。** ✅ 数据已有
per-horizon LPIPS(和/或 SSIM)随 horizon 的曲线,多平台叠加:
- **RT-1(压缩 FSQ,256×320)**:LPIPS 随 horizon 显著上升(有 headroom);
- iVideoGPT/RoboNet(64×64 轻压缩):近天花板、平。
一图说清:① RT-1 长序列质量确实衰减;② 不同平台的可改善空间不同。由于平台同时改变了
tokenizer、分辨率、数据和模型，图中不把差异单独归因于压缩强度。

**Fig 5 — C2 主效应与 GAE 辅助消融。** ✅ 数据已有
(a) **逐 horizon C2 增益**：正式 K=16/low-var-KL 协议下绘制 return−seq 的 LPIPS delta；只在
置信区间和方向确实支持时讨论后段效应。
(b) **λ 扫描（辅助）**：K=8 内部评测中，GAE0.7 相对 return 在 LPIPS、LPIPS-last、MSE 上
3/3 改善，但相对 sequence baseline 的平均 LPIPS 为 +0.00325（更差）。因此该子图只说明
GAE 可缓解纯 return 的一部分方差，不把 GAE0.7 标成全局最优或 headline 方法。

**Table 2 — C2 主结果(RT-1,配对种子)。** ✅
正式主表保持 seq / Temporal Return 的 K=16、low-var-KL 配对比较；GAE0.7 放在辅助消融表，
不得与不同 K 的正式结果混成一个排名表。

## C. 主观图（对标 RLVR-World Fig. 6，但采用可复核选择协议）

**Fig 6 — 递增 horizon 的定性对比。** 🔜 需从已有 checkpoint 生成帧
主文放两个场景，补充材料放另外两个场景。每个场景采用下列布局：
```
              t=2      t=4      t=6      t=7(末)
Ground Truth  [img]    [img]    [img]    [img]
Base          [img]    [img]    [img]    [img]
RLVR-World    [img]    [img]    [img]    [img]
Ours          [img]    [img]    [img]    [img]
```
- 场景选择只使用 GT 轨迹：先在 episode-disjoint 测试集上按运动量和初始帧视觉差异生成 shortlist，
  冻结 `scene_manifest.json` 后才加载任何模型；禁止依据 ours-vs-baseline 的效果挑 scene。
- 若自动 shortlist 未覆盖桌面、抽屉等语义场景，可在只含 GT 的 contact sheet 上按行号预先冻结
  4 个场景；行号、顺序和选择规则写入 manifest，仍不得查看任何模型输出后重选。
- 每个模型固定 `K=1`、candidate index 0 和同一 generation seed；禁止按 GT 指标做 best-of-K。
- 训练模型选全局 held-out LPIPS 的中位 seed，并在 manifest 中记录选择来源；禁止使用最好 seed。
- 主图不靠局部归一化的误差热图制造视觉差异。补充材料可给 `|pred-GT|`，但所有模型共享同一色标。
- 主要观察后段帧，并与 Fig. 4/5 的逐 horizon 定量结果互相校验；若某场景 ours 不优，也原样保留。

**数据来源**：base、官方 RLVR 及冻结后的 ours checkpoint。导出脚本必须同时保存原始帧、生成参数、
checkpoint hash、scene manifest 和逐 horizon 指标。

## D. "多场景"如何满足(诚实,不靠单数据集)

- **C1 的多场景 = 两层跨平台证据**：Fig. 2 在同一批 RT-1 帧上比较 CNN-FSQ、压缩 FSQ 与
  Cosmos DV-FSQ 的编解码重建误差；Table 1 在 RT-1 single/multi 与 iVideoGPT-VP2 的真实候选组上验证
  排序修复。前者证明问题跨 codec，后者证明后果跨世界模型实现。
- **C2 的多场景 = 跨 horizon + 平台 headroom 谱**:Fig 4 对照多个平台,RT-1 在有 headroom 端;
  诚实说明 64×64 轻压缩平台近天花板无空间(即时监督充分)。
- **主观图多场景 = 多 RT-1 episode**:Fig 6 给 3-4 个不同 scene。

## E. 待生成清单

- 🔜 Fig 6 定性帧：重写当前仅支持单步的 `export_frames.py`，先生成 GT-only shortlist，再从冻结的
  scene manifest 导出多步对齐帧（4 scenes × 递增 horizon）；
- 🔜 Fig 2 重建残差定性帧：统一导出 `GT / reachable target / residual`。Cosmos 承担独立 FSQ codec
  的外部重建误差证据；iVideoGPT-VP2 保留在排序表，不声称未观察到的生成训练增益；主线平台 = RT-1 + VP2 + Cosmos（仅重建误差审计）；
- 🔜 Fig 2 柱状图 + Fig 4/5 曲线:数据已有(headroom_curves/、λ 扫描、floor 审计),画图即可;
- ✅ Table 1/2、Fig 1/3:数据已有。
