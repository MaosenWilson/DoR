# RC-GRPO 论文数据统筹与展示 shortlist

**日期：** 2026-07-19  
**原则：** real-only；不混协议；不在 null-C2 平台做「改进主观图」  
**本地数据根：** `tmp/paper_figures_data/`  
**服务器全量记录镜像：** `tmp/server_records_20260718/`（JSON/配置；不含权重）  
**筛选 skill：** `paper-result-presentation` v2.1  

---

## 0. 一页结论（投稿主张上限）

| 贡献 | 可写强度 | Headline 证据 | 禁止写 |
|---|---|---|---|
| **C1 可达性 / 重建残差改变组内排序（CE）** | **强** | 跨 FSQ 不可忽略的编解码重建误差 + 跨架构 Δρ↑/Δflip↓ | 跨平台 raw-GT 训练普适增益；Cosmos 有排序实验 |
| **C2 帧块时间回报** | **趋势可写、措辞保守** | RT-1 E1：return_rc 最优；LPIPS −1.39% vs seq_rc（4/5）；MSE 5/5 | 时间对应因果（缺 E2）；VP2/iVideoGPT 有生成改进 |
| **主观改进图** | **仅 RT-1 多 episode** | GT / RLVR ckpt / Ours，偏后段 t | iVideoGPT base vs ours「更好看」挑帧 |
| **C1×C2 超加** | **omit** | interaction 跨零 | 第三贡献 |

**主文数字锚点（E1 episode-disjoint，window_macro，manifest `fb22f86c…`）：**

- return_rc vs seq_rc：LPIPS **−1.39%**（4/5，p=0.125）；LPIPS-last **−1.93%**（4/5）；MSE **−1.91%**（**5/5**，p=0.074）
- 同协议锚点：base LPIPS 0.2127 / RLVR 0.2040 / **return_rc 0.1966**

**Formal 四臂（K=16，low_var_kl，in-training 8-window）分表支持，勿与 E1 混行。**

---

## 1. 本地数据分类清单

| 目录 | 用途 | 图/表 | 文件量级 |
|---|---|---|---|
| `C1_rank_floor/` | 翻转律、重建误差、RT-1 rank audit | Fig1–3, T1 | 11 |
| `C1_ra_rc/` | RA-RC pilot（附录） | appendix | 13 |
| `C2_main/` | 正式 factorial + E1 + base/rlvr | T2, Fig5b | 25 |
| `C2_horizon_headroom/` | per-horizon 退化 / headroom | Fig4 | 9 |
| `C2_gae_lambda/` | λ/GAE 扫描（K=8 辅助） | Fig5a | 117 |
| `external_rank/` | VP2 rank95（主线） | T1 | 主用 vp2_* |
| `external_c2_boundary/` | RoboDesk/RoboSuite null-C2 | Fig4 边界, limit | 73 |
| `analysis_support/analysis_all/` | 其余 analysis JSON | 备查 | 83 |
| `images/` | 已有小图 | 视情况 | 13 |
| `checkpoints_index/` | 服务器 ckpt 路径索引 | Fig6 生成用 | 1 |

**rsync 策略：** 只拉 JSON/CSV/md/小图；排除 `*.safetensors *.pt *.bin` 与大 npz。权重留 AutoDL。

---

## 2. 图表 ↔ 数据 ↔ 角色

### A. C1（重建残差改变 GRPO 组内排序 / CE）

| ID | 内容 | 数据路径 | 角色 | 状态 |
|---|---|---|---|---|
| **Fig1** | 实测 flip vs `arccos(ρ)/π`，y=x | `C1_rank_floor/reward_spaces_s{0..4}_rankflip.json` | headline 机制 | ✅ 可画 |
| **Fig2** | FSQ 家族重建误差柱 + 定性 GT/recon/residual | 标量：见 §3.1；定性帧 🔜 | headline 泛化 | 标量部分可用；Cosmos JSON 需冻结 |
| **Table1** | 跨架构同候选审计 | `rank_calibration_audit.json` + `external_rank/*_rank95.json` | headline 表 | ✅ |
| **Fig3** | raw→RC：ρ↑ flip↓ | 同上 paired | headline 闭环 | ✅ |
| **重建残差定性** | iVideoGPT/RT-1：GT vs reachable vs raw | 🔜 export；**不做改进对比** | floor-qual | 待生成 |

### B. C2（长序列时间信用）

| ID | 内容 | 数据路径 | 角色 | 状态 |
|---|---|---|---|---|
| **Fig4** | per-horizon LPIPS：RT-1 上升 vs 64×64 近天花板 | `C2_horizon_headroom/msp_horizon_*.csv` + `headroom*.json` + `external_c2_boundary/**/headroom*.json` | headline 长序列 | ✅ 可画 |
| **Fig5a** | λ 扫描（K=8）；标 gae0.7 仅相对 return 内部更优 | `C2_gae_lambda/msp_lam_*` | support / 辅助 | ✅；**非 headline 主方法** |
| **Fig5b** | 逐 horizon C2 增益（return−seq） | E1 / horizon fullmetrics | support 签名 | ✅ 可算 |
| **Table2** | 主结果 paired | `C2_main/msp_episode_disjoint_*` + formal factorial | headline 表 | ✅ |

### C. 主观图

| ID | 内容 | 规则 | 状态 |
|---|---|---|---|
| **Fig6** | RT-1：GT / RLVR / Ours，t=2,4,6,7；3–4 episode | 先冻 GT-only scene manifest；中位 seed；K=1 deterministic；禁止 null 平台假改进 | 🔜 GPU 导出 |

---

## 3. Claim → 数字 map（主文可用）

### 3.1 C1 编解码重建误差（encode–decode reconstruction residual）

| codec | LPIPS floor（约） | 来源 | strength |
|---|---:|---|---|
| CNN-FSQ（single） | **0.0518** | `C1_rank_floor/rank_preservation.json#phi_tok_lpips` | supported |
| 压缩 FSQ（multi，叙事 ~0.077） | ~0.077 | cosmos 脚本对照注释 + 叙事；**freeze 前再核一次 matched-frame JSON** | trend / 待核 |
| Cosmos DV-FSQ | **0.190** | `narrative_fsq_generalization.md` / 服务器审计叙述；**建议补存 `cosmos_floor_audit.json`** | supported（叙述链）勿无 JSON 上主表到 camera-ready |
| `floor_metrics.json` LPIPS floor_mean | 0.112 | 另一套 drowned 分析（n=184），**不要与上表三件套静默混用** | appendix 另注 |

**写法：** 「三种独立 FSQ 实例均不可忽略的编解码重建误差」；**不要**在无 bpp 控制时把柱高写成严格压缩强度单调定律（figure_plan 已收紧）。

### 3.2 C1 排序修复（同候选 raw vs RC）

| 设置 | Δρ (RC−raw) | Δflip | n/备注 | path | strength |
|---|---:|---:|---|---|---|
| RT-1 single CNN-FSQ | **+0.0168** CI[0.0035,0.0289] | −0.0082 | 148 groups / 20 ep | `rank_calibration_audit.json#single_cnnfsq` | supported |
| RT-1 multi compressive | **+0.0145** CI[0.0105,0.0176] | **−0.0087** | 1152 groups | `#multi_compressive_all` | supported |
| VP2 H2 | **+0.0932** CI95[0.067,0.122] | −0.0409 | 64 clusters | `external_rank/vp2_h2_rank95.json` | supported GREEN |
| VP2 H8 | **+0.0753** CI95[0.041,0.110] | −0.0346 | 8 clusters | `vp2_h8_rank95.json` | supported GREEN |

**Cosmos：** 只进 Fig2 重建误差审计，**不进 Table1 排序列**（无匹配候选世界模型）。

### 3.3 C2 主结果 — E1（推荐主表）

协议：episode-disjoint；aggregation=window_macro；seeds 0–4；对比 **return_rc − seq_rc**。

| metric | seq_rc | return_rc | Δ mean | wins | two-sided p | rel % | strength |
|---|---:|---:|---:|---:|---:|---:|---|
| LPIPS ↓ | 0.19933 | **0.19657** | −0.00276 | 4/5 | 0.125 | **−1.39%** | trend headline |
| LPIPS-last ↓ | 0.21176 | **0.20767** | −0.00409 | 4/5 | 0.149 | **−1.93%** | support |
| MSE ↓ | 0.013820 | **0.013556** | −0.000264 | **5/5** | 0.074 | **−1.91%** | strongest training |
| PSNR ↑ | 19.382 | **19.447** | +0.065 | 4/5 | 0.155 | — | support |
| SSIM ↑ | 0.75577 | **0.75908** | +0.00331 | 4/5 | 0.175 | — | support |

源：`C2_main/msp_episode_disjoint_eval_summary.json` + per-seed `*_s{0..4}.json`。

### 3.4 C2 Formal 四臂（分表，K=16 low_var_kl）

| arm | LPIPS | LPIPS-last | MSE |
|---|---:|---:|---:|
| seq_raw | 0.20224 | 0.21278 | 0.013109 |
| seq_rc | 0.20214 | 0.21141 | 0.013178 |
| return_raw | 0.20158 | 0.21151 | 0.013287 |
| **return_rc** | **0.19935** | **0.20775** | **0.012756** |

return_rc − seq_rc（LPIPS）：mean **−0.00279**，**5/5**，t=−2.32，p=0.081，boot95 [−0.0051,−0.0012]。  
源：`C2_main/msp_lvkl_factorial_s0_4.json`。

### 3.5 外部锚点（E1 同 manifest，勿称完整 recipe 复现）

| ckpt | LPIPS | LPIPS-last | MSE |
|---|---:|---:|---:|
| base | 0.21273 | 0.22783 | 0.01548 |
| RLVR official | 0.20403 | 0.21207 | 0.01439 |
| **return_rc** | **0.19657** | **0.20767** | **0.01356** |

相对 base ≈ **−7.6%** LPIPS；相对 RLVR ckpt ≈ **−3.7%**（仅同 held-out 协议）。

### 3.6 GAE λ 扫描（K=8，辅助；非 headline）

final step 均值（n=3 seeds）：

| arm | LPIPS | LPIPS-last | MSE |
|---|---:|---:|---:|
| seq | **0.20256** | 0.21237 | 0.01320 |
| return | 0.20802 | 0.21624 | 0.01408 |
| **gae_0.7** | 0.20581 | 0.21409 | 0.01375 |
| gae_0.5/0.6/0.9 | 更差或接近 | — | — |

- gae0.7 − return：LPIPS/last/MSE 均 **3/3 改善**  
- gae0.7 − seq：LPIPS mean **+0.00325（更差）**  
→ **只写「缓解 pure return 方差」；不写全局最优 / 不与 K=16 主表混排。**

### 3.7 Headroom / null-C2（Fig4 边界）

| 平台 | 信号 | path | 用途 |
|---|---|---|---|
| RT-1 | 有 headroom（horizon LPIPS 上升） | `msp_horizon_*.csv` | C2 主战场 |
| RoboSuite headroom | verdict NEAR-CEILING | `external_c2_boundary/robosuite/headroom.json` | Fig4 对照 / limit |
| RoboDesk h15 | LPIPS 量级 ~0.005–0.011 | `.../robodesk/headroom_h15.json` | 近天花板 |
| VP2 C2 训练 | null | external sweeps | **limit**；禁止改进主观图 |

### 3.8 RA-RC pilot（附录 only）

- vs raw LPIPS −0.00117，3/3；vs pure RC −0.00060，2/3，CI 跨 0  
- path：`C1_ra_rc/ra_rc_rt1_single_pilot.json`  
- **不能**写完整 C1 训练门「双赢已闭合」

---

## 4. 选择规则（本轮）

1. Real-only：上表每个 headline 数都有 path#field。  
2. Checkpoint：final after 30 updates；非 best-seed / 未表注 best-step。  
3. Seeds：C2 用完整 paired 0–4；GAE/RA 披露 n=3。  
4. Primary：full-rollout LPIPS；last/MSE 为 support（MSE 可强调 5/5）。  
5. 协议分表：E1 ≠ formal 8-window ≠ GAE K=8。  
6. KL：headline 仅 `low_var_kl`。  
7. 定性：改进图 ⊆ 定量有 headroom 的 RT-1；iVideoGPT → 重建残差/CE 图。  
8. 「最漂亮」= 真实池内叙事最清晰配置（return_rc），不是造数。

---

## 5. Cannot write / Limitations

1. RA-RC 已证明同时优于 raw 与 pure RC（缺多平台 + 相对 RC 不稳）。  
2. Temporal Return 收益**因为**正确时间对应（缺 E2 shuffled）。  
3. 收益随 horizon 单调增强已严格证明（缺 E3 / 需 Fig5b 与 CI 对齐后再写）。  
4. RC×Return 统计超加 / C3。  
5. 击败 RLVR-World **完整训练 recipe**。  
6. iVideoGPT/VP2 上 base vs ours 像素级可见改进（定量 null）。  
7. Cosmos 进入排序主表。

---

## 6. Fig6 / 多场景执行清单（下一步 GPU）

1. GT-only shortlist → 冻结 `scene_manifest.json`（桌面/抽屉/抓取/推动等语义，**不看模型输出**）。  
2. ckpt：base / 官方 RLVR / return_rc **中位 seed**（按 held-out LPIPS）。  
3. `export_frames.py`：`--deterministic`，同 context seed，K=1。  
4. 主文 2 scene，附录 +2；后段 t 对齐 Fig4/5。  
5. 可选 `|pred−GT|` 统一色标。  
6. iVideoGPT 帧另做 **floor-qual**，永不进「Ours 更清晰」格。

---

## 7. 与写作目标对齐（本轮）

- **C1/CE：** GPT 系（iVideoGPT）+ RT-1 tokenizer + Cosmos 重建误差 → 展示重建残差对排序的干扰与排序校准，不是假改进。  
- **C2：** 深度在 RT-1；长序列退化 + Temporal Return。  
- **主观图：** 对标 RLVR Fig6 版式，场景用多 RT-1 episode。  
- **诚实多场景：** 改进多样性 = RT-1 任务；generality = 跨 tokenizer/架构。

---

## 8. 文件索引（复制用）

```
tmp/paper_figures_data/C1_rank_floor/rank_calibration_audit.json
tmp/paper_figures_data/C1_rank_floor/reward_spaces_s*_rankflip.json
tmp/paper_figures_data/C1_rank_floor/rank_preservation.json
tmp/paper_figures_data/C1_rank_floor/floor_metrics.json
tmp/paper_figures_data/external_rank/vp2_h2_rank95.json
tmp/paper_figures_data/external_rank/vp2_h8_rank95.json
tmp/paper_figures_data/C2_main/msp_episode_disjoint_eval_summary.json
tmp/paper_figures_data/C2_main/msp_lvkl_factorial_s0_4.json
tmp/paper_figures_data/C2_main/{base,rlvr}_eval.json
tmp/paper_figures_data/C2_horizon_headroom/msp_horizon_fullmetrics.csv
tmp/paper_figures_data/C2_gae_lambda/msp_lam_{seq,return,gae_0.7}/
tmp/paper_figures_data/external_c2_boundary/{robodesk,robosuite}/
docs/aaai2027/figure_plan.md
docs/aaai2027/rt1_presentation_shortlist.md
docs/aaai2027/abstract_last_paragraph_v3.md
```
