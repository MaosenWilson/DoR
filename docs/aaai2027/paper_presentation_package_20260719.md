# RC-GRPO 论文结果统筹包（Presentation Package）

**日期：** 2026-07-19  
**服务器：** `ssh -p 25017 root@connect.westd.seetacloud.com` → `/root/autodl-tmp/vote2world`  
**本地分类镜像：** `tmp/paper_figures_data/`  
**技能：** `paper-result-presentation` v3（只选真实 artifact，不改数、不静默混协议）  
**写作目标对齐：** C1 重建残差对排序的干扰+排序校准（3 图 1 表）· C2 长序列信用 · 诚实主观图（多 RT-1 场景）· iVideoGPT/null 平台只进 CE/重建残差

---

## 0. 一句话方案（先看这个）

| 板块 | 主文放什么 | 为什么最漂亮（真实池内） |
|---|---|---|
| **C1** | Fig1 翻转律 + Fig2 跨 codec 重建误差 + T1 跨架构排序 + Fig3 raw→RC 闭环 | 机制可复述、跨 tokenizer/架构 GREEN、与「CE/重建残差对排序的干扰」贡献句同向 |
| **C2** | Fig4 per-horizon 退化 + Fig5(a) λ 辅助/(b) 后段增益签名 + T2 `return_rc` 四臂 | RT-1 有 headroom；`return_rc` 正式协议四臂最优、5/5 同向 |
| **Qual** | Fig6 **仅 RT-1** 多 episode 后段网格（GT / base / RLVR / Ours） | 与 Fig4/5 后段签名一致；**禁止** iVideoGPT 改进对比 |
| **CE 场景扩展** | iVideoGPT / Cosmos / 抽屉等 → **重建残差/可达目标** 并排，不是 base vs ours | 场景丰富且经得起复测 |

**Claim 强度默认：** **conservative / trend**（C2 full LPIPS formal p≈0.081）；C1 rank **supported**。

**旧摘要数字问题：**「相对序列级 GRPO 降 4.9% / 7.7% / 9.2%」**与当前 formal/E1 artifact 不符**，已在 §9 用可回溯数字替换。

---

## 1. 本地数据目录（分类拷贝完成）

| 本地目录 | 角色 | 主要内容 |
|---|---|---|
| `tmp/paper_figures_data/C1_rank_floor/` | C1 机制+重建误差 | `rank_calibration_audit.json`, `reward_spaces_s*_rankflip.json`, `floor_metrics.json`, horizon rank guard |
| `tmp/paper_figures_data/C1_ra_rc/` | C1 训练 pilot | `ra_rc_rt1_single_pilot.json`, VP2 smoke |
| `tmp/paper_figures_data/C2_main/` | C2 headline | `msp_lvkl_factorial_s0_4.json`, `msp_episode_disjoint_eval_summary.json`, `msp_episode_disjoint_eval/*` |
| `tmp/paper_figures_data/C2_gae_lambda/` | Fig5a 辅助 | `msp_lam_{seq,return,gae_*}/sweep_*.json`（K=8 协议） |
| `tmp/paper_figures_data/C2_horizon_headroom/` | Fig4 | `msp_horizon_*.json/csv`, `rt1_horizon_h15.*`, `prop_*.json` |
| `tmp/paper_figures_data/external_rank/` | T1 扩展行 | `vp2_h{2,8}_rank95.json`（主线） |
| `tmp/paper_figures_data/external_c2_boundary/` | limit | VP2 training / RCTR 边界 |
| `tmp/paper_figures_data/analysis_support/` | appendix | n10/n5 recheck, coupling |
| `tmp/paper_figures_data/_server_analysis_sync/` | 全量 analysis 镜像 | 2026-07-19 rsync，JSON/CSV only |
| `tmp/paper_figures_data/checkpoints_index/` | 权重索引 only | 服务器 ckpt 路径列表，**不拉权重** |
| `tmp/server_records_20260718/` | 前一日归档 | tar + FILES.txt（可对照完整性） |

**rsync 策略：** exclude `*.safetensors` / `*.pt` / `*.bin`；断线后用本地镜像继续写本 MD。

---

## 2. Comparison axes

| claim_id | 贡献句（展示层） | primary | 方向 | 主对照 | 推断单位 |
|---|---|---|---|---|---|
| C1-floor | FSQ 家族存在不可忽略的编解码重建误差（codec 固有） | recon floor LPIPS | 存在/非零 | encode–decode vs raw | matched frames |
| C1-flip | 实测 flip ≈ arccos(ρ)/π | flip vs theory | y=x | 多 reward space | seed / group |
| C1-rank | RC 提高 ρ、降低 flip（跨架构） | Δρ ↑, Δflip ↓ | 修复 | raw vs RC 同候选 | episode cluster / boot |
| C1-train | RA-RC 训练增益 | raw LPIPS | ↓ | raw / pure RC | pilot seeds（弱） |
| C2-main | Temporal Return 改善多步 fidelity | full LPIPS | ↓ | **seq_rc** | formal paired 0–4 |
| C2-tail | 末帧 / 后段增益（支持） | LPIPS-last / per-h Δ | ↓ | seq_rc | 同上 |
| C2-scope | 仅 headroom 平台可训练改进 | headroom curve | — | 多平台叠加 | platform |
| Qual-imp | 主观改进图 | 与 C2 同向后段 | — | base/RLVR/Ours | **RT-1 only** |
| Qual-floor | 重建残差对排序的干扰主观 | GT / reachable / raw | — | 跨 codec/平台 | CE 环节 |

---

## 3. Headline 数字（Real-only，已核对 JSON）

### 3.1 C1 — Rank repair（Table 1 候选）

| 平台 | metric | value | n / 备注 | source | role | strength |
|---|---|---|---|---|---|---|
| RT-1 single CNN-FSQ | Δρ | **+0.0168** [0.0035, 0.0289] | 148 groups / 20 ep | `C1_rank_floor/rank_calibration_audit.json#single_cnnfsq` | headline | **supported** |
| RT-1 single | Δflip | **−0.0082** | 同上 | 同上 | headline | supported |
| RT-1 multi compressive | Δρ | **+0.0145** [0.0105, 0.0176] | 1152 groups | `#multi_compressive_all` | headline | **supported** |
| RT-1 multi | Δflip | **−0.0087** CI 不含 0 | 同上 | 同上 | headline | **supported** |
| VP2 H2 | Δρ (Spearman) | **+0.0932** [0.067, 0.122] | cluster boot | `external_rank/vp2_h2_rank95.json` | headline | **supported** |
| VP2 H2 | Δflip | **−0.0409** | 同上 | 同上 | headline | supported |
| VP2 H8 | Δρ | **+0.0753** [0.041, 0.110] | 同上 | `vp2_h8_rank95.json` | support | supported |
| 全表 | verdict | **GREEN** | 机制门 | 各 `verdict` 字段 | — | — |

**Fig1 数据：** `reward_spaces_s{0..4}_rankflip.json`（实测 flip vs 理论）。

**Fig2 重建误差标量（experiments.md 冻结读数；投稿前再读 artifact 核验）：**

| codec | floor LPIPS | 用途 |
|---|---|---|
| CNN-FSQ | ~0.053 | Fig2 柱 |
| compressive FSQ | ~0.077 | Fig2 柱 |
| Cosmos DV-FSQ | **0.190** | Fig2 柱；**仅报告重建误差，不进排序表** |

`floor_metrics.json` 为另一协议聚合（LPIPS floor_mean≈0.112），**勿与上表三 codec 柱静默混用**；画 Fig2 时表注 codec 与评价骨干。

### 3.2 C2 — Formal factorial（Table 2 headline）

**协议：** T=8, K=16, steps=30, `low_var_kl`, γ=0.95, eval 8-window, seeds 0–4, final ckpt  
**源：** `C2_main/msp_lvkl_factorial_s0_4.json`

| arm | LPIPS ↓ | LPIPS-last ↓ | MSE ↓ |
|---|---:|---:|---:|
| sequence + raw | 0.20224 | 0.21278 | 0.013109 |
| sequence + RC | 0.20214 | 0.21141 | 0.013178 |
| Temporal Return + raw | 0.20158 | 0.21151 | 0.013287 |
| **Temporal Return + RC** | **0.19935** | **0.20775** | **0.012756** |

**主效应 `return_rc − seq_rc`（paired）：**

| metric | mean Δ | rel % vs seq_rc | wins | t | two-sided p | boot 95% | strength |
|---|---:|---:|---|---:|---:|---|---|
| LPIPS | **−0.00279** | **−1.38%** | **5/5** | −2.32 | **0.081** | [−0.0051, −0.0012] | **trend** |
| LPIPS-last | **−0.00366** | **−1.73%** | 5/5 | −3.11 | **0.036** | [−0.0057, −0.0016] | **supported**（副指标） |
| MSE | **−0.00042** | **−3.20%** | — | — | — | — | support |

**interaction (RC×Return) LPIPS：** mean −0.00214, p≈0.59 → **omit as C3 / 超加性**。

### 3.3 C2 — E1 episode-disjoint（support，勿与 formal 混行）

**源：** `C2_main/msp_episode_disjoint_eval_summary.json` + `.../base_eval.json` + `rlvr_eval.json`  
**聚合：** window_macro；manifest `fb22f86c…`

| arm | LPIPS ↓ | LPIPS-last ↓ | MSE ↓ |
|---|---:|---:|---:|
| seq_raw | 0.19915 | 0.21180 | 0.013679 |
| seq_rc | 0.19933 | 0.21176 | 0.013820 |
| return_raw | 0.20118 | 0.21487 | 0.013941 |
| **return_rc** | **0.19657** | **0.20767** | **0.013556** |
| base（锚点） | 0.21273 | 0.22783 | 0.015480 |
| rlvr 官方 ckpt（锚点） | 0.20403 | 0.21207 | 0.014390 |

| 对比 | LPIPS Δ | rel % | wins | p | strength |
|---|---:|---:|---|---:|---|
| return_rc − seq_rc | −0.00276 | −1.39% | 4/5 | 0.125 | **trend** |
| return_rc − base | −0.01616 | **−7.60%** | 锚点 | — | support（**非** formal 主对照） |
| return_rc − rlvr | −0.00746 | **−3.66%** | 锚点 | — | support（同协议 readout） |

### 3.4 Fig5a — GAE λ（辅助 only，K=8）

来自 `tmp/server_records_20260718/README.md` 重算（3 seeds）：

| Arm | LPIPS | vs Return | vs Sequence |
|---|---:|---|---|
| Sequence | 0.20256 | — | — |
| Return | 0.20802 | — | 更差 |
| GAE 0.7 | 0.20581 | 改善 | **仍差于 Sequence（+0.00325）** |

→ **support-ablation**；**不得**写成 headline「全局最优」。Fig6 若用 gae0.7 帧，caption 必须写「K=8 辅助 ckpt」或改用 formal `return_rc` median seed（推荐后者）。

---

## 4. Display roles（图/表）

| ID | 内容 | role | 数据就绪 | 备注 |
|---|---|---|---|---|
| **Fig1** | flip vs arccos(ρ)/π，y=x | **headline-mech** | ✅ rankflip JSON | 多 reward space |
| **Fig2** | FSQ 家族重建误差柱 + 可选 GT/reachable/residual 帧 | **headline-floor** | ✅ 标量；帧 🔜 | Cosmos 仅做重建误差审计；含 iVideoGPT 场景仅作 CE |
| **Fig3** | raw→RC ρ↑ flip↓ 闭环 | **headline** | ✅ rank audit | 同候选配对 |
| **Fig4** | multi-platform per-horizon LPIPS | **headline-C2** | ✅ horizon JSON | RT-1 上升 vs 64×64 近天花板 |
| **Fig5a** | λ / GAE 扫描 | **support** | ✅ | 标 K=8；不升格 |
| **Fig5b** | per-horizon C2 gain | **headline-sign** | ✅ E1/horizon | 后段更负 = 定量签名 |
| **T2** | formal 四臂 + return−seq | **headline-C2** | ✅ | primary=full LPIPS |
| **T2s** | E1 + base/rlvr | **support** | ✅ | 分表 |
| **Fig6** | 多 RT-1 episode 后段网格 | **headline-qual** | 🔜 需 GPU 出帧 | 见 §5 |
| VP2/iVideoGPT 训练增益 | — | **limit / cannot-qual** | ✅ null | **禁止**改进主观图 |
| RA-RC pilot | appendix | **appendix** | ✅ | 未证明优于 pure RC |
| Super-additivity | omit | **omit** | ✅ | p≈0.59 |

---

## 5. 主观图 / 多场景诚实合同

### 5.1 改进图（Fig6）— 仅 RT-1

```
              t=2      t=4      t=6      t=7
GT            ...
Base          ...
RLVR-World    ...      ← 后段重复/模糊（若有）
Ours          ...      ← formal return_rc 中位 seed（推荐）
```

| 规则 | 要求 |
|---|---|
| 平台门控 | 仅 RT-1（定量有 headroom + Δ 可见） |
| 多场景 | **3–4 个不同 RT-1 episode**（抓取/推动/放置/抽屉等语义，由 **GT-only** shortlist 冻结） |
| 禁止 | 按 ours 好看挑 scene；iVideoGPT base vs ours；跨 episode 拼贴 |
| 时间焦点 | 后段 t，呼应 Fig4/5 |
| 对齐 | 同 context、同 seed、`--deterministic`、K=1 |
| seed 选择 | held-out LPIPS **中位 seed**，写入 `scene_manifest.json` |
| 热图 | 可选 `|pred−GT|` 共用色标，主文可不放 |

**ckpt 索引（服务器，不拉权重）：** 见 `checkpoints_index/server_ckpt_paths_20260719.txt`；E1 中 base/rlvr 路径已写在 eval JSON。

### 5.2 重建残差 / CE 图（可含 iVideoGPT、Cosmos、RoboDesk 抽屉）

并排：`GT | decode(encode(GT)) | raw candidate | residual`  
**只支撑 C1「重建残差改变组内排序」**，文案禁止 “Ours improves generation on iVideoGPT”。

### 5.3 Qual roles

| 帧类 | role |
|---|---|
| RT-1 多 episode 后段改进网格 | `headline-qual` |
| 同 claim 额外 episode | `support-qual` |
| iVideoGPT/Cosmos 可达重建残差 | `floor-qual` |
| iVideoGPT base vs ours | `cannot-qual` |

---

## 6. Paper preview（展示预估）

### Main tables

**T1 Rank calibration（draft）**

| Setting | Δρ | Δflip | Status |
|---|---:|---:|---|
| RT-1 CNN-FSQ single | +0.0168 | −0.0082 | GREEN |
| RT-1 compressive multi | +0.0145 | −0.0087 | GREEN |
| iVideoGPT-VP2 H2 | +0.093 | −0.041 | GREEN |
| iVideoGPT-VP2 H8 | +0.075 | −0.035 | GREEN |
| Cosmos DV-FSQ | — | — | **floor-only（TBD row not for rank）** |

**T2 Multistep RT-1 formal（draft）** — 用 §3.2 四臂表；脚注：paired n=5, final, `low_var_kl`.

### Contribution sentences

**Strong（边界内最大）：**  
跨 RT-1 / VP2 的同候选审计中，可达目标重评分一致提高排序相关并降低翻转；在 RT-1 正式 `low_var_kl` 多步协议下，Temporal Return+RC 相对 sequence+RC 在 full LPIPS 上 5/5 种子同向改善（均值 −1.4%，p=0.081），末帧 LPIPS 改善达 p=0.036。

**Conservative（推荐摘要）：**  
同候选可达性审计表明，RC 在多种 tokenizer 与世界模型上修复 GRPO 排序信号；在固定 RT-1 多步协议与配对种子下，帧块时间回报相对序列级 RC 一致降低完整序列与末帧 LPIPS（正式评测均值分别约 −1.4% / −1.7%），统计显著性因样本量为趋势级证据。近天花板平台不作生成改进主张。

### Gaps

| 缺口 | 影响 | 动作 |
|---|---|---|
| Fig6 对齐帧未生成 | 缺 headline-qual | 服务器跑 `export_frames.py` / multistep export |
| Fig2 重建残差定性帧 | 版式欠丰 | 导出 GT/reachable/residual |
| Cosmos 三 codec 柱 artifact 再读 | Fig2 数需冻结 | 从原始 cosmos audit 再 dump 一次 |
| E2 shuffled 因果 | 不能写时间对应因果 | 补跑或 limit |
| formal p=0.081 | 摘要勿写 “significantly” | 用 consistent / reduces |

---

## 7. Selection rules this round

1. **Real-only** — 数字均有 path#field；无插值。  
2. **Checkpoint** — formal/E1 用 final after 30 updates；非 best-seed。  
3. **Seeds** — C2 n=5 paired 0–4；GAE n=3 披露。  
4. **Primary** — full-rollout LPIPS；last 不偷换 primary。  
5. **协议分表** — formal 8-win ≠ E1 32-win ≠ GAE K=8。  
6. **KL** — headline 仅 `low_var_kl`。  
7. **Qual 门控** — 改进图 ⊆ 定量 headroom 平台（RT-1）；null → floor-only。  
8. **最漂亮** = 叙事清晰 + 可比 + 效果可见，**不是**编造更小数。

---

## 8. Appendix / limit / cannot-write

| 类 | 内容 |
|---|---|
| **Appendix** | RA-RC pilot；GAE 全 λ；E1 per-seed；旧 linear-KL |
| **Limit** | VP2/iVideoGPT C2 训练 null；跨平台 raw-GT 转换未成立；RA 未稳定优于 pure RC |
| **Cannot** | 超加性 C3；E2 因果；iVideoGPT 改进主观图；把 GAE0.7 当正式最优；摘要 4.9%/7.7%/9.2% |
| **Omit** | 噪声 smoke、与贡献无关的 grpo_v* 探索 |

---

## 9. 摘要最后一段 — 修订稿

### 原文（问题）

> 跨 RT-1、VP2-RoboSuite 的冻结候选审计……（**旧百分比 4.9%/7.7%/9.2% 与 formal/E1 不符，已废弃**）

问题：相对 **seq_rc** 的 formal 真实读数为约 **−1.4% / −1.7% / −3.2%**；4.9% 等无法在当前 formal/E1 主 artifact 中复现。若相对 base 的 −7.6% 亦未对应三指标那组百分数。

### 推荐替换（conservative，对齐本统筹包）

> 跨 RT-1（CNN-FSQ 与压缩 FSQ）与 iVideoGPT-VP2 的**冻结同候选**审计表明，将比较目标换为 codec 可达重建后，组内排序与解码器输入表示误差更一致，候选对翻转减少；独立 codec 审计进一步显示 CNN-FSQ、压缩 FSQ 与 Cosmos DV-FSQ 均存在不可忽略的编解码重建误差，支撑「重建残差改变组内排序」的一般性，而非单一数据集假象。在固定 RT-1 多步、`low_var_kl`、五种子配对协议下，帧块时间回报与 RC 组合（Temporal Return+RC）相对序列级 RC，使完整序列 LPIPS 与末帧 LPIPS 分别降低约 **1.4%** 与 **1.7%**（5/5 种子同向；完整序列双侧 p=0.081，末帧 p=0.036），MSE 约降 **3.2%**。该方法的生成改进主张限定在具有可测 headroom 的长序列设置；近天花板平台用于重建残差与排序诊断，而不报告虚假的 base–ours 观感提升。

### 备选更短版

> 同候选可达性审计在 RT-1 与 VP2 上一致显示 RC 提高排序相关并降低翻转；跨 FSQ 家族的编解码重建误差审计支撑目标侧腐蚀的一般性。固定 RT-1 多步协议下，Temporal Return+RC 相对序列级 RC 以五种子同向降低完整序列与末帧 LPIPS（约 1.4% / 1.7%）。改进评价与定性展示均限定在有 headroom 的设置，重建误差诊断平台只用于 CE 诊断。

---

## 10. 下一步（按优先级）

1. **GPU：** Fig6 — GT-only scene manifest → 对齐导出 base / rlvr / return_rc（中位 seed）多 episode 后段帧  
2. **本地画图：** Fig1–5、T1–T2（数据已在 `tmp/paper_figures_data/`）  
3. **Fig2 帧：** 跨 codec / iVideoGPT 的 GT–reachable–residual  
4. **摘要/引言：** 用 §9 替换末段；全文搜杀 4.9%/7.7%/9.2%  
5. **投稿前：** 再读 Cosmos 与 CNN-FSQ floor 原始 JSON，冻结 Fig2 三位小数  

---

## 11. Verification checklist

- [x] Headline 数字有 path + field  
- [x] 选择规则写入 §7  
- [x] 无虚构 p；协议未静默混行  
- [x] 贡献句强度 ≤ strength  
- [x] null 平台未进改进主观图方案  
- [x] 本地 shortlist/package MD 已落盘  
- [ ] Fig6 帧文件（待生成）  
- [ ] Fig2 三 codec 柱最终 freeze 再读  
