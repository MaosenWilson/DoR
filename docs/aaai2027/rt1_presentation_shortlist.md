# RT-1 结果呈现 shortlist（paper-result-presentation）

**日期：** 2026-07-17  
**服务器：** AutoDL `connect.westd.seetacloud.com:25017` → `/root/autodl-tmp/vote2world`  
**本地镜像：** `tmp/server_rt1_snapshot/`  
**对照主线：** `RC-GRPO_研究主线与贡献大纲.docx`（C1 可达排序校准 + C2 帧块时间回报）  
**技能：** `paper-result-presentation` v2（只选真实 artifact，不改数、不静默混协议）

---

## 0. 一页结论（先看这个）

| 贡献 | 主线要求 | RT-1 现状 | 是否够“漂亮”支撑主文 |
|---|---|---|---|
| **C1 机制** | raw/RC 同候选排序失配可诊断；RC 降低 flip、提高 ρ | 三个 FSQ codec 均有不可忽略的编解码重建误差；RT-1 single/multi 与 VP2 排序均 GREEN | **是**（重建误差图 + 机制表 headline） |
| **C1 训练** | RA-RC 同时优于 raw 与 pure RC（raw fidelity + reachable consistency） | 仅 single-step pilot s0–2：相对 raw 有改善；相对 pure RC **未稳定优于** | **否**（不能写“普适 fidelity 提升”） |
| **C2 训练** | Temporal Return 改善 full-rollout LPIPS；与 RC 统一 | `return_rc` 四臂最优；formal 5/5、p≈0.081；E1 未见 episode 4/5、p≈0.125 | **趋势支持，主表可写，措辞须保守** |
| **C2 因果** | 正确时间对应是原因（E2 shuffled 等） | **未跑** | **不能写** |
| **C1×C2 超加** | 完整组合是否统计超加 | interaction 跨零、p≈0.59 | **已拒绝，不列 C3** |

**总判：** 与主线 **方向一致**，但 **证据强度分层**：  
- 可写强：C1 **跨平台 rank repair**；C2 **RT-1 上 return_rc 最优 + 方向一致**。  
- 只能弱写：full-rollout 双侧 p 临界 / 复评 4/5。  
- 不能写：RA-RC 完整 C1 训练门、时间对应因果、跨平台 raw-GT 训练普适增益。

相对提升量级约 **LPIPS −1.4% vs seq_rc**、**E1 上 vs base ≈−7.6%、vs 官方 RLVR ckpt ≈−3.7%**（协议需表注，勿与 formal 8-window 混表）。

---

## 1. Inventory

| path（服务器 / 本地镜像） | 内容 | seeds | protocol tags | 主指标 |
|---|---|---|---|---|
| `outputs/analysis/msp_lvkl_factorial_s0_4.json` | 正式四臂 in-training eval | 0–4 | T=8,K=16,steps=30,`low_var_kl`,γ=0.95, eval 8-win | LPIPS / last / MSE |
| `outputs/analysis/msp_episode_disjoint_eval_summary.json` | E1 未见 episode 复评 | 0–4 | 8 eps × 32 win，window_macro；manifest `fb22f86c…` | 同上 + PSNR/SSIM |
| `outputs/analysis/msp_episode_disjoint_eval/{base,rlvr}_eval.json` | base / 官方 RLVR 同协议 | — | 同 E1 | 同上 |
| `outputs/analysis/rank_calibration_audit.json` | RT-1 same-cand rank audit | ep-cluster | single CNN-FSQ；multi compressive | Δρ, Δflip |
| `outputs/external/analysis/vp2_*_rank95.json` | VP2 rank | cluster boot | frozen cache | Δρ, Δflip |
| Cosmos DV-FSQ floor audit | 独立 codec 编解码重建误差 | frame audit | frozen codec | LPIPS-VGG floor |
| `outputs/analysis/ra_rc_rt1_single_pilot.json` | RA-RC 三臂 pilot | 0–2 | single-step RT-1 | LPIPS/MSE/… vs raw/RC |

运行目录：`outputs/msp_lvkl_seq/`、`outputs/msp_lvkl_return/`。服务器当前 **无训练进程**（GPU 空闲）。

---

## 2. Comparison axes（对齐 Word 主线）

| claim_id | 贡献句（主线） | primary metric（↓除非注明） | 对比 arms | 推断单位 |
|---|---|---|---|---|
| C1-mech | 可达残差改变 GRPO 排序；RC 修复排序 | Δρ ↑, Δflip ↓ | raw vs RC 同候选 | episode cluster |
| C1-train | RA-RC 相对 raw 与 pure RC 双赢 | held-out raw LPIPS | raw / RC / RA-RC | paired seed |
| C2-main | 帧块 Temporal Return 改善多步预测 | full-rollout LPIPS | return_rc vs seq_rc | paired seed 0–4 |
| C2-tail | 尾帧质量（支持） | LPIPS-last | 同上 | 同上 |
| C2-corr | 收益来自正确时间对应 | full vs shuffled | pending E2 | — |

**Baselines：** seq_raw；seq_rc（C2 主对照）；base / rlvr（E1 外部锚点，分栏）。

---

## 3. Shortlist（ranked）

| rank | 对比 | metric | value / Δ | consistency | role 建议 | path |
|---|---|---|---|---|---|---|
| 1 | return_rc vs seq_rc | formal LPIPS | **−0.002792 (−1.38%)** | **5/5**, t=−2.32, **p=0.081**, boot [−0.0051,−0.0012] | **headline**（trend） | `msp_lvkl_factorial_s0_4.json#metrics.eval_lpips` |
| 2 | return_rc vs seq_rc | formal LPIPS-last | **−0.003658 (−1.73%)** | **5/5**, t=−3.11, **p=0.036**, boot [−0.0057,−0.0016] | **support / 副指标** | 同上 `#eval_lpips_last` |
| 3 | 四臂均值 | formal LPIPS | return_rc **0.19935** 最优 | 四臂中最低 | headline 主表行 | 同上 `#arm_means` |
| 4 | return_rc vs seq_rc | **E1** LPIPS | **−0.002764 (−1.39%)** | **4/5**, p=0.125, boot [−0.0053,−0.0003] | **support**（泛化复评） | `msp_episode_disjoint_eval_summary.json` |
| 5 | return_rc vs seq_rc | E1 LPIPS-last | −0.004093 | 4/5, p=0.149 | support | 同上 |
| 6 | return_rc vs seq_rc | E1 MSE | −0.000264 | **5/5**, p=0.074 | support | 同上 |
| 7 | return_rc vs base/rlvr | E1 LPIPS | 0.19657 vs 0.21273 / 0.20403 | 单协议锚点 | support（**勿与 formal 混行**） | `base_eval.json` / `rlvr_eval.json` |
| 8 | RC vs raw rank | RT-1 multi Δρ / Δflip | +0.0145 / −0.0087 | CI 不含 0（Δρ） | **C1 mechanism headline** | `rank_calibration_audit.json` |
| 9 | 跨平台 rank | VP2 Δρ | +0.075~+0.093 | H2/H8 均 GREEN | C1 mechanism 扩展表 | `outputs/external/analysis/vp2_*` |
| 10 | 跨 codec floor | CNN / compressive / Cosmos LPIPS | 0.053 / 0.077 / 0.190 | 均非零 | C1 generality 图 | floor audit artifacts |
| 11 | RA-RC − raw | pilot LPIPS | −0.00117 | 3/3 | appendix / pilot | `ra_rc_rt1_single_pilot.json` |
| 12 | RA-RC − RC | pilot LPIPS | −0.00060 | 2/3, CI 跨 0 | **limit** | 同上 |
| 13 | interaction (RC×Return) | formal LPIPS | −0.00214 | 3/5, p=0.59 | **omit as C3** | factorial interaction |

---

## 4. Display roles

| item | role | note |
|---|---|---|
| CNN-FSQ + compressive FSQ + Cosmos floor audit | **headline** C1 | 证明问题跨独立 codec 存在 |
| RT-1 + VP2 rank audit | **headline** C1 | 证明排序修复跨世界模型成立 |
| formal 四臂表 + return_rc−seq_rc | **headline** C2 | 主指标 full LPIPS；p 临界须写 “consistent / improves” |
| LPIPS-last formal | **support** | 不可替换 pre-specified primary |
| E1 episode-disjoint | **support** | 方向保留、统计更保守；分表 |
| base / rlvr E1 | **support** | 同协议锚点，非完整 recipe 复现声明 |
| RA-RC pilot | **appendix** | GREEN vs raw；未证明优于 pure RC |
| super-additivity | **omit** | 拒绝 C3 |
| E2/E3 | **cannot** | 未完成则不写因果/长度外推 |

---

## 5. Claim → cell map

| claim | metric | value | n | selection rule | source | strength |
|---|---|---|---|---|---|---|
| C1-mech RT-1 single | Δρ | +0.0168 [0.0035, 0.0289] | 148 groups | episode bootstrap | `rank_calibration_audit.json#single_cnnfsq` | **supported** |
| C1-mech RT-1 multi | Δρ | +0.0145 [0.0105, 0.0176] | 1152 | 同上 | `#multi_compressive_all` | **supported** |
| C1-mech VP2 H2 | Δρ_s | +0.0932 [0.0674, 0.1217] | — | 95% cluster | `vp2_h2_rank95.json` | **supported** |
| C1-floor Cosmos DV-FSQ | LPIPS-VGG floor | 0.190 | matched RT-1 frames | frozen encode--decode | Cosmos floor artifact | **supported** |
| C1-train RA vs raw | LPIPS | −0.00117 | 3 | final, pilot | `ra_rc_rt1_single_pilot.json` | **trend (pilot only)** |
| C1-train RA vs RC | LPIPS | −0.00060, 2/3 | 3 | 同上 | 同上 | **boundary / cannot 强写** |
| C2 full LPIPS formal | Δ LPIPS | −0.002792 | 5 | final ckpt, paired | factorial `return_effect_under_rc` | **trend (5/5, p=0.081)** |
| C2 last formal | Δ LPIPS-last | −0.003658 | 5 | 同上 | 同上 | **supported (p=0.036)** |
| C2 full E1 | Δ LPIPS | −0.002764 | 5 | episode-disjoint | ED summary | **trend (4/5)** |
| C2 corr | — | — | — | — | E2 missing | **cannot** |

### Formal 四臂主表（in-training 8-window，`low_var_kl`）

| arm | LPIPS ↓ | LPIPS-last ↓ | MSE ↓ |
|---|---:|---:|---:|
| sequence + raw | 0.20224 | 0.21278 | 0.013109 |
| sequence + RC | 0.20214 | 0.21141 | 0.013178 |
| Temporal Return + raw | 0.20158 | 0.21151 | 0.013287 |
| **Temporal Return + RC** | **0.19935** | **0.20775** | **0.012756** |

### E1 四臂 + 锚点（episode-disjoint，window_macro）

| arm | LPIPS ↓ | LPIPS-last ↓ | MSE ↓ |
|---|---:|---:|---:|
| seq_raw | 0.19915 | 0.21180 | 0.013679 |
| seq_rc | 0.19933 | 0.21176 | 0.013820 |
| return_raw | 0.20118 | 0.21487 | 0.013941 |
| **return_rc** | **0.19657** | **0.20767** | **0.013556** |
| base | 0.21273 | 0.22783 | 0.015480 |
| rlvr (official ckpt) | 0.20403 | 0.21207 | 0.014390 |

---

## 6. Selection rules used this round

1. **Real-only：** 数字均来自上表 path；本地 `tmp/server_rt1_snapshot/` 与服务器 2026-07-17 一致。  
2. **Checkpoint：** final after 30 updates（非 best-seed / 未表注 best-step）。  
3. **Seeds：** C2 formal/E1 用完整 paired 0–4；RA-RC 披露 n=3。  
4. **Primary：** full-rollout LPIPS；LPIPS-last 不得偷换 primary。  
5. **协议分表：** formal 8-window ≠ E1 32-window；禁止混成一行。  
6. **KL：** 仅 `low_var_kl` 正式四臂进 headline；旧 linear KL 不混写。  
7. **“最漂亮”含义：** 在真实池内选叙事最清晰配置 = **return_rc**；不是编造更小数。

---

## 7. Appendix / limitations / cannot-write

**Appendix**

- RA-RC RT-1 single pilot 全指标明细；RA decision=`GREEN`（非灾难界，非完整 C1 门）。  
- E1 per-seed / per-window 明细目录：`msp_episode_disjoint_eval/*_s{0..4}.json`。  
- RC alone under sequence：formal/E1 上 raw-GT LPIPS **近零/略负**——机制修复 ≠ 单臂训练大增益。

**Limitations（与 Word §当前证据 一致）**

- 纯 RC / return 的 **跨平台 raw-GT 训练收益未成立**（VP2 training 另册；Cosmos 无匹配 policy training）。  
- n=5 时 full LPIPS 双侧 p 易临界；依赖 wins + bootstrap 方向。  
- E1 降低 train–eval episode 重叠，但 **推断单位仍是 5 个 training seeds**。

**Cannot write（本轮）**

1. “RA-RC 已证明同时优于 raw 与 pure RC 的完整 C1 框架”——**缺多平台 + 相对 RC 不稳**。  
2. “Temporal Return 的收益 **因为** 正确时间信用分配”——**缺 E2 shuffled**。  
3. “收益随 horizon 增长”——**缺 E3**。  
4. “RC 与 Temporal Return 统计超加 / 第三贡献”——**已拒绝**。  
5. “击败 RLVR-World 完整 recipe”——E1 仅 **同 held-out 协议下 frozen ckpt 比较**。

---

## 8. 与 Word 主线符合度（审稿视角）

Word 固定主线：

1. **可达范围内可靠比较（C1）**  
2. **原始目标下受约束更新（RA-RC）**  
3. **自回归时间轴上正确归因（C2）**

| 主线环节 | 符合？ | 证据 |
|---|---|---|
| 排序失配诊断 | ✅ | 三平台 rank audit |
| RC verifier 改目标不改 metric | ✅ | 机制 + 训练接口一致 |
| Raw-anchored 完整训练门 | ⚠️ pilot only | RT-1 s0–2；未过“双赢”硬门 |
| Temporal Return 块级信用 | ✅ 趋势 | return_rc 最优，5/5 formal |
| 时间对应因果 | ❌ pending | E2 |
| 两贡献统一系统 | ⚠️ 叙事可写、统计不超加 | 完整臂最好，无 C3 |

**投稿强度建议**

- **Strong（机制 + RT-1 方法组合）：** C1 = cross-platform rank calibration；C2 = RT-1 multi-step Temporal Return under RC，以 return_rc 为主结果。  
- **Conservative：** C2 写 “consistent improvements / best mean arm”，避免 “significantly proves”；C1 训练层只写 RT-1 pilot + conversion boundary。  
- **不够支撑的强主张：** 完整 RA-RC 三件套已闭合；时间信用的因果必然性；三平台像素级全面碾压。

---

## 9. 下一步（决定主张上限）

1. **E2** temporal controls（L=1/3/full + shuffled）— 决定 C2 能否写 “correspondence”。  
2. 若要坚持完整 C1 训练句：补 **RA-RC multi-step / 跨平台**，且相对 pure RC 要稳。  
3. 主文表：formal 四臂 + rank 机制表；E1 作 support 表；RA-RC 附录。

---

*生成方式：SSH 盘点 → scp 至 `tmp/server_rt1_snapshot/` → 按 skill 模板筛选。未修改任何原始 JSON。*
