# AAAI-2027 实验运行手册

配套：`story_spine.md`（叙事/决定）、`experiment_matrix.md`（计划）、`configs/aaai2027/space_sweep.yaml`（设计）。
所有命令在**训练服务器**上跑（沙箱无 GPU/数据/权重，脚本未实跑，仅静态对齐 API）。

---

## Block 0 — 离线去风险（先做，不训练即可证伪 thesis）

5090 即可。先把 thesis 用数据砸实，再决定要不要花训练算力。

```bash
# 1) 缓存每个候选在所有空间的奖励距离 + DINOv2 参照（≥5 seed 求稳）
for SEED in 0 1 2 3 4; do
  python scripts/cache_reward_spaces.py --n_windows 200 --K 16 \
      --seed $SEED --out outputs/analysis/reward_spaces_s$SEED.npz
done

# 2) 秩保持度分析（Fig 2/3 数据 + 自动 verdict）
python scripts/analyze_rank_preservation.py --cache outputs/analysis/reward_spaces_s0.npz
```

**判读**：脚本末尾打印 `[verdict]`。若 `ON-TRACK`（pre-decode 的 code/phi 显著高于 pixel），
继续 B1；若 `WEAK`，**先回炉 thesis，别开训练**。

`phi_tok`（奖励噪声地板，LPIPS 单位）也会打印，对应 Fig 1 的地板值。

### ⚠️ 一个写代码时发现的关键点（影响 R2 / A3 baseline）

floor-校准像素若**减一个常数地板**，对**组内排序是不变的**（Spearman 只看秩，常数平移不改秩）
——即"减常数 floor 的 A3" 在排序上等于 A0，会变成稻草人。**A3 必须减逐样本/逐区域的
地板估计**（per-window `floor_lpips` 已在 npz 里，可直接用；更强可做 per-region）。这点反而是
论文的好论据：它解释了"简单减地板"为什么救不了——只有真正换到低噪空间才行。

---

## Block 1 — 核心空间扫描训练（5090，RT-1 single-step）

`train_grpo.py` 目前只支持 `pixel,code,hybrid`。接入新臂是**可选、非破坏**的两步：

1. 在 `scripts/train_grpo.py` 把 `REWARDS` 扩成 `("pixel","mse","ssim","floor","multi","phi","code","hybrid")`。
2. 把 `dor.grpo` 里训练用的 `gt_reward` 换成 `dor.reward_spaces.gt_reward`
   （pixel/code/hybrid 逐位一致，不影响既有 run）。

> 注意：`floor` 臂需要把 `phi_tok` 传进 `gt_reward`（取 B0 的 `floor_lpips` 均值）。
> 这要在 `dor.grpo.train` 里多穿一个参数；改动小但要测。其余臂无需该参数。

```bash
python scripts/train_grpo.py \
    --rewards pixel,mse,ssim,multi,phi,code --modes gt_only \
    --steps 150 --K 16 --seed 0 \
    --out outputs/grpo/space_sweep_s0.json
# 对 seed 0..4 重复；floor 臂在打好上面的 phi_tok 补丁后单独跑
```

主结论只用独立指标（eval LPIPS / PSNR / FVD / 光流一致性）；`code_rms` 仅进附录。

---

## Block 2–5（简）

- **B2 理论验证**：在 B0 的 npz 上算"组内某对候选被噪声翻转"的实测概率，对照 bound（Fig 5）。
- **B3 稳健性（H800）**：更长步数、第二域 RECON、多步 rollout、C2 大 K。
- **B4 主观图**：`scripts/figures/export_frames.py` 导出随机 + 精选运动 case（Fig 6）。
- **B5 消融**：β 敏感性、reward scale、A5(phi) vs A6(code) 量化影响、consensus×code 交互。

---

## 新增/改动文件一览

| 文件 | 作用 | 状态 |
|---|---|---|
| `src/dor/reward_spaces.py` | 7 个奖励臂统一实现（train 的 drop-in 超集） | 新增，未实跑 |
| `scripts/cache_reward_spaces.py` | B0 缓存：各空间距离 + DINOv2 参照 + 地板 | 新增，未实跑 |
| `scripts/analyze_rank_preservation.py` | B0 分析：秩保持度 + verdict（逻辑已用合成数据自测） | 新增 |
| `configs/aaai2027/space_sweep.yaml` | 实验设计记录 | 新增 |
| `train_grpo.py` / `grpo.py` | 接新臂的 2 步 opt-in 补丁（见上） | **未改动**，待你接 |
