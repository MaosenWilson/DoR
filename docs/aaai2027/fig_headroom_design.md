# 论文图:Per-Horizon Prediction Headroom（预测余量）

> 2026-07-18。核心机制图之一。数据与草图在 `outputs/headroom_curves/`。

## 这张图要传达什么

**一句话:时序信用（C2）只能在 base world model 留有余量的地方起作用——即 per-horizon
误差随 horizon 增长的地方。** 这解释了为什么 C2 在 RT-1 有效、在近天花板的 RoboDesk 归零,
把"C2 不是普适地涨点"从尴尬转成"有原则的适用条件"。

## 三个面板（草图 `fig_headroom_draft.png`）

- **(a) VP2 base LPIPS vs horizon**:RoboSuite 从 0.010 涨到 0.048（有余量），RoboDesk
  从 0.005 只到 0.011（近天花板）。同 iVideoGPT 架构、同 64×64 分辨率,两条线可直接叠。
- **(b) VP2 base SSIM vs horizon**:RoboSuite 从 0.98 掉到 0.87,RoboDesk 稳在 0.98。
- **(c) RT-1 C2 效应 vs horizon**:return−seq 的 LPIPS delta 从 h2 的 −0.0013 扩大到 h7 的
  −0.0041——**有余量处,C2 的收益随 horizon 增长**。这条把 (a)/(b) 的"余量"和 C2 的"收益"接上。

## 数据来源与状态

| 曲线 | 来源 | 状态 |
|---|---|---|
| RoboSuite base LPIPS/SSIM (h1–15) | `headroom_audit.py`，robosuite ckpt，16 ctx×K8 | ✅ 已测 |
| RoboDesk base LPIPS/SSIM (h1–15) | 同上，robodesk ckpt | ✅ 已测 |
| RT-1 return−seq per-horizon delta (h2–7) | experiments.md §3.3（已有 5-seed 复评） | ✅ 已有 |
| **RT-1 base LPIPS per-horizon** | `eval_msp_horizon.py` base 臂 | ⏳ 待跑（需 GPU 空闲 + 修 transformers 环境；base miniconda hub 坏、venv_ivideogpt 的 Llama causal_mask OOM） |
| **VP2 after-C2 per-horizon**（RoboSuite return vs seq 逐 horizon） | 需 `--save_checkpoints` 重跑 + per-horizon eval | ⏳ 待做（当前 h15 训练未存 ckpt） |

## 论文版还要补的两层（让"应用前/后"完整）

1. **VP2 的 after-C2 overlay**:在有余量的 RoboSuite 上,把 return 训练后的 per-horizon LPIPS
   叠加到 base 曲线上,直观显示"C2 把后段误差压下来"。前提:RoboSuite h15 C2 训练判为正
   （3-seed 方向 → 5-seed 坐实）后,带 `--save_checkpoints` 重跑,再逐 horizon eval seq/return。
2. **RT-1 base 曲线**:补全 (c) 的绝对参照。修好环境后一条命令。

## 诚实边界

- RT-1（256×320）与 VP2（64×64）LPIPS 尺度不同,不在同一面板叠绝对值;跨数据集比较看
  "增长趋势/SSIM"，或各自归一化到 h1。
- 余量存在 ≠ C2 一定有效,只是必要条件;RoboSuite 是否真获益仍看训练判决（进行中）。
- headroom_audit 的 base 候选用 temperature=1.0、K=8，与训练采样一致;报 mean 与 best 候选。
