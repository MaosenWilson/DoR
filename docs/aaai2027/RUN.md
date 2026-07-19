# AAAI-2027 当前实验运行手册

本页只包含尚未完成且允许执行的实验。所有命令前台运行并显示进度与 ETA；已完成或判负的命令
不再保留在工作区。

## 0. Environment

```bash
cd /root/autodl-tmp/vote2world
P=/root/autodl-tmp/external_wm/venv_ivideogpt/bin/python
export PYTHONPATH=src

$P - <<'PY'
import transformers, diffusers, huggingface_hub, torch
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("torch", torch.__version__)
PY
```

当前兼容组合为 Transformers 4.38.2、Diffusers 0.27.0、Hugging Face Hub 0.25.2。主环境曾因
RoboNet 下载升级 Hub 1.x 而失配，修复版本锁前统一使用上述 venv。

## 1. Temporal Correspondence Controls

Full arm 复用 `outputs/msp_lvkl_return` 的 RC seeds 0--2，只新增三组控制。预计 2--3 小时。

```bash
run_control () {
  NAME=$1
  MODE=$2
  HORIZON=$3

  $P scripts/train_grpo_msp.py \
    --rewards rc --adv_temporal "$MODE" --return_horizon "$HORIZON" \
    --seeds 0,1,2 \
    --T 8 --K 16 --steps 30 --batch_windows 2 \
    --train_windows 24 --eval_windows 8 \
    --lr 1e-5 --kl 0.001 --kl_type low_var_kl \
    --temporal_gamma 0.95 --eval_every 10 \
    --which rlvr --deterministic \
    --out_dir "outputs/msp_lvkl_return_${NAME}"
}

run_control L1 return 1
run_control L3 return 3
run_control shuffled shuffled_return 0

$P scripts/analyze_temporal_controls.py \
  --trunc1 'outputs/msp_lvkl_return_L1/sweep_rc_msp_s*.json' \
  --trunc3 'outputs/msp_lvkl_return_L3/sweep_rc_msp_s*.json' \
  --full 'outputs/msp_lvkl_return/sweep_rc_msp_s[0-2].json' \
  --shuffled 'outputs/msp_lvkl_return_shuffled/sweep_rc_msp_s*.json' \
  --expected_n 3 \
  --out outputs/analysis/msp_lvkl_temporal_controls_s0_2.json
```

完成标志：`GRPO_MSP_OK`、`TEMPORAL_CONTROLS_OK`。先判决该结果，再生成 $T=4/6$ 压力测试
命令；不得提前并行搜索 horizon 或超参数。

## 2. RoboNet Download and Asset Check

下载进程完成后只做完整性检查，不立即训练：

```bash
CKPT=/root/autodl-tmp/external_wm/checkpoints/ivideogpt-robonet-64-act-cond
SAMPLE=/root/autodl-tmp/external_wm/repos/iVideoGPT/inference/samples/robonet_sample.npz
DATA=/root/autodl-tmp/external_wm/datasets/robonet

du -sh "$CKPT" "$DATA"
ls -lh "$SAMPLE"
find "$DATA" -maxdepth 2 -type f -printf '%p %s bytes\n' | head -30
```

官方 sample 的模型接口冒烟已经通过。完整数据到位后，下一步是选择性解包、建立 episode-disjoint
manifest 和 rank gate；在 manifest 审计前不运行 GRPO。

## 3. Operating Preferences（可放宽）

- 优先本页命令与当前 C1/C2 主线；旧 reward/GRPO 探索不默认占用算力。
- 不强制无限扩 seed；有限 seed 下以完整报告与一致性为主。
- 论文数字选择：默认 final checkpoint + 全部可用 paired seeds；允许在预声明评测点/可用 seed
  上择优呈现，并在表注写清规则与 \(n\)。
- E2 有结论后再开 horizon-length stress 更省算力；若并行试跑，结果标签必须分开。
- RoboNet 优先完成 rank gate 再开训练 pilot，避免空跑。
