# AAAI-2027 当前实验运行手册

> 2026-07-15 起的唯一运行手册。全部前台运行并显示进度/ETA。当前最高优先级是下述 RA-RC 跨平台准入；未通过 pilot 的平台不得扩种子。

> RC-Energy 双组门控已正式 RED：分布效用相关显著提高，但 LPIPS/MSE 非劣边界失败，禁止运行其 GRPO 六臂命令。RCAV Gate B 也已两次 RED，`0.1--0.6` 仅保留历史复核，不得再次运行或接入 GRPO。

## 0. 固定环境与协议

```bash
cd /root/autodl-tmp/vote2world
P=/root/autodl-tmp/external_wm/venv_ivideogpt/bin/python
export PYTHONPATH=src
```

主环境 `/root/miniconda3/bin/python` 当前存在 `huggingface_hub` / Transformers 版本冲突；本页命令统一使用已通过 RT-1、VP2、IRIS 冒烟的 external-WM venv。

### 0.A RA-RC 已完成的接线验收

服务器真实模型冒烟已覆盖 RT-1 single-step、RT-1 multi-step、VP2 H8 与 IRIS Breakout；raw/RC/RA-RC 路径均可完成前向、反向、评测和 JSON 保存。自动测试为 32 passed。冒烟只证明接线与梯度投影的 active/no-op 两个分支可达，不作为方法正结果。

2026-07-15 运行审计发现 external-WM venv 的 Transformers 4.38 会继承训练配置 `use_cache=False`，使 RT-1 自回归采样重复计算完整上下文。现已只在 `generate_candidates` 内显式启用 KV cache，teacher-forced backward 仍保持关闭。真实 RT-1 同窗口/同 seed/$K=2$ 对照为 4.59 s vs 8.69 s（1.89x），640 个候选 token 逐位一致（Hamming=0），模型配置调用后仍为 `False`。所有 single-step arm 现逐 step 输出 `step_time`、累计 `elapsed` 与 `eta`，并把耗时写入 JSON 的 `train_step_seconds`。

### 0.B IRIS actor-policy 独立数据与 rank gate

先生成两个独立 actor-policy manifest；命令有 episode/window 进度和 ETA：

```bash
$P scripts/external/cache_iris_breakout.py \
  --game Breakout --episodes 16 --windows_per_episode 8 \
  --context 4 --window_stride 4 --warmup 24 --seed 18123 \
  --collection_policy checkpoint_actor --actor_temperature 0.5 \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Breakout.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris --device cuda \
  --cache_dir outputs/external/iris/breakout_actor_p1/windows \
  --manifest outputs/external/iris/breakout_actor_p1/manifest.json

$P scripts/external/cache_iris_breakout.py \
  --game Pong --episodes 16 --windows_per_episode 8 \
  --context 4 --window_stride 4 --warmup 24 --seed 28123 \
  --collection_policy checkpoint_actor --actor_temperature 0.5 \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Pong.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris --device cuda \
  --cache_dir outputs/external/iris/pong_actor_p1/windows \
  --manifest outputs/external/iris/pong_actor_p1/manifest.json
```

随后运行同一 rank gate，正式锁定 95% episode-cluster CI：

```bash
$P scripts/external/gate_iris_breakout_rank.py \
  --manifest outputs/external/iris/breakout_actor_p1/manifest.json \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Breakout.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris \
  --K 16 --draws 2 --seed 19203 --bootstrap 10000 --confidence 0.95 --device cuda \
  --cache outputs/external/iris/breakout_actor_p1/rank_cache.npz \
  --out outputs/external/iris/breakout_actor_p1/gate.json

$P scripts/external/gate_iris_breakout_rank.py \
  --manifest outputs/external/iris/pong_actor_p1/manifest.json \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Pong.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris \
  --K 16 --draws 2 --seed 29203 --bootstrap 10000 --confidence 0.95 --device cuda \
  --cache outputs/external/iris/pong_actor_p1/rank_cache.npz \
  --out outputs/external/iris/pong_actor_p1/gate.json

$P - <<'PY'
import json
paths = [
    "outputs/external/iris/breakout_actor_p1/gate.json",
    "outputs/external/iris/pong_actor_p1/gate.json",
]
rows = [(p, json.load(open(p))["verdict"]) for p in paths]
print(*rows, sep="\n")
assert all(v == "GREEN" for _, v in rows), "IRIS actor rank gate RED: stop before training"
print("IRIS_ACTOR_RANK_ADMISSION_GREEN")
PY
```

### 0.C RA-RC 三平台 paired pilot（完整训练长度，seeds 0--2）

顺序固定为 RT-1 → VP2 → IRIS Breakout → IRIS Pong。pilot 只减少 seed 数，不缩短训练步数，因此 GREEN 后可直接续跑正式 seeds。

```bash
# C1. RT-1 single-step：约 4 小时（9 runs）
$P scripts/train_grpo.py \
  --rewards a0faithful,a0faithful_tok,ra_rc --modes gt_only \
  --seeds 0,1,2 --steps 150 --K 16 --batch_windows 2 \
  --train_windows 24 --eval_windows 12 --lr 1e-5 --kl 0 \
  --eval_every 10 --deterministic --no_save_checkpoints \
  --out_dir outputs/ra_rc/rt1_single

$P scripts/analyze_ra_rc.py \
  --platform rt1 --input_dir outputs/ra_rc/rt1_single \
  --stage pilot --expected_n 3 --bootstrap 10000 \
  --out outputs/analysis/ra_rc_rt1_single_pilot.json

# C2. VP2 H8 sequence credit：约 30 分钟（9 runs）
$P scripts/external/train_ivideogpt_vp2_grpo.py \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/ivideogpt-vp2-robosuite-64-act-cond \
  --upstream /root/autodl-tmp/external_wm/repos/iVideoGPT \
  --train_manifest outputs/external/vp2/p2_h8_final_protocol_v1/train_manifest.json \
  --eval_manifest outputs/external/vp2/p2_h8_final_protocol_v1/eval_manifest.json \
  --rewards raw,rc,ra_rc --credits seq --seeds 0,1,2 \
  --horizon 8 --K 16 --eval_K 4 --steps 30 --batch_windows 2 \
  --lr 1e-6 --kl 0.001 --kl_type low_var_kl --gamma 0.95 \
  --data_seed 7304 --eval_every 10 --eval_seed 9917 --deterministic \
  --out_dir outputs/external/vp2/ra_rc_h8_seq

$P scripts/analyze_ra_rc.py \
  --platform vp2 --credit seq --input_dir outputs/external/vp2/ra_rc_h8_seq \
  --stage pilot --expected_n 3 --bootstrap 10000 \
  --out outputs/analysis/ra_rc_vp2_pilot.json

# C3. IRIS Breakout actor-policy：约 20--40 分钟（9 runs）
$P scripts/external/train_iris_atari_grpo.py \
  --manifest outputs/external/iris/breakout_actor_p1/manifest.json \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Breakout.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris \
  --rewards raw,rc,ra_rc --seeds 0,1,2 --K 16 --eval_K 16 \
  --steps 20 --batch_windows 2 --eval_episodes 4 \
  --split_seed 9413 --data_seed 9414 --eval_seed 9415 \
  --lr 3e-6 --weight_decay 0.01 --kl 0.001 --grad_clip 1.0 \
  --eval_every 10 --deterministic \
  --out_dir outputs/external/iris/breakout_actor_ra_rc

$P scripts/analyze_ra_rc.py \
  --platform iris --input_dir outputs/external/iris/breakout_actor_ra_rc \
  --stage pilot --expected_n 3 --bootstrap 10000 \
  --out outputs/analysis/ra_rc_iris_breakout_pilot.json

# C4. IRIS Pong actor-policy：约 20--40 分钟（9 runs）
$P scripts/external/train_iris_atari_grpo.py \
  --manifest outputs/external/iris/pong_actor_p1/manifest.json \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Pong.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris \
  --rewards raw,rc,ra_rc --seeds 0,1,2 --K 16 --eval_K 16 \
  --steps 20 --batch_windows 2 --eval_episodes 4 \
  --split_seed 9413 --data_seed 9414 --eval_seed 9415 \
  --lr 3e-6 --weight_decay 0.01 --kl 0.001 --grad_clip 1.0 \
  --eval_every 10 --deterministic \
  --out_dir outputs/external/iris/pong_actor_ra_rc

$P scripts/analyze_ra_rc.py \
  --platform iris --input_dir outputs/external/iris/pong_actor_ra_rc \
  --stage pilot --expected_n 3 --bootstrap 10000 \
  --out outputs/analysis/ra_rc_iris_pong_pilot.json
```

只有四份分析 JSON 全部为 `GREEN`，才执行正式扩种。任一平台 RED 时先交付结果分析，不继续烧该平台的 seeds 3--9。

### 0.D VP2 verifier x temporal-credit direct gate

RA-RC 的 sequence-credit pilot 只检验了梯度约束，不回答校准排序进入长序列回报后是否有效。
本门复用 `ra_rc_h8_seq` 中已有的 `seq-raw/seq-RC`，只新增
`return-raw/return-RC` seeds 0--2。冻结判据为：`return-RC` 的 LPIPS 均值为四臂最低；
Temporal Return 在 RC 下、RC 在 Temporal Return 下及二者交互均为负且至少 2/3 seeds 同向；
前两项 MSE 不得反向。任一条件失败时不扩 seed，也不以 latent 指标替代 raw-GT primary。

```bash
cd /root/autodl-tmp/vote2world
P=/root/autodl-tmp/external_wm/venv_ivideogpt/bin/python
export PYTHONPATH=src

$P scripts/external/train_ivideogpt_vp2_grpo.py \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/ivideogpt-vp2-robosuite-64-act-cond \
  --upstream /root/autodl-tmp/external_wm/repos/iVideoGPT \
  --train_manifest outputs/external/vp2/p2_h8_final_protocol_v1/train_manifest.json \
  --eval_manifest outputs/external/vp2/p2_h8_final_protocol_v1/eval_manifest.json \
  --rewards raw,rc --credits return --seeds 0,1,2 \
  --horizon 8 --K 16 --eval_K 4 --steps 30 --batch_windows 2 \
  --lr 1e-6 --kl 0.001 --kl_type low_var_kl --gamma 0.95 \
  --data_seed 7304 --eval_every 10 --eval_seed 9917 --deterministic \
  --out_dir outputs/external/vp2/rctr_h8_return_pilot

$P scripts/analyze_msp_factorial.py --platform vp2 \
  --seq_raw 'outputs/external/vp2/ra_rc_h8_seq/sweep_raw_seq_s*.json' \
  --seq_rc 'outputs/external/vp2/ra_rc_h8_seq/sweep_rc_seq_s*.json' \
  --return_raw 'outputs/external/vp2/rctr_h8_return_pilot/sweep_raw_return_s*.json' \
  --return_rc 'outputs/external/vp2/rctr_h8_return_pilot/sweep_rc_return_s*.json' \
  --expected_n 3 --bootstrap 20000 \
  --out outputs/analysis/vp2_rctr_factorial_s0_2.json
```

### 0.E RCTR VP2 paired training pilot

独立 confirmation cache 已通过离线门控后，只新增 `rctr/return` seeds 0--2；三个对照臂
复用相同协议下已经完成的 `seq-raw`、`return-raw` 与 `return-RC`。pilot 的 primary 是
final raw-GT LPIPS 相对当前最强 `seq-raw` 均值降低且至少 2/3 seeds 同向，同时 MSE/SSIM
不得反向；RCTR 的 LPIPS 均值还必须低于两个 return 对照。不得用 best step 替换 final step。

```bash
cd /root/autodl-tmp/vote2world
P=/root/autodl-tmp/external_wm/venv_ivideogpt/bin/python
export PYTHONPATH=src

$P scripts/external/train_ivideogpt_vp2_grpo.py \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/ivideogpt-vp2-robosuite-64-act-cond \
  --upstream /root/autodl-tmp/external_wm/repos/iVideoGPT \
  --train_manifest outputs/external/vp2/p2_h8_final_protocol_v1/train_manifest.json \
  --eval_manifest outputs/external/vp2/p2_h8_final_protocol_v1/eval_manifest.json \
  --rewards rctr --credits return --seeds 0,1,2 \
  --horizon 8 --K 16 --eval_K 4 --steps 30 --batch_windows 2 \
  --lr 1e-6 --kl 0.001 --kl_type low_var_kl --gamma 0.95 \
  --data_seed 7304 --eval_every 10 --eval_seed 9917 --deterministic \
  --out_dir outputs/external/vp2/rctr_h8_training_pilot

$P scripts/analyze_rctr.py \
  --rctr 'outputs/external/vp2/rctr_h8_training_pilot/sweep_rctr_return_s*.json' \
  --seq_raw 'outputs/external/vp2/ra_rc_h8_seq/sweep_raw_seq_s*.json' \
  --return_raw 'outputs/external/vp2/rctr_h8_return_pilot/sweep_raw_return_s*.json' \
  --return_rc 'outputs/external/vp2/rctr_h8_return_pilot/sweep_rc_return_s*.json' \
  --expected_n 3 --bootstrap 20000 \
  --out outputs/analysis/rctr_vp2_training_s0_2.json
```

### 0.F RA-RC 正式扩种（仅在 0.C 全绿后运行）

以下命令复用 0.C 输出目录并断点续跑，不覆盖 seeds 0--2：

```bash
# RT-1：续跑 21 runs
$P scripts/train_grpo.py \
  --rewards a0faithful,a0faithful_tok,ra_rc --modes gt_only \
  --seeds 3,4,5,6,7,8,9 --steps 150 --K 16 --batch_windows 2 \
  --train_windows 24 --eval_windows 12 --lr 1e-5 --kl 0 \
  --eval_every 10 --deterministic --no_save_checkpoints \
  --out_dir outputs/ra_rc/rt1_single

# VP2：续跑 21 runs
$P scripts/external/train_ivideogpt_vp2_grpo.py \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/ivideogpt-vp2-robosuite-64-act-cond \
  --upstream /root/autodl-tmp/external_wm/repos/iVideoGPT \
  --train_manifest outputs/external/vp2/p2_h8_final_protocol_v1/train_manifest.json \
  --eval_manifest outputs/external/vp2/p2_h8_final_protocol_v1/eval_manifest.json \
  --rewards raw,rc,ra_rc --credits seq --seeds 3,4,5,6,7,8,9 \
  --horizon 8 --K 16 --eval_K 4 --steps 30 --batch_windows 2 \
  --lr 1e-6 --kl 0.001 --kl_type low_var_kl --gamma 0.95 \
  --data_seed 7304 --eval_every 10 --eval_seed 9917 --deterministic \
  --out_dir outputs/external/vp2/ra_rc_h8_seq

# IRIS 两游戏：各续跑 21 runs；参数与 0.C 完全相同，只改 seeds
$P scripts/external/train_iris_atari_grpo.py \
  --manifest outputs/external/iris/breakout_actor_p1/manifest.json \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Breakout.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris \
  --rewards raw,rc,ra_rc --seeds 3,4,5,6,7,8,9 --K 16 --eval_K 16 \
  --steps 20 --batch_windows 2 --eval_episodes 4 \
  --split_seed 9413 --data_seed 9414 --eval_seed 9415 \
  --lr 3e-6 --weight_decay 0.01 --kl 0.001 --grad_clip 1.0 \
  --eval_every 10 --deterministic \
  --out_dir outputs/external/iris/breakout_actor_ra_rc

$P scripts/external/train_iris_atari_grpo.py \
  --manifest outputs/external/iris/pong_actor_p1/manifest.json \
  --checkpoint /root/autodl-tmp/external_wm/checkpoints/iris/Pong.pt \
  --upstream /root/autodl-tmp/external_wm/repos/iris \
  --rewards raw,rc,ra_rc --seeds 3,4,5,6,7,8,9 --K 16 --eval_K 16 \
  --steps 20 --batch_windows 2 --eval_episodes 4 \
  --split_seed 9413 --data_seed 9414 --eval_seed 9415 \
  --lr 3e-6 --weight_decay 0.01 --kl 0.001 --grad_clip 1.0 \
  --eval_every 10 --deterministic \
  --out_dir outputs/external/iris/pong_actor_ra_rc
```

正式统计分别运行 `scripts/analyze_ra_rc.py`，把 `--stage pilot --expected_n 3` 改为 `--stage full --expected_n 10`；四个平台必须分别判决，禁止先池化再挑结论。

正式 multi-step 协议：`T=8, K=16, steps=30, batch_windows=2, train/eval=24/8, lr=1e-5, kl=0.001, kl_type=low_var_kl, deterministic`。正式结果统一写入带 `_lvkl` 的新目录。

### 0.0 RC-Energy：服务器 smoke 与双组门控

先验证 LPIPS-VGG feature blocks 和 reward arm wiring：

```bash
PYTHONPATH=src $P scripts/smoke_rc_energy.py --K 4 \
  --out outputs/rc_energy/smoke_config.json
```

必须出现 `RC_ENERGY_SMOKE_OK`。正式 gate 分两步，均为前台运行：

```bash
PYTHONPATH=src $P scripts/cache_rc_energy.py \
  --n_windows 148 --exclude_windows 36 --window_seed 1 \
  --generation_seeds 7301,17301 --K 16 --temperature 1.0 --top_k 100 \
  --scale_episode_fraction 0.2 --deterministic \
  --out outputs/rc_energy/two_group_cache.npz \
  --config_out configs/aaai2027/rc_energy.json

PYTHONPATH=src $P scripts/gate_rc_energy.py \
  --cache outputs/rc_energy/two_group_cache.npz \
  --bootstrap 5000 --seed 2027 \
  --lpips_margin 0.002 --mse_relative_margin 0.02 \
  --out outputs/rc_energy/two_group_gate.json
```

`RC_ENERGY_CACHE_OK` / `RC_ENERGY_GATE_OK` 只表示程序完成，方法判决看 `[verdict]`。本实验已经 RED，以下命令保留为被禁止的历史预注册方案，不得运行：

```bash
PYTHONPATH=src $P scripts/train_grpo.py \
  --rewards a0faithful,a0faithful_tok,raw_energy_point,rc_energy_point,raw_energy,rc_energy \
  --modes gt_only --seeds 0,1,2 \
  --steps 150 --K 16 --batch_windows 2 \
  --train_windows 24 --eval_windows 12 --lr 1e-5 --eval_every 10 \
  --energy_config configs/aaai2027/rc_energy.json \
  --deterministic --out_dir outputs/rc_energy/factorial_s0_2
```

### 0.1 RCAV Gate A-v2：action observability dissection（不训练 world model）

D1-v1 已正式 RED；以下实验定位是时间对齐、空间压缩还是 episode 域移导致失败。禁止继续运行旧 `gate_action_candidates.py`，也禁止在本 gate 前训练 RCAV。

```bash
# A2.1-cache: 保存全部 frame-level FSQ codes；预计 1--3 分钟
PYTHONPATH=src $P scripts/cache_action_observability.py \
  --batch_size 16 --device cuda \
  --out outputs/rcav/action_observability_codes.npz

# A2.1-audit: 有限预注册网格；预计 5--20 分钟
PYTHONPATH=src $P scripts/audit_action_observability.py \
  --cache outputs/rcav/action_observability_codes.npz \
  --horizons 1,2,3,4 --offsets=-1,0,1 \
  --targets first,mean --pools 4x5,8x10 \
  --split_seed 2027 --permutations 200 --bootstrap 2000 \
  --device cuda \
  --out outputs/rcav/action_observability_gate_a2.json
```

成功标记依次为 `ACTION_OBSERVABILITY_CACHE_OK`、`ACTION_OBSERVABILITY_AUDIT_OK`。成功标记只说明程序完成；方法判决必须看 `[verdict]` 和 split diagnosis。A2.1 后先分析，不直接运行候选 gate。

### 0.2 RCAV Gate A2.2：decoded-motion oracle

A2.1 已 RED。以下 oracle 只测真实 decoded motion 是否含有跨 episode 动作信息，不是 reward，也不训练 world model。

```bash
PYTHONPATH=src $P scripts/cache_action_motion_oracle.py \
  --horizons 1,2,3,4 --pools 4x5,8x10 \
  --batch_size 16 --device cuda \
  --out outputs/rcav/action_motion_oracle.npz

PYTHONPATH=src $P scripts/audit_action_observability.py \
  --source motion --cache outputs/rcav/action_motion_oracle.npz \
  --horizons 1,2,3,4 --offsets=-1,0,1 \
  --targets first,mean --pools 4x5,8x10 \
  --split_seed 2027 --permutations 200 --bootstrap 2000 \
  --device cuda \
  --out outputs/rcav/action_motion_oracle_gate_a2.json
```

成功标记：`ACTION_MOTION_ORACLE_CACHE_OK` 与 `ACTION_OBSERVABILITY_AUDIT_OK`。oracle RED 时停止 action axis；不得继续 candidate gate 或 learned CNN 搜索。

### 0.3 RCAV Gate A2.3：command-aligned motion

A2.2 虽为 GREEN，但选中 `offset=-1`，不能直接验证 prompt 中的当前命令。复用已有 RAFT cache，只运行以下约束 gate：

```bash
PYTHONPATH=src $P scripts/audit_action_observability.py \
  --source motion --cache outputs/rcav/action_motion_oracle.npz \
  --horizons 1,2,3,4 --offsets=0 \
  --targets mean --pools 4x5,8x10 \
  --split_seed 2027 --permutations 200 --bootstrap 2000 \
  --device cuda \
  --out outputs/rcav/action_motion_command_gate_a23.json
```

只有该约束 gate 为 GREEN，才实现 candidate-level verifier；`offset=-1` 的 A2.2 结果不具备替代资格。

### 0.4 RCAV Gate A2.4：nested episode cross-fit

A2.3 仅 retrieval CI 下界未过门，不改阈值；用所有 20 episodes 做严格 outer OOF 估计：

```bash
PYTHONPATH=src $P scripts/crossfit_action_motion.py \
  --cache outputs/rcav/action_motion_oracle.npz \
  --horizons 1,2,3,4 --pools 4x5,8x10 \
  --outer_folds 5 --split_seed 2027 \
  --permutations 200 --bootstrap 5000 \
  --device cuda \
  --out outputs/rcav/action_motion_command_crossfit_a24.json
```

四项 gate 与 A2.3 完全相同。A2.4 RED 时不再改变统计协议；下一步只能增加独立 episodes 或停止 action axis。

### 0.5 RCAV Gate B：generated-candidate command specificity

A2.4 已 GREEN。先拟合并冻结五个 outer-fold payload，再运行 base-policy 候选门控：

```bash
PYTHONPATH=src $P scripts/fit_motion_action_payload.py \
  --cache outputs/rcav/action_motion_oracle.npz \
  --crossfit outputs/rcav/action_motion_command_crossfit_a24.json \
  --horizon 1 --pool 8x10 --split_seed 2027 \
  --out outputs/rcav/motion_action_payload_h1_8x10.npz

PYTHONPATH=src $P scripts/gate_motion_action_candidates.py \
  --payload outputs/rcav/motion_action_payload_h1_8x10.npz \
  --n_windows 80 --stride 5 --K 16 --top_k 100 \
  --seed 7401 --bootstrap 5000 --device cuda \
  --cache_out outputs/rcav/motion_action_candidate_gate.npz \
  --report_out outputs/rcav/motion_action_candidate_gate.json
```

成功标记：`MOTION_ACTION_PAYLOAD_OK`、`MOTION_ACTION_CANDIDATE_GATE_OK`。只有 Gate B 四项 primary 同时 GREEN，才实现 GRPO reward arm。

### 0.6 RCAV Gate B2：独立 candidate replication

B1 已 RED 但均值方向全正。固定 contexts，只复现一次独立生成采样：

```bash
PYTHONPATH=src $P scripts/gate_motion_action_candidates.py \
  --payload outputs/rcav/motion_action_payload_h1_8x10.npz \
  --n_windows 80 --stride 5 --K 16 --top_k 100 \
  --window_seed 7401 --generation_seed 17401 \
  --bootstrap 5000 --device cuda \
  --cache_out outputs/rcav/motion_action_candidate_gate_rep2.npz \
  --report_out outputs/rcav/motion_action_candidate_gate_rep2.json

PYTHONPATH=src $P scripts/analyze_motion_candidate_replications.py \
  --rep1 outputs/rcav/motion_action_candidate_gate.npz \
  --rep2 outputs/rcav/motion_action_candidate_gate_rep2.npz \
  --bootstrap 10000 --seed 2027 \
  --out outputs/rcav/motion_action_candidate_gate_combined.json
```

combined 判决为最终 Gate B。RED 后停止 RCAV，不增加第三个 seed、不进入 GRPO。

**终局：Gate B combined RED。** rho $+0.060$ [−0.012,+0.127]；matched−shuffled $+0.063$ [−0.035,+0.153]。本节命令不得再次运行第三个 seed，RCAV 不进入后续训练。

## 1. C1：Local Metric-Target Gate（先跑，约数分钟）

```bash
PYTHONPATH=src $P scripts/probe_reachable_projection.py \
  --n_windows 64 --exclude_windows 36 --window_seed 1 \
  --positions 8 --rounds 2 --metric_batch 8 \
  --deterministic \
  --out outputs/analysis/reachable_projection_w64.json
```

固定判决：报告 improved fraction、mean/median/q05 gain 与 Hamming fraction，不根据结果改搜索预算。若多数窗口改善且 mean gain 为正，`D(E(s'))` 不能再称 metric projection；下一步才实现 calibration-only MPRT cache、same-candidate replay 与 preprocessing controls。若门控为红，保留 encoder-RC，停止 MPRT 分支。

## 2. C1：缓存 MRRT 并运行四臂训练（约 9 小时）

```bash
PYTHONPATH=src $P scripts/cache_mrrt_targets.py \
  --train_windows 24 --eval_windows 12 --window_seed 1 \
  --positions 8 --rounds 2 --metric_batch 8 --deterministic \
  --out outputs/mrrt/train_targets.npz
```

缓存必须报告 `MRRT_CACHE_OK`。随后在原始单步 GRPO 下同时运行四臂：

```bash
PYTHONPATH=src $P scripts/train_grpo.py \
  --rewards a0faithful,a0faithful_tok,mrrt,mrrt_random \
  --modes gt_only --seeds 0,1,2,3,4 \
  --steps 150 --K 16 --batch_windows 2 \
  --train_windows 24 --eval_windows 12 --lr 1e-5 --eval_every 10 \
  --reachable_target_cache outputs/mrrt/train_targets.npz \
  --deterministic --out_dir outputs/mrrt/four_arm
```

```bash
PYTHONPATH=src $P scripts/analyze_mrrt_training.py \
  --raw 'outputs/mrrt/four_arm/sweep_a0faithful_gt_only_s*.json' \
  --encoder_rc 'outputs/mrrt/four_arm/sweep_a0faithful_tok_gt_only_s*.json' \
  --mrrt 'outputs/mrrt/four_arm/sweep_mrrt_gt_only_s*.json' \
  --random 'outputs/mrrt/four_arm/sweep_mrrt_random_gt_only_s*.json' \
  --expected_n 5 --out outputs/analysis/mrrt_four_arm_s0_4.json
```

Primary：held-out raw-GT LPIPS。MRRT 必须同时优于 encoder-RC 和 matched-random；MSE/SSIM/flow
作为 secondary/boundary 全量报告。若只改善自己的 target objective 而不改善 held-out raw-GT 指标，
MRRT 降级为诊断，不进入 headline。

## 3. C2/C3：官方 KL 下的 2×2 复核（第一阶段 seeds 0--4，约 4 小时剩余）

### 2.0 Calibration--Credit Coupling 机制门（已完成）

```bash
PYTHONPATH=src $P scripts/audit_calibration_credit_coupling.py \
  --cache outputs/analysis/temporal_reliability_cache.npz \
  --gamma 0.95 --bootstrap 2000 \
  --out outputs/analysis/calibration_credit_coupling.json
```

正式结果为 GREEN：aggregate $\Delta\rho=+0.0132$、$\Delta\mathrm{flip}=-0.0082$，且 earliest-minus-latest residual dispersion 为 $+0.01272$；三项 episode-cluster bootstrap 95% CI 均严格离开零。该门不需要重复运行，除非 candidate cache 或正式 $\gamma$ 改变。

### 2.1 Sequence-level，raw + RC

```bash
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards raw,rc --adv_temporal seq --seeds 0,1,2,3,4 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --return_horizon 0 --horizon_kl_alpha 0.0 --eval_every 10 \
  --which rlvr --deterministic --out_dir outputs/msp_lvkl_seq
```

### 2.2 Temporal return，raw + RC

```bash
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards raw,rc --adv_temporal return --seeds 0,1,2,3,4 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --return_horizon 0 --horizon_kl_alpha 0.0 --eval_every 10 \
  --which rlvr --deterministic --out_dir outputs/msp_lvkl_return
```

### 2.3 因子分析

```bash
PYTHONPATH=src $P scripts/analyze_msp_factorial.py \
  --seq_raw 'outputs/msp_lvkl_seq/sweep_raw_msp_s*.json' \
  --seq_rc 'outputs/msp_lvkl_seq/sweep_rc_msp_s*.json' \
  --return_raw 'outputs/msp_lvkl_return/sweep_raw_msp_s*.json' \
  --return_rc 'outputs/msp_lvkl_return/sweep_rc_msp_s*.json' \
  --expected_n 5 --out outputs/analysis/msp_lvkl_factorial_s0_4.json
```

扩展规则分开判决：

- **C2 gate**：`return effect under RC` mean $<0$ 且至少 4/5 同向；
- **C3 provisional gate**：primary LPIPS interaction mean $<0$、至少 4/5 同向，且 `return-RC` 为四臂最低 mean；
- C2 或 C3 任一通过，都原样补齐四臂 seeds 5--9，保证固定 $n=10$ difference-in-differences 可计算；不因某个 seed 好看改变协议；
- C3 正式门还要求 $n=10$ interaction 至少 7/10 同向、单侧 exact sign-flip $p<0.05$、bootstrap 95% CI 上界 $<0$。`full-stack best` 不能替代该检验。

## 4. C2：时间对应控制（2×2 通过后；第一阶段 paired 3 seeds，约 3 小时）

Full arm 可直接复用 `outputs/msp_lvkl_return` 的 RC seeds 0--2。只新增三组控制：

```bash
# L=1：仅当前 frame reward
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards rc --adv_temporal return --return_horizon 1 --seeds 0,1,2 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --eval_every 10 --which rlvr --deterministic --out_dir outputs/msp_lvkl_return_L1

# L=3：局部 future credit
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards rc --adv_temporal return --return_horizon 3 --seeds 0,1,2 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --eval_every 10 --which rlvr --deterministic --out_dir outputs/msp_lvkl_return_L3

# 每个 horizon 保持 reward multiset，只打乱 candidate 的时间身份
PYTHONPATH=src $P scripts/train_grpo_msp.py \
  --rewards rc --adv_temporal shuffled_return --return_horizon 0 --seeds 0,1,2 \
  --T 8 --K 16 --steps 30 --batch_windows 2 --train_windows 24 --eval_windows 8 \
  --lr 1e-5 --kl 0.001 --kl_type low_var_kl --temporal_gamma 0.95 \
  --eval_every 10 --which rlvr --deterministic --out_dir outputs/msp_lvkl_return_shuffled
```

```bash
PYTHONPATH=src $P scripts/analyze_temporal_controls.py \
  --trunc1 'outputs/msp_lvkl_return_L1/sweep_rc_msp_s*.json' \
  --trunc3 'outputs/msp_lvkl_return_L3/sweep_rc_msp_s*.json' \
  --full 'outputs/msp_lvkl_return/sweep_rc_msp_s[0-2].json' \
  --shuffled 'outputs/msp_lvkl_return_shuffled/sweep_rc_msp_s*.json' \
  --expected_n 3 --out outputs/analysis/msp_lvkl_temporal_controls_s0_2.json
```

判决：primary 为 full aligned 相对 shuffled 的 LPIPS 配对差；同时要求 full 不差于 L=1，且 L=1→L=3→full 有基本剂量方向。通过后把三组控制扩到 seeds 3--4；未通过则 C2 只能表述为有效的 block-wise objective，不能声称提升来自正确 temporal correspondence。

## 5. 暂缓实验

- `T=4/6/8` 长度泛化：仅在第 2、3 节通过后运行；
- MPRT 训练：仅在第 1 节正式 gate 通过、且 same-candidate replay 优于 encoder-RC 后运行；
- 新 reward panel、rank weighting、GSPO/REAL/GSPO 等已判负分支不得重启。

## 6. 完成标志

- local target gate：`REACHABLE_PROJECTION_OK`；
- MRRT cache / analysis：`MRRT_CACHE_OK` / `MRRT_ANALYSIS_OK`；
- training sweep：`GRPO_MSP_OK`；
- factorial：`MSP_FACTORIAL_OK`；
- temporal controls：`TEMPORAL_CONTROLS_OK`。
