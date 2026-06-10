# Codex 任务说明：构建 Vote2World 单步无 GT 输入管线与核查脚本

> **用途**：交给 Codex 执行  
> **阶段定位**：输入侧工程实现与验证  
> **任务目标**：在不修改 RLVR-World 官方模型输入格式的前提下，编写一组用于 Vote2World 的输入准备、动作 schema 核查、无 GT 适配样本构造与接口验收脚本。  
> **重要约束**：本阶段只实现输入管线和验证脚本，不实现候选采样奖励、不实现 GRPO、不训练模型、不引入 horizon guard。

---

## 0. 背景与当前决策

我们计划基于 RLVR-World 官方 RT-1 单步视觉世界模型权重，实现一个 **future-GT-free self-consensus RL post-training** 方法。

第一版方法的目标是：

\[
(o_{t-3:t}, a_{t-3:t})
\rightarrow
K \text{ 个候选下一帧}
\rightarrow
\text{候选之间自共识奖励}
\rightarrow
\text{GRPO}
\]

当前阶段只处理左侧输入部分。

### 当前已经确认的设计原则

1. **世界模型输入与 RLVR-World 保持一致**；
2. 使用最近 4 张历史图像：
   \[
   o_{t-3}, o_{t-2}, o_{t-1}, o_t
   \]
3. 使用对应的 4 个动作向量：
   \[
   a_{t-3}, a_{t-2}, a_{t-1}, a_t
   \]
4. 每个动作向量保留完整 13 维；
5. 动作逐维量化到 256 bins；
6. 图像先通过官方 visual tokenizer 编码；
7. 单步模型预测下一帧视觉 token；
8. 未来 GT：
   \[
   o_{t+1}^{\mathrm{gt}}
   \]
   在 adaptation 路径中必须不可访问；
9. GT 只允许在独立 evaluation 路径中使用；
10. 奖励侧后续会按动作语义字段拆分，不允许直接使用完整 13 维范数。

---

## 1. 本阶段要实现的工程目标

请在现有项目中新增一组独立脚本和模块，用于：

1. 核查并冻结 RT-1 动作字段顺序；
2. 生成正式的 `action_schema.json`；
3. 构建 **generation-only adaptation dataloader**；
4. 构建独立的 evaluation dataloader；
5. 验证适配路径无法读取未来 GT；
6. 验证输入帧和动作的时间对齐；
7. 验证每个动作维度的语义切片；
8. 验证官方 tokenizer / processor 输入长度；
9. 输出一份完整的输入侧验收报告。

本阶段完成后，应能够稳定回答：

\[
\boxed{
\text{Vote2World 单步无 GT 后训练到底向模型输入什么？}
}
\]

以及：

\[
\boxed{
\text{奖励模块未来应该从当前动作中读取哪些语义字段？}
}
\]

---

## 2. 本阶段禁止事项

请严格遵守以下边界：

- 不修改 RLVR-World 官方 Transformer 架构；
- 不修改 visual tokenizer；
- 不重新预训练 tokenizer；
- 不重新预训练 single-step Transformer；
- 不实现 GRPO；
- 不实现 self-consensus reward；
- 不加入 horizon guard；
- 不实现多步预测；
- 不把自然语言指令加入模型输入；
- 不删除 13 维动作中的任何字段；
- 不把 future GT 传给 adaptation dataloader；
- 不硬编码未经验证的动作索引；
- 不使用完整 13 维动作范数作为物理运动强度；
- 不大规模下载完整 RT-1 数据；
- 不对服务器已有数据和项目目录做破坏性操作。

---

## 3. 已知输入结构

### 3.1 单步样本时间结构

官方单步任务使用：

\[
(o_{t-3:t}, a_{t-3:t})
\rightarrow
\hat{o}_{t+1}
\]

对应：

```text
o_{t-3} + a_{t-3}
        ↓
o_{t-2} + a_{t-2}
        ↓
o_{t-1} + a_{t-1}
        ↓
o_t     + a_t
        ↓
预测 o_{t+1}
```

其中：

\[
a_\tau
\]

表示在状态：

\[
o_\tau
\]

下执行动作，使环境转移到：

\[
o_{\tau+1}.
\]

### 3.2 官方单步关键参数

| 参数 | 当前已知值 |
|---|---:|
| `context_length` | 4 |
| `segment_length` | 5 |
| `action_dim` | 13 |
| `action_bins` | 256 |
| `visual_token_num` | 4375 |
| 每帧视觉 token 数 | 320 |
| 生成输入 token 长度 | 1333 |
| 生成输出 token 长度 | 321 |
| processor 类型 | `simple` |

### 3.3 单步 token 结构

每个历史时间步：

\[
320 \text{ 个视觉 token}
+
13 \text{ 个动作 token}
=
333 \text{ 个 token}
\]

四步历史：

\[
333 \times 4 = 1332
\]

加上 BOS：

\[
1332 + 1 = 1333
\]

模型生成：

\[
320 \text{ 个下一帧视觉 token}
+
1 \text{ 个 EOS}
=
321
\]

---

## 4. 动作 schema：需要核查并固化

### 4.1 原始 RT-1 动作字段

当前核查结果显示，原始动作字段包括：

| 字段 | 维度 | 语义 |
|---|---:|---|
| `gripper_closedness_action` | 1 | 夹爪开合 |
| `terminate_episode` | 3 | episode 模式 / 终止相关表示 |
| `base_displacement_vector` | 2 | 底盘平移 |
| `rotation_delta` | 3 | 机械臂旋转 |
| `base_displacement_vertical_rotation` | 1 | 底盘旋转 |
| `world_vector` | 3 | 机械臂末端平移 |

合计：

\[
1 + 3 + 2 + 3 + 1 + 3 = 13
\]

### 4.2 图中 7 维机械臂动作

论文图中展示的：

\[
(x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw},\mathrm{gripper\ openness})
\]

对应：

\[
\texttt{world\_vector}
+
\texttt{rotation\_delta}
+
\texttt{gripper\_closedness\_action}
\]

但这 7 个维度不一定是展平向量的前 7 维。

### 4.3 当前 provisional 索引映射

基于 metadata 顺序，当前推测为：

| 索引范围 | 字段 | 维度 |
|---|---|---:|
| `[0:1]` | `gripper_closedness_action` | 1 |
| `[1:4]` | `terminate_episode` | 3 |
| `[4:6]` | `base_displacement_vector` | 2 |
| `[6:9]` | `rotation_delta` | 3 |
| `[9:10]` | `base_displacement_vertical_rotation` | 1 |
| `[10:13]` | `world_vector` | 3 |

该映射在官方 converter 实际运行前只能标记为：

```text
provisional
```

### 4.4 必须完成的核查

请不要直接使用上面的 provisional 映射。

需要真实执行或等价复现以下验证：

1. 读取一个原始 episode；
2. 打印：
   ```text
   step["action"].keys()
   ```
3. 记录真实迭代顺序；
4. 按官方 converter 的逻辑展平动作；
5. 输出展平后的 13 维向量；
6. 对每个字段的原始值与展平后的切片做逐项一致性检查；
7. 生成正式：
   ```text
   configs/vote2world/action_schema.json
   ```

---

## 5. 要求生成的文件

请新增以下文件。命名可按现有项目结构微调，但职责必须保持清晰。

### 5.1 `configs/vote2world/action_schema.json`

用途：

- 固化 RT-1 13 维动作字段切片；
- 提供给 adaptation dataloader；
- 提供给未来 reward module；
- 避免在多个脚本中重复硬编码索引。

必须包含：

| 字段 | 要求 |
|---|---|
| dataset name | `fractal20220817_data` |
| schema status | `confirmed` 或 `provisional` |
| flatten key order | 原始字段真实拼接顺序 |
| field slices | 每个字段的起止索引 |
| field dimensions | 每个字段维度 |
| semantic group | `translation`、`rotation`、`gripper`、`base`、`mode` |
| include in model input | 全部为 `true` |
| include in motion magnitude | 按语义设置 |
| evidence | 核查来源与时间 |

### 5.2 `scripts/audit_rt1_action_schema.*`

用途：

- 核查原始动作字段；
- 打印真实 key 顺序；
- 展平动作；
- 逐切片对齐检查；
- 生成 `action_schema.json`；
- 输出 markdown 报告。

要求：

- 只处理少量 episode；
- 不下载完整数据；
- 不修改原始数据；
- 若 TFDS 不可用，应输出明确错误和安装建议；
- 若 schema 与当前 provisional 映射不同，应显式报出差异；
- 不允许静默覆盖已有 confirmed schema；
- 覆盖前必须备份并记录变更。

### 5.3 `vote2world/data/adaptation_dataset.*`

用途：

构建 generation-only adaptation 样本。

每个样本只返回：

| 字段 | shape | 用途 |
|---|---|---|
| `context_frames` | `(4, C, H, W)` | 世界模型视觉条件 |
| `context_actions` | `(4, 13)` | 世界模型动作条件 |
| `sample_id` | 标量或字符串 | 日志、复现、缓存索引 |
| `episode_id` | 标量或字符串 | 追踪 episode |
| `start_index` | 标量 | 追踪时间窗口 |

禁止返回：

```text
target_frame
future_frame
ground_truth
label
next_observation
```

### 5.4 `vote2world/data/evaluation_dataset.*`

用途：

构建独立评估样本。

每个样本返回：

| 字段 | shape | 用途 |
|---|---|---|
| `context_frames` | `(4, C, H, W)` | 世界模型视觉条件 |
| `context_actions` | `(4, 13)` | 世界模型动作条件 |
| `target_frame` | `(C, H, W)` | 仅评估使用 |
| `sample_id` | 标量或字符串 | 日志、复现 |
| `episode_id` | 标量或字符串 | 追踪 episode |
| `start_index` | 标量 | 追踪时间窗口 |

注意：

- evaluation dataset 与 adaptation dataset 必须是独立类或独立模式；
- adaptation 代码路径不得 import 或调用 `target_frame`；
- target GT 不得写入 adaptation batch；
- 后续 reward module 不得接收 evaluation dataset batch。

### 5.5 `scripts/validate_vote2world_input_pipeline.*`

用途：

完成输入链路验收。

需要验证：

1. adaptation dataset 单个样本字段；
2. evaluation dataset 单个样本字段；
3. adaptation batch 不包含 future GT；
4. context frame 数量为 4；
5. context action 数量为 4；
6. 每个动作维度为 13；
7. action schema 与实际数据一致；
8. 当前帧为：
   \[
   o_t = \texttt{context\_frames[-1]}
   \]
9. 当前动作为：
   \[
   a_t = \texttt{context\_actions[-1]}
   \]
10. 时间窗口不存在 off-by-one；
11. tokenizer 输出每帧 320 个视觉 token；
12. 量化后每步动作得到 13 个动作 token；
13. 4 步历史拼接后长度为 1332；
14. 加 BOS 后生成输入长度为 1333；
15. 模型下一帧输出长度为 321；
16. decoder 可以从下一帧 token 还原预测图像。

### 5.6 `tests/test_no_future_gt_leakage.*`

用途：

专门验证无 GT 适配路径。

必须覆盖：

- adaptation dataset 的返回字段中不存在 future GT；
- adaptation batch 序列化后不存在 future GT；
- adaptation processor 接口不要求 target frame；
- adaptation rollout 路径不加载 target frame；
- future GT 只在 evaluation dataset 中存在；
- future GT 不出现在 reward 预留接口；
- 任何 adaptation 路径访问 `target_frame` 时必须立即失败。

### 5.7 `reports/vote2world_input_pipeline_report.md`

用途：

输出最终输入侧报告。

报告必须包含：

1. 已确认 schema；
2. 13 维动作字段切片；
3. 官方输入结构；
4. adaptation dataset 结构；
5. evaluation dataset 结构；
6. token 长度核查；
7. 时间对齐核查；
8. future GT 隔离测试结果；
9. 输入样例；
10. 未解决问题；
11. 是否满足进入候选采样阶段的条件。

---

## 6. Adaptation Dataset 设计要求

### 6.1 单个样本构造

对于 episode：

\[
\{o_0,\ldots,o_{T-1}\}
\]

以及：

\[
\{a_0,\ldots,a_{T-1}\}
\]

构造样本窗口：

\[
(o_{t-3:t}, a_{t-3:t})
\]

有效范围至少满足：

\[
t-3 \ge 0
\]

并且评估时：

\[
t+1 < T
\]

adaptation 阶段虽然不返回：

\[
o_{t+1}
\]

但为了保证未来可离线评估，建议保留可追溯的：

```text
episode_id
start_index
sample_id
```

不要在 adaptation batch 中嵌入 GT。

### 6.2 时间对齐要求

必须验证：

```text
context_frames[0]  ↔ o_{t-3}
context_actions[0] ↔ a_{t-3}

context_frames[1]  ↔ o_{t-2}
context_actions[1] ↔ a_{t-2}

context_frames[2]  ↔ o_{t-1}
context_actions[2] ↔ a_{t-1}

context_frames[3]  ↔ o_t
context_actions[3] ↔ a_t
```

模型目标为：

```text
target_frame ↔ o_{t+1}
```

但 adaptation dataset 不返回该目标。

### 6.3 当前动作与奖励模块衔接

未来 reward module 只需读取：

```text
current_frame  = context_frames[-1]
current_action = context_actions[-1]
```

因此 adaptation dataset 应保证：

- 最后一张 frame 始终是当前帧；
- 最后一个 action 始终是当前动作；
- 历史顺序不可打乱；
- 不进行会破坏时间顺序的数据增强。

---

## 7. 动作 schema 的奖励侧语义分组

虽然本阶段不实现奖励函数，但输入脚本必须为后续 reward module 预留以下语义切片。

### 7.1 机械臂平移

字段：

```text
world_vector
```

语义：

\[
(x,y,z)
\]

未来用途：

- 主动作幅度分桶；
- 静态复制过滤；
- 动作条件统计。

### 7.2 机械臂旋转

字段：

```text
rotation_delta
```

语义：

\[
(\mathrm{roll},\mathrm{pitch},\mathrm{yaw})
\]

未来用途：

- 单独旋转幅度统计；
- 静态复制过滤辅助项；
- 不与平移直接混用原始尺度。

### 7.3 夹爪事件

字段：

```text
gripper_closedness_action
```

未来用途：

- 夹爪开合事件判断；
- 单独处理；
- 不并入连续位姿范数。

### 7.4 底盘动作

字段：

```text
base_displacement_vector
base_displacement_vertical_rotation
```

当前少量 RT-1 样本中可能为全 0。

未来用途：

- 模型输入保留；
- 奖励侧独立统计；
- 不与机械臂幅度直接混合；
- 若扩展至移动底盘环境，单独设置阈值。

### 7.5 episode 模式字段

字段：

```text
terminate_episode
```

未来用途：

- 仅记录；
- 必要时过滤 terminal step；
- 不参与运动强度；
- 不参与静态复制门控。

---

## 8. 模型输入侧要求

### 8.1 不删除任何动作字段

即使当前样本中底盘动作全部为 0，也必须保留完整 13 维：

\[
a_t \in \mathbb{R}^{13}
\]

原因：

1. 官方 checkpoint 已按 13 维输入训练；
2. 删除字段会改变 token 布局；
3. 影响 checkpoint 兼容性；
4. 影响与 RLVR-World 的公平对比；
5. 后续目标域可能出现非零底盘动作。

### 8.2 不加入自然语言

RT-1 原始数据可能包含语言指令，但 RLVR-World 单步视频模型链路没有使用该字段。

第一版禁止加入：

```text
natural_language_instruction
natural_language_embedding
```

原因：

- 会改变模型输入接口；
- 无法直接复用官方权重；
- 引入新的变量；
- 不利于验证 reward 替换本身的效果。

### 8.3 保留官方量化方式

每个动作维度：

1. 按 `action_ranges_path` 读取 min/max；
2. 归一化至 `[0,1]`；
3. clip；
4. 乘以 `action_bins=256`；
5. floor；
6. 再次 clip 至 `[0,255]`；
7. 添加视觉 token offset。

不要自行替换为：

- z-score；
- k-means；
- 动作联合编码；
- 自定义 tokenizer；
- 动态 bin 数。

这些可留作后续研究，但不属于第一版。

---

## 9. Adaptation Processor 设计要求

官方 MLE processor 会读取长度为 5 的 segment，并把第 5 帧作为 response。

Vote2World 需要新增一个 **generation-only adaptation processor**，其职责为：

1. 接收：
   ```text
   context_frames:  (B, 4, C, H, W)
   context_actions: (B, 4, 13)
   ```
2. 编码 4 张历史帧；
3. 量化 4 个动作；
4. 按官方顺序拼接 token；
5. 添加 BOS；
6. 输出：
   ```text
   gen_input_ids
   attention_mask
   position_ids
   metadata
   ```
7. 不生成 labels；
8. 不读取 future frame；
9. 不需要 target token；
10. 不返回 future GT。

### 9.1 输出长度验收

必须验证：

\[
(320+13)\times 4 + 1 = 1333
\]

即：

```text
gen_input_ids.shape[-1] == 1333
```

### 9.2 不允许的行为

adaptation processor 不得：

- 编码第 5 帧；
- 构造 `labels`；
- 构造 `target_tokens`；
- 接收 `target_frame`；
- 从 evaluation batch 中取 future GT；
- 静默复用 MLE processor 的 future-frame response 路径。

---

## 10. Evaluation Processor 设计要求

evaluation processor 可以复用官方评估逻辑，但必须与 adaptation processor 分开。

允许读取：

\[
o_{t+1}^{\mathrm{gt}}
\]

用途仅限：

- MSE；
- PSNR；
- SSIM；
- LPIPS；
- 定性可视化；
- 后续 proxy-reward correlation 分析。

禁止将 evaluation target 传回 adaptation trainer。

---

## 11. 动作统计脚本要求

请新增或复用动作统计脚本，并输出：

### 11.1 每个字段统计

| 字段 | min | max | mean | std | exact-zero ratio |
|---|---:|---:|---:|---:|---:|

### 11.2 语义组统计

| 指标 | 统计要求 |
|---|---|
| `arm_translation_norm` | min / max / mean / std / zero ratio / 分位数 |
| `arm_rotation_norm` | min / max / mean / std / zero ratio / 分位数 |
| `gripper_abs` | min / max / mean / std / zero ratio / 分位数 |
| `base_translation_norm` | 同上 |
| `base_rotation_abs` | 同上 |
| `terminate_episode` | 离散模式计数 |

### 11.3 分位数

至少报告：

```text
p00
p10
p25
p50
p75
p90
p95
p99
p100
```

### 11.4 后续奖励设计建议

统计脚本只输出建议，不实现奖励：

- 哪些字段适合作为主动作幅度分桶；
- 哪些字段需要单独阈值；
- 哪些字段必须排除；
- terminal step 是否建议排除；
- 零动作桶比例；
- 是否需要按平移 / 旋转 / 夹爪事件分别分析。

---

## 12. Future-GT 泄漏防护要求

### 12.1 代码层隔离

请确保：

```text
adaptation_dataset
adaptation_processor
adaptation_rollout
```

三条路径均无法读取：

```text
target_frame
next_frame
future_frame
ground_truth
label
```

### 12.2 报错机制

若 adaptation 路径收到 GT 字段，应主动报错。

不要：

- 静默忽略；
- 自动删除后继续；
- 仅打印 warning。

必须失败并提示：

```text
Future-GT leakage detected in adaptation path.
```

### 12.3 数据缓存隔离

后续若缓存候选预测，建议目录分开：

```text
cache/
├── adaptation_inputs/
├── candidate_predictions/
└── evaluation_only/
```

其中：

```text
evaluation_only/
```

可以包含 GT。

其他目录禁止保存 GT。

---

## 13. 验收报告要求

最终生成：

```text
reports/vote2world_input_pipeline_report.md
```

必须回答以下问题。

### 13.1 Schema 相关

1. 原始 `step["action"].keys()` 的真实顺序是什么？
2. 展平动作最终是否为 13 维？
3. 每个字段对应哪些索引？
4. provisional schema 与 confirmed schema 是否一致？
5. 是否已生成 `action_schema.json`？

### 13.2 输入相关

1. adaptation dataset 返回哪些字段？
2. evaluation dataset 返回哪些字段？
3. context frame 数量是否为 4？
4. context action 数量是否为 4？
5. 每个动作是否为 13 维？
6. 当前帧是否为 `context_frames[-1]`？
7. 当前动作是否为 `context_actions[-1]`？
8. 时间对齐是否正确？

### 13.3 Token 相关

1. 每帧视觉 token 是否为 320？
2. 每步动作 token 是否为 13？
3. 四步历史 token 是否为 1332？
4. 加 BOS 后是否为 1333？
5. 生成输出是否为 321？
6. decoder 是否可正常输出下一帧图像？

### 13.4 泄漏防护相关

1. adaptation dataset 是否不返回 future GT？
2. adaptation processor 是否不读取 future GT？
3. adaptation rollout 是否不读取 future GT？
4. future GT 是否只存在于 evaluation 路径？
5. 泄漏测试是否全部通过？

### 13.5 是否允许进入下一阶段

请在报告结尾明确给出：

```text
READY_FOR_CANDIDATE_SAMPLING = YES / NO
```

若为 `NO`，列出阻塞项。

---

## 14. 单元测试清单

请至少覆盖以下测试。

### 14.1 Dataset tests

- adaptation sample 字段白名单测试；
- evaluation sample 字段测试；
- context frame shape 测试；
- context action shape 测试；
- episode 边界测试；
- 不足 5 帧 episode 处理测试；
- sample_id 可追溯性测试。

### 14.2 Schema tests

- 13 维总长度测试；
- 字段切片无重叠测试；
- 字段切片无遗漏测试；
- 原始字段与展平切片一致性测试；
- confirmed schema 不允许静默覆盖测试。

### 14.3 Processor tests

- 视觉 token 数量测试；
- 动作 token 数量测试；
- 输入长度 1333 测试；
- BOS 位置测试；
- 动作 offset 测试；
- decoder 输出图像 shape 测试。

### 14.4 Leakage tests

- adaptation dataset 无 GT；
- adaptation processor 无 GT；
- adaptation rollout 无 GT；
- evaluation GT 独立存在；
- 传入 GT 时主动失败。

---

## 15. 日志要求

运行输入核查脚本时，至少记录：

| 日志项 | 说明 |
|---|---|
| repo commit SHA | 锁定官方代码版本 |
| dataset path | 原始数据路径 |
| processed data path | 转换后数据路径 |
| number of episodes | 核查 episode 数 |
| number of steps | 核查 step 数 |
| schema status | confirmed / provisional |
| action key order | 实际顺序 |
| flattened action dim | 应为 13 |
| context length | 应为 4 |
| segment length | 应为 5 |
| action bins | 应为 256 |
| tokens per frame | 应为 320 |
| gen input length | 应为 1333 |
| gen output length | 应为 321 |
| leakage tests | pass / fail |
| unresolved blockers | 列表 |

---

## 16. 推荐目录结构

可根据现有项目调整，但建议保持以下职责分离。

```text
vote2world/
├── configs/
│   └── vote2world/
│       └── action_schema.json
├── scripts/
│   ├── audit_rt1_action_schema.*
│   ├── analyze_rt1_action_statistics.*
│   └── validate_vote2world_input_pipeline.*
├── vote2world/
│   ├── data/
│   │   ├── adaptation_dataset.*
│   │   └── evaluation_dataset.*
│   └── processors/
│       └── adaptation_processor.*
├── tests/
│   ├── test_action_schema.*
│   ├── test_adaptation_dataset.*
│   ├── test_adaptation_processor.*
│   └── test_no_future_gt_leakage.*
├── reports/
│   └── vote2world_input_pipeline_report.md
└── cache/
    ├── adaptation_inputs/
    ├── candidate_predictions/
    └── evaluation_only/
```

---

## 17. Codex 执行顺序

请严格按以下顺序执行。

### 第一步：读取现有核查文档

优先读取：

```text
rlvr_world_input_pipeline.md
rt1_action_schema.md
rt1_action_statistics.md
rt1_npz_audit.md
audit_summary.md
```

总结已确认事实与未确认项。

### 第二步：检查当前项目目录

确认：

- RLVR-World 仓库路径；
- commit SHA；
- RT-1 原始数据路径；
- 是否已有 TFDS 环境；
- 是否已有转换后的 `.npz`；
- 是否已有 provisional schema；
- 是否已有相关输入脚本。

### 第三步：完成真实 schema 核查

- 安装或复用可用 TFDS 环境；
- 只处理少量 episode；
- 输出真实 action key 顺序；
- 生成 confirmed schema；
- 与 provisional schema 对比；
- 保存证据。

### 第四步：构建 adaptation dataset

要求：

- 只返回 4 帧历史与 4 步动作；
- 不返回 future GT；
- 保留 sample_id、episode_id、start_index；
- 严格检查时间窗口。

### 第五步：构建 evaluation dataset

要求：

- 与 adaptation dataset 分离；
- 额外返回 target_frame；
- 仅供评估使用。

### 第六步：构建 adaptation processor

要求：

- 编码 4 张历史帧；
- 量化 4 个动作；
- 拼接为 1333-token 生成输入；
- 不构造 labels；
- 不读取 target frame。

### 第七步：编写验收测试

覆盖：

- schema；
- 时间对齐；
- token 长度；
- decoder；
- leakage。

### 第八步：运行输入验收

输出：

```text
reports/vote2world_input_pipeline_report.md
```

### 第九步：停止

当报告生成后停止。

不要进入：

- 候选采样；
- 共识特征；
- reward；
- GRPO；
- 模型训练。

---

## 18. 验收条件

只有同时满足以下条件，才允许进入下一阶段。

- [ ] 真实 action key 顺序已确认；
- [ ] `action_schema.json` 已生成；
- [ ] schema 状态为 `confirmed`；
- [ ] flattened action dim 为 13；
- [ ] adaptation dataset 不返回 future GT；
- [ ] evaluation dataset 与 adaptation dataset 分离；
- [ ] context frame 数为 4；
- [ ] context action 数为 4；
- [ ] 每步动作维度为 13；
- [ ] 每帧 visual token 数为 320；
- [ ] 每步 action token 数为 13；
- [ ] gen input length 为 1333；
- [ ] gen output length 为 321；
- [ ] decoder 可以还原下一帧图像；
- [ ] leakage tests 全部通过；
- [ ] 输入报告生成；
- [ ] 报告末尾为：
  ```text
  READY_FOR_CANDIDATE_SAMPLING = YES
  ```

---

## 19. 最终说明

本阶段的核心原则是：

\[
\boxed{
\text{世界模型输入完全复用 RLVR-World}
}
\]

即：

\[
\boxed{
(o_{t-3:t},a_{t-3:t})
}
\]

其中：

\[
a_t \in \mathbb{R}^{13}
\]

且：

\[
\boxed{
\text{future GT 不得进入 adaptation 路径}
}
\]

我们的创新不发生在输入结构，而发生在候选输出之后的自共识奖励设计。

完成本阶段后，再进入下一阶段：

\[
\boxed{
\text{同一输入条件下生成 }K\text{ 个候选下一帧}
}
\]

并继续讨论候选采样配置、缓存格式和共识特征空间。
