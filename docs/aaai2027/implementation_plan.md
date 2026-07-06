# 基于 RLVR-World 的方法实现设计：Codec-Calibrated Reward + Codec-Aware Segmental GRPO

> 目标读者：Codex / Claude Code / 其他代码 agent。  
> 目标仓库：https://github.com/thuml/RLVR-World  
> 目标目录：`vid_wm/`，重点是 `vid_wm/verl/` 的 RLVR post-training 代码。  
> 设计原则：尽量少侵入原项目；先保证单步可跑，再扩展多步；保留原始 GRPO 与原始 reward 作为 baseline。

---

## 0. 背景和实现目标

RLVR-World 的视频世界模型 post-training 使用 `verl` 中的 GRPO 流程。仓库 README 中给出的入口包括：

- 单步 RLVR：`vid_wm/verl/examples/grpo_trainer/run_vgpt.sh`
- 多步 RLVR：`vid_wm/verl/examples/grpo_trainer/run_ctx_msp_vgpt.sh`

当前原始视频 RLVR reward 主要基于 decoded prediction 与 GT frame/trajectory 之间的图像指标，例如 LPIPS、MSE、MAE、SSIM、PSNR 等。我们的目标是在不破坏原始流程的情况下实现两个模块：

1. **Codec-Calibrated Token-Visual Reward**
   - 计算 GT 经 encoder-decoder 重建后的 codec reconstruction floor。
   - 使用 decoder 前 token/codebook reward。
   - 保留 decoder 后 LPIPS/MSE 等指标，但进行 floor calibration。
   - 将 token reward 与 decoded reward 融合。

2. **Codec-Aware Spatial-Temporal Segmental GRPO**
   - 把原始 rollout-level GRPO advantage 改成 segment-level advantage。
   - 单步时退化为 spatial segmental GRPO。
   - 多步时扩展为 spatial-temporal segmental GRPO。
   - 根据 codec floor 调整 advantage normalization。

---

## 1. 需要优先阅读的原项目文件

建议代码 agent 先阅读这些文件，确认接口和张量命名。

```text
vid_wm/README.md
vid_wm/verl/examples/grpo_trainer/run_vgpt.sh
vid_wm/verl/examples/grpo_trainer/run_ctx_msp_vgpt.sh
vid_wm/verl/verl/trainer/main_vgpt_ppo.py
vid_wm/verl/verl/trainer/ppo/ray_trainer.py
vid_wm/verl/verl/trainer/ppo/core_algos.py
vid_wm/verl/verl/workers/reward_manager/naive.py
vid_wm/verl/verl/workers/fsdp_workers.py
vid_wm/verl/ivideogpt/
vid_wm/ivideogpt/
```

已知关键点：

- `run_vgpt.sh` 使用 `python3 -m verl.trainer.main_vgpt_ppo`，并设置 `algorithm.adv_estimator=grpo`。
- `run_ctx_msp_vgpt.sh` 也是 `main_vgpt_ppo`，但设置了 `processor.processor_type=ctx_msp`、`processor.tokens_per_frame=80`、`data.video.segment_length=8` 等多步相关配置。
- `main_vgpt_ppo.py` 构造 `RayVGPTPPOTrainer`。
- `ray_trainer.py` 中会生成 rollout、计算 `token_level_scores`，然后设置 `token_level_rewards`，再调用 `compute_advantage(...)`。
- `core_algos.py` 中已有 `compute_grpo_outcome_advantage(...)`，当前实现是 outcome-level scalar reward，即每条 response 只有一个 score，然后广播到 response token。

---

## 2. 总体实现策略

推荐采用“三层改动”：

```text
Layer A: Reward 计算模块
    新增 codec-calibrated reward，输出 segment_rewards 和必要诊断信息。

Layer B: Advantage 计算模块
    新增 codec_seg_grpo advantage estimator，将 segment_rewards 映射为 token-level advantages。

Layer C: Config + Script
    新增配置项和运行脚本，保留原始 run_vgpt.sh / run_ctx_msp_vgpt.sh。
```

不要一开始大改 actor、rollout 或 vLLM 逻辑。先在 `reward_fn -> token_level_scores -> compute_advantage -> update_actor` 这条链路里实现。

---

## 3. 统一张量约定

### 3.1 基础符号

- `B`：原始 batch size。
- `G`：每个 prompt/context 的 rollout 数，即 `actor_rollout_ref.rollout.n`。
- `BG = B * G`：rollout 后 batch size。
- `H`：预测 horizon。
  - 单步：`H = 1`
  - 多步：`H > 1`
- `K`：每帧空间 segment 数。
  - 例如 `2x2` grid，则 `K = 4`。
  - 例如 `4x4` grid，则 `K = 16`。
- `T`：response token length。
- `tokens_per_frame`：每帧 visual token 数。多步脚本中已有 `processor.tokens_per_frame=80`。

### 3.2 新增 batch 字段

建议在 `DataProto.batch` 中新增以下 tensor：

```python
batch.batch["segment_rewards"]      # shape: [BG, H, K]
batch.batch["segment_floors"]       # shape: [BG, H, K] or [B, H, K] repeated to [BG, H, K]
batch.batch["token_segment_ids"]    # shape: [BG, T], int64, value in [0, H*K-1]
batch.batch["global_scores"]        # shape: [BG], optional, for logging/global advantage
batch.batch["token_level_scores"]   # shape: [BG, T], keep compatible with original trainer
```

注意：

- `segment_rewards` 是我们真正用于 segmental GRPO 的 reward。
- `token_level_scores` 仍然保留，兼容原始 logging、KL reward、metric 统计。可以把 global scalar reward 放到最后一个有效 response token 上，类似原始实现。
- `token_segment_ids` 用来把 `[BG, H, K]` 的 segment advantage 映射回 `[BG, T]` 的 token-level advantage。

---

## 4. 配置项设计

建议在 `vid_wm/verl/verl/trainer/config/vgpt_ppo_trainer.yaml` 或对应 hydra config 中加入：

```yaml
algorithm:
  adv_estimator: codec_seg_grpo  # 新增，可选 grpo / codec_seg_grpo
  codec_seg_grpo:
    lambda_seg: 0.7
    gamma_floor: 0.2
    eps: 1.0e-6
    horizon_discount: 0.9
    use_global_blend: true
    floor_norm: mean       # mean / mad / rank / none
    detach_segment_reward: true

reward:
  method: codec_fused       # original / calibrated_decoded / token / fixed_fusion / codec_fused
  decoded_metrics: [mse, lpips]
  mse_weight: 1.0
  lpips_weight: 1.0
  token_weight: 1.0
  decoded_weight: 1.0
  alpha_reliability: 1.0
  residual_floor_clip: true
  token_reward_type: codebook_l2  # codebook_l2 / exact_match
  broadcast_frame_lpips_to_segments: true

segment:
  mode: grid                # grid / token_chunk / frame_only
  grid_h: 2
  grid_w: 2
  min_patch_size_for_lpips: 32

video:
  train_horizon: 1          # 1 for single-step; >1 for multi-step
  eval_horizons: [1, 3, 5, 8]
```

配置兼容关系：

- `algorithm.adv_estimator=grpo`：完全走原始 GRPO。
- `algorithm.adv_estimator=codec_seg_grpo`：使用我们的 segmental advantage。
- `reward.method=original`：只计算原始 decoded metric reward。
- `reward.method=codec_fused`：计算 fused reward 并额外返回 `segment_rewards`。

---

## 5. Reward 模块设计

建议新增文件：

```text
vid_wm/verl/verl/utils/video_reward/codec_calibrated_reward.py
vid_wm/verl/verl/utils/video_reward/segments.py
vid_wm/verl/verl/utils/video_reward/metrics.py
```

### 5.1 `segments.py`

负责构造 token 到 segment 的映射。

```python
from dataclasses import dataclass
import torch

@dataclass
class SegmentConfig:
    mode: str = "grid"
    grid_h: int = 2
    grid_w: int = 2
    tokens_per_frame: int = 80
    token_grid_h: int | None = None
    token_grid_w: int | None = None
    horizon: int = 1


def infer_token_grid(tokens_per_frame: int) -> tuple[int, int]:
    """
    根据 tokenizer 结构推断 token grid。
    若项目中已有 processor/tokenizer 提供 grid shape，应优先读取。
    对 ctx_msp 的 tokens_per_frame=80，不能盲目假设为正方形，必须查看 tokenizer 输出布局。
    """
    raise NotImplementedError


def build_token_segment_ids(
    response_mask: torch.Tensor,
    response_length: int,
    cfg: SegmentConfig,
    device=None,
) -> torch.Tensor:
    """
    返回 token_segment_ids。

    Args:
        response_mask: [BG, T]
        response_length: T
        cfg: segment 配置

    Returns:
        token_segment_ids: [BG, T], int64
            value = h * K + k
            padding 或非视觉 token 可设为 -1，后续用 response_mask 过滤。
    """
    # 1. 根据 horizon H 和 tokens_per_frame 定位每个 token 属于哪一帧 h。
    # 2. 根据 token 在单帧内的位置，映射到空间 segment k。
    # 3. segment id = h * K + k。
    pass
```

实现重点：

- 单步 `run_vgpt.sh` 的 `data.max_response_length=321` 不一定全是 visual token，需确认 BOS/EOS/action/special token 的布局。
- 多步 `ctx_msp` 使用 `tokens_per_frame=80`，但 80 不是正方形，必须查 tokenizer/processor 中 token 排布。
- 如果短期无法精确做空间 grid，第一版可以使用 `mode=token_chunk`：把每帧的 visual tokens 按顺序均匀分成 K 段。

推荐第一版：

```text
v0: frame_only, K=1
v1: token_chunk, K=4 or 8
v2: grid, K=4 or 16
```

---

### 5.2 `metrics.py`

负责 MSE、LPIPS 等 metric 的局部计算。

```python
import torch
import torch.nn.functional as F


def patch_mse(x_pred: torch.Tensor, x_gt: torch.Tensor, patch_map) -> torch.Tensor:
    """
    Args:
        x_pred: [BG, H, C, Height, Width]
        x_gt:   [BG, H, C, Height, Width]
        patch_map: 描述 K 个空间区域

    Returns:
        mse: [BG, H, K]
    """
    pass


def frame_lpips_broadcast(lpips_model, x_pred, x_gt, K: int) -> torch.Tensor:
    """
    LPIPS 不建议在过小 patch 上算。
    第一版可以整帧算 LPIPS，再 broadcast 到 K 个 segment。

    Returns:
        lpips: [BG, H, K]
    """
    pass


def combine_decoded_distance(mse, lpips, mse_weight=1.0, lpips_weight=1.0):
    return mse_weight * mse + lpips_weight * lpips
```

实现重点：

- MSE 可以 patch-level。
- LPIPS 如果 patch 太小会不稳定，建议先整帧计算再 broadcast。
- 所有 metric 必须统一成“distance 越小越好”。reward 使用负 distance。

---

### 5.3 `codec_calibrated_reward.py`

核心 reward 计算。

```python
import torch

class CodecCalibratedReward:
    def __init__(self, tokenizer, decoder, codebook=None, lpips_model=None, cfg=None):
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.codebook = codebook
        self.lpips_model = lpips_model
        self.cfg = cfg

    @torch.no_grad()
    def compute_gt_tokens(self, gt_frames):
        """
        gt_frames: [B or BG, H, C, Height, Width]
        returns: z_gt, shape depends on tokenizer, eventually [B or BG, H, tokens_per_frame]
        """
        return self.tokenizer.encode(gt_frames)

    @torch.no_grad()
    def decode_tokens(self, z):
        """
        z: [BG, H, tokens_per_frame] or flattened response tokens
        returns decoded frames: [BG, H, C, Height, Width]
        """
        return self.decoder.decode(z)

    @torch.no_grad()
    def compute_codec_floor(self, gt_frames, z_gt, segment_info):
        """
        b_{h,k} = d(D(E(x_{t+h}))_k, x_{t+h,k})

        Returns:
            floor_raw: [B or BG, H, K]
            floor_norm: [B or BG, H, K]
        """
        x_recon_gt = self.decode_tokens(z_gt)
        floor_raw = compute_decoded_distance_by_segment(
            x_recon_gt, gt_frames, segment_info, self.lpips_model, self.cfg
        )
        floor_norm = normalize_floor(floor_raw, mode=self.cfg.floor_norm)
        return floor_raw, floor_norm

    @torch.no_grad()
    def compute_token_reward(self, z_pred, z_gt, segment_info):
        """
        r_tok.
        第一版推荐 codebook embedding L2；exact match 作为 ablation。

        Returns:
            r_tok: [BG, H, K]
        """
        if self.cfg.token_reward_type == "exact_match":
            return token_exact_match_reward(z_pred, z_gt, segment_info)
        elif self.cfg.token_reward_type == "codebook_l2":
            return codebook_l2_reward(z_pred, z_gt, self.codebook, segment_info)
        else:
            raise ValueError(self.cfg.token_reward_type)

    @torch.no_grad()
    def compute_decoded_reward(self, z_pred, gt_frames, floor_raw, segment_info):
        """
        r_dec = -[d(D(z_pred), x_gt) - b]_+

        Returns:
            r_dec: [BG, H, K]
            decoded_dist: [BG, H, K]
        """
        x_pred = self.decode_tokens(z_pred)
        decoded_dist = compute_decoded_distance_by_segment(
            x_pred, gt_frames, segment_info, self.lpips_model, self.cfg
        )
        residual = decoded_dist - floor_raw
        if self.cfg.residual_floor_clip:
            residual = torch.clamp(residual, min=0.0)
        r_dec = -residual
        return r_dec, decoded_dist

    @torch.no_grad()
    def __call__(self, batch, pixels, z_pred, return_dict=True):
        """
        Main interface for trainer.

        Args:
            batch: DataProto after rollout.
            pixels: GT frames, repeated to [BG, H, C, Height, Width] if needed.
            z_pred: predicted visual tokens parsed from batch.responses.

        Returns:
            result dict:
                token_level_scores: [BG, T]
                segment_rewards: [BG, H, K]
                segment_floors: [BG, H, K]
                token_segment_ids: [BG, T]
                metrics: dict
        """
        segment_info = build_segment_info(batch, self.cfg)
        z_gt = self.compute_gt_tokens(pixels)

        floor_raw, floor_norm = self.compute_codec_floor(pixels, z_gt, segment_info)
        r_tok = self.compute_token_reward(z_pred, z_gt, segment_info)
        r_dec, decoded_dist = self.compute_decoded_reward(z_pred, pixels, floor_raw, segment_info)

        q = torch.exp(-self.cfg.alpha_reliability * floor_norm)
        r_fused = (1.0 - q) * r_tok + q * r_dec

        global_score = aggregate_segment_rewards(
            r_fused,
            horizon_discount=self.cfg.horizon_discount,
        )

        token_level_scores = scatter_global_score_to_last_token(
            global_score,
            batch.batch["responses"],
            batch.batch["attention_mask"],
        )

        token_segment_ids = build_token_segment_ids(...)

        metrics = {
            "reward/token_mean": r_tok.mean().item(),
            "reward/decoded_mean": r_dec.mean().item(),
            "reward/fused_mean": r_fused.mean().item(),
            "reward/codec_floor_mean": floor_raw.mean().item(),
            "reward/codec_floor_max": floor_raw.max().item(),
            "reward/reliability_q_mean": q.mean().item(),
            "reward/decoded_dist_mean": decoded_dist.mean().item(),
        }

        return {
            "token_level_scores": token_level_scores,
            "segment_rewards": r_fused,
            "segment_floors": floor_norm,
            "token_segment_ids": token_segment_ids,
            "global_scores": global_score,
            "metrics": metrics,
        }
```

---

## 6. Advantage 模块设计

需要修改：

```text
vid_wm/verl/verl/trainer/ppo/core_algos.py
vid_wm/verl/verl/trainer/ppo/ray_trainer.py
```

### 6.1 新增 advantage estimator enum

在 `ray_trainer.py` 的 `AdvantageEstimator` 中新增：

```python
class AdvantageEstimator(str, Enum):
    GAE = 'gae'
    GRPO = 'grpo'
    CODEC_SEG_GRPO = 'codec_seg_grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'
```

并在 `RayPPOTrainer.__init__` 或 `RayVGPTPPOTrainer.__init__` 的 `use_critic` 判断中加入：

```python
elif self.config.algorithm.adv_estimator in [
    AdvantageEstimator.GRPO,
    AdvantageEstimator.CODEC_SEG_GRPO,
    AdvantageEstimator.REINFORCE_PLUS_PLUS,
    AdvantageEstimator.REMAX,
    AdvantageEstimator.RLOO,
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
]:
    self.use_critic = False
```

---

### 6.2 `core_algos.py` 新增函数

```python
import torch
from collections import defaultdict


def compute_codec_segment_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    segment_rewards: torch.Tensor,
    segment_floors: torch.Tensor,
    token_segment_ids: torch.Tensor,
    lambda_seg: float = 0.7,
    gamma_floor: float = 0.2,
    horizon_discount: float = 0.9,
    epsilon: float = 1e-6,
):
    """
    Codec-Aware Spatial-Temporal Segmental GRPO.

    Args:
        token_level_rewards: [BG, T], kept for compatibility/logging.
        response_mask: [BG, T]
        index: [BG], prompt uid. Same uid means same context with G sampled rollouts.
        segment_rewards: [BG, H, K]
        segment_floors: [BG, H, K]
        token_segment_ids: [BG, T], value in [0, H*K-1], padding may be -1

    Returns:
        advantages: [BG, T]
        returns: [BG, T]
    """
    with torch.no_grad():
        device = segment_rewards.device
        BG, H, K = segment_rewards.shape
        T = response_mask.shape[1]

        # 1. Segment-level group normalization by uid.
        A_seg = torch.zeros_like(segment_rewards)

        uid_to_rows = defaultdict(list)
        for row in range(BG):
            uid_to_rows[index[row]].append(row)

        for uid, rows in uid_to_rows.items():
            rows_t = torch.tensor(rows, device=device, dtype=torch.long)
            r = segment_rewards[rows_t]       # [G, H, K]
            b = segment_floors[rows_t]        # [G, H, K] or repeated floor

            mu = r.mean(dim=0, keepdim=True)  # [1, H, K]
            std = r.std(dim=0, keepdim=True, unbiased=False)

            # codec-aware denominator
            denom = std + gamma_floor * b + epsilon
            A_seg[rows_t] = (r - mu) / denom

        # 2. Global rollout-level advantage, compatible with original GRPO.
        weights = torch.tensor(
            [horizon_discount ** h for h in range(H)],
            device=device,
            dtype=segment_rewards.dtype,
        ).view(1, H, 1)
        global_scores = (segment_rewards * weights).mean(dim=(1, 2))  # [BG]
        A_global = torch.zeros_like(global_scores)

        for uid, rows in uid_to_rows.items():
            rows_t = torch.tensor(rows, device=device, dtype=torch.long)
            s = global_scores[rows_t]
            if len(rows) <= 1:
                A_global[rows_t] = 0.0
            else:
                A_global[rows_t] = (s - s.mean()) / (s.std(unbiased=False) + epsilon)

        # 3. Blend segment and global advantage.
        A_final_seg = lambda_seg * A_seg + (1.0 - lambda_seg) * A_global.view(BG, 1, 1)

        # 4. Gather segment advantage to token-level advantage.
        advantages = torch.zeros_like(token_level_rewards)
        flat_A = A_final_seg.reshape(BG, H * K)

        valid = (token_segment_ids >= 0) & (response_mask > 0)
        safe_ids = torch.clamp(token_segment_ids, min=0)
        gathered = torch.gather(flat_A, dim=1, index=safe_ids)
        advantages[valid] = gathered[valid]
        advantages = advantages * response_mask

        # For GRPO-style actor update, returns can be same as advantages.
        returns = advantages.clone()

    return advantages, returns
```

实现注意：

- 原始 `compute_grpo_outcome_advantage` 当前是 outcome reward：`scores = token_level_rewards.sum(dim=-1)`，然后同 uid 组内标准化。新函数不能只用 sum scalar。
- `index` 使用 `data.non_tensor_batch['uid']`，确保 rollout 后同一 context 的 G 个样本 uid 相同。
- `std(..., unbiased=False)` 比默认 `unbiased=True` 更稳定，尤其 group size 小时。
- `segment_floors` 如果是 `[B,H,K]`，需要在 reward 阶段 repeat 到 `[BG,H,K]`，避免 advantage 函数里形状分支太多。

---

### 6.3 修改 `compute_advantage(...)`

在 `ray_trainer.py` 的 `compute_advantage` 增加分支：

```python
elif adv_estimator == AdvantageEstimator.CODEC_SEG_GRPO:
    cfg = data.meta_info.get("codec_seg_grpo", {})
    advantages, returns = core_algos.compute_codec_segment_grpo_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        segment_rewards=data.batch["segment_rewards"],
        segment_floors=data.batch["segment_floors"],
        token_segment_ids=data.batch["token_segment_ids"],
        lambda_seg=cfg.get("lambda_seg", 0.7),
        gamma_floor=cfg.get("gamma_floor", 0.2),
        horizon_discount=cfg.get("horizon_discount", 0.9),
        epsilon=cfg.get("eps", 1e-6),
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

如果不想把 config 塞进 `data.meta_info`，也可以直接扩展 `compute_advantage` 函数签名，传入 `config.algorithm.codec_seg_grpo`。

---

## 7. Trainer 集成设计

重点修改 `RayVGPTPPOTrainer.fit()` 里视频 reward 计算的位置。

原流程大致是：

```text
batch -> rollout -> compute old_log_probs/ref_log_prob -> reward_fn -> token_level_scores -> token_level_rewards -> compute_advantage -> update_actor
```

需要变为：

```text
batch -> rollout
      -> parse predicted visual tokens
      -> compute codec-calibrated reward
      -> fill token_level_scores + segment_rewards + segment_floors + token_segment_ids
      -> token_level_rewards
      -> compute codec_seg_grpo advantage
      -> update_actor
```

### 7.1 推荐新增 trainer 辅助函数

在 `RayVGPTPPOTrainer` 内部增加：

```python
def _compute_video_codec_reward(self, batch, pixels):
    """
    Adapter between existing RayVGPTPPOTrainer and CodecCalibratedReward.
    """
    responses = batch.batch["responses"]

    # 1. 从 responses 中解析预测 visual tokens。
    z_pred = self._parse_response_to_video_tokens(responses)

    # 2. 确保 pixels 与 rollout 后 batch 对齐。
    # 原项目已有 pixels.repeat_interleave(rollout.n, dim=0) 的逻辑。
    pixels_rep = pixels.repeat_interleave(self.config.actor_rollout_ref.rollout.n, dim=0)

    # 3. 调用 reward 模块。
    result = self.codec_reward_fn(
        batch=batch,
        pixels=pixels_rep,
        z_pred=z_pred,
        return_dict=True,
    )

    # 4. 写回 DataProto。
    batch.batch["token_level_scores"] = result["token_level_scores"]
    batch.batch["segment_rewards"] = result["segment_rewards"]
    batch.batch["segment_floors"] = result["segment_floors"]
    batch.batch["token_segment_ids"] = result["token_segment_ids"]

    if "global_scores" in result:
        batch.batch["global_scores"] = result["global_scores"]

    return batch, result.get("metrics", {})
```

### 7.2 reward_fn 分支逻辑

在 `fit()` 中原本类似：

```python
reward_fn = self.msp_reward_fn if self.config.processor.processor_type == 'ctx_msp' else self.reward_fn
reward_tensor, losses = reward_fn(...)
batch.batch['token_level_scores'] = reward_tensor
```

建议改为：

```python
if self.config.reward.method == "codec_fused":
    batch, reward_metrics = self._compute_video_codec_reward(batch, pixels)
    metrics.update(reward_metrics)
else:
    # keep original reward path
    reward_fn = self.msp_reward_fn if self.config.processor.processor_type == 'ctx_msp' else self.reward_fn
    reward_tensor, losses = reward_fn(
        batch,
        pixels.repeat_interleave(self.config.actor_rollout_ref.rollout.n, dim=0),
        return_reward_tensor=True,
        save_pred=False,
        pixels_before_repeat=pixels,
    )
    batch.batch["token_level_scores"] = reward_tensor
    metrics.update(losses)
```

---

## 8. 解析 response visual tokens

这是实现中最容易出错的地方。

建议新增函数：

```python
def parse_response_to_video_tokens(
    responses: torch.Tensor,
    processor_cfg,
    tokenizer_cfg,
) -> torch.Tensor:
    """
    Convert generated response ids to visual-token tensor.

    Returns:
        z_pred: [BG, H, tokens_per_frame]
    """
    # 单步 simple processor:
    #   需要确认 response 中 special token / EOS / padding 的位置。
    # 多步 ctx_msp processor:
    #   可利用 processor.tokens_per_frame 和 data.video.segment_length。
    pass
```

建议 agent 查找这些信息：

```text
vid_wm/verl/ivideogpt/
vid_wm/ivideogpt/
processor.processor_type=simple
processor.processor_type=ctx_msp
processor.tokens_per_frame
processor.bos_token_id / eos_token_id / pad_token_id
```

临时实现策略：

1. 用 response mask 去掉 padding。
2. 去掉 BOS/EOS/special tokens。
3. 截取或 reshape 成 `[H, tokens_per_frame]`。
4. 如果 token 不够，使用 pad token 或 mask 标记无效 segment。
5. 如果 token 超长，按 `H * tokens_per_frame` 截断，并记录 warning。

---

## 9. 单步和多步统一实现

统一配置：

```python
H = config.video.train_horizon
K = config.segment.grid_h * config.segment.grid_w
```

单步：

```text
H = 1
segment_rewards: [BG, 1, K]
token_segment_ids: value in [0, K-1]
```

多步：

```text
H > 1
segment_rewards: [BG, H, K]
token_segment_ids: value = h * K + k
```

同一 advantage 函数处理两者，不要写两套 loss。

---

## 10. 最小可运行版本 v0

为了尽快跑通，建议先实现 v0：

```text
reward.method = calibrated_decoded
segment.mode = frame_only
K = 1
algorithm.adv_estimator = codec_seg_grpo
lambda_seg = 0.7
```

v0 实际做的是：

- 计算每步 frame-level codec floor。
- decoded reward 使用 `metric(pred, gt) - floor`。
- 每个未来步一个 segment。
- 单步时 `[BG,1,1]`，多步时 `[BG,H,1]`。

这样可以先验证多步维度和 DataProto 接口。

---

## 11. 推荐迭代路线

### v0：Frame-level codec-calibrated reward

- `K=1`
- 不做空间 patch。
- 不做 token reward或 token reward 权重设为 0。
- 目标：跑通 pipeline，验证 calibration 不出错。

### v1：加入 token reward

- `r_fused = (1-q) r_tok + q r_dec`
- token reward 先用 exact match。
- 目标：验证 decoder 前 reward 是否改善 token accuracy。

### v2：加入 codebook embedding reward

- 从 tokenizer/codebook 读取 embedding。
- 用 codebook L2 替代 exact match。
- 目标：让 token reward 更平滑。

### v3：Spatial segment

- `K=4`，先用 `token_chunk`，再尝试 grid。
- MSE patch-level。
- LPIPS frame-level broadcast。
- 目标：验证局部 credit assignment。

### v4：Multi-step training

- `H_train=3`，再到 `H_train=5`。
- 使用同一个 `compute_codec_segment_grpo_advantage`。
- 目标：验证 spatial-temporal segmental GRPO。

---

## 12. 关键测试用例

建议新增：

```text
vid_wm/verl/tests/test_codec_reward_shapes.py
vid_wm/verl/tests/test_codec_seg_grpo_advantage.py
vid_wm/verl/tests/test_token_segment_mapping.py
```

### 12.1 Shape test

```python
def test_codec_seg_grpo_shapes():
    BG, H, K, T = 8, 3, 4, 240
    segment_rewards = torch.randn(BG, H, K)
    segment_floors = torch.rand(BG, H, K)
    token_segment_ids = torch.randint(0, H*K, (BG, T))
    response_mask = torch.ones(BG, T)
    token_level_rewards = torch.zeros(BG, T)
    uid = np.array([0,0,0,0,1,1,1,1])

    adv, ret = compute_codec_segment_grpo_advantage(
        token_level_rewards,
        response_mask,
        uid,
        segment_rewards,
        segment_floors,
        token_segment_ids,
    )
    assert adv.shape == (BG, T)
    assert ret.shape == (BG, T)
    assert torch.isfinite(adv).all()
```

### 12.2 Group normalization test

```python
def test_same_uid_group_mean_near_zero():
    # 当 gamma_floor=0, lambda_seg=1 时，同 uid 组内每个 segment 的 advantage 均值应接近 0。
    pass
```

### 12.3 Degeneration test

```python
def test_single_step_degenerates_to_spatial_grpo():
    # H=1 时，token_segment_ids 只应落在 [0,K-1]。
    pass
```

### 12.4 Original GRPO compatibility

```python
def test_original_grpo_path_unchanged():
    # algorithm.adv_estimator=grpo 时，不要求 segment_rewards 存在。
    pass
```

---

## 13. 需要记录的训练指标

在 wandb/console 中加入：

```text
reward/fused_mean
reward/token_mean
reward/decoded_mean
reward/codec_floor_mean
reward/codec_floor_std
reward/codec_floor_max
reward/reliability_q_mean
reward/reliability_q_min
reward/mse_raw
reward/mse_residual
reward/lpips_raw
reward/lpips_residual
adv/seg_mean
adv/seg_std
adv/global_mean
adv/global_std
adv/final_mean
adv/final_std
adv/floor_gamma
adv/lambda_seg
```

多步时额外记录：

```text
eval/h1_lpips
eval/h3_lpips
eval/h5_lpips
eval/h8_lpips
eval/h1_mse
eval/h3_mse
eval/h5_mse
eval/h8_mse
```

---

## 14. 运行脚本建议

不要改原始脚本，新增两个脚本：

```text
vid_wm/verl/examples/grpo_trainer/run_vgpt_codec_seg_grpo.sh
vid_wm/verl/examples/grpo_trainer/run_ctx_msp_vgpt_codec_seg_grpo.sh
```

### 14.1 单步脚本示例

```bash
set -x
python3 -m verl.trainer.main_vgpt_ppo \
    algorithm.adv_estimator=codec_seg_grpo \
    reward.method=codec_fused \
    reward.decoded_metrics='[mse,lpips]' \
    reward.token_reward_type=codebook_l2 \
    reward.alpha_reliability=1.0 \
    segment.mode=token_chunk \
    segment.grid_h=2 \
    segment.grid_w=2 \
    video.train_horizon=1 \
    algorithm.codec_seg_grpo.lambda_seg=0.7 \
    algorithm.codec_seg_grpo.gamma_floor=0.2 \
    processor.processor_type=simple \
    data.video.dataset_path={path_to_preprocessed_data} \
    processor.tokenizer.path={path_to_pretrained_perframe_tokenizer} \
    actor_rollout_ref.model.path={path_to_pretrained_single_step_pred_transformer} \
    actor_rollout_ref.rollout.n=16 \
    trainer.experiment_name='vgpt_codec_seg_grpo_h1' \
    $@ | tee verl_vgpt_codec_seg_grpo.log
```

### 14.2 多步脚本示例

```bash
set -x
python3 -m verl.trainer.main_vgpt_ppo \
    actor_rollout_ref.actor.interact=True \
    actor_rollout_ref.rollout.interact=True \
    processor.interact=True \
    processor.tokenizer.name=ctx_cnn \
    processor.processor_type=ctx_msp \
    processor.tokens_per_frame=80 \
    data.video.segment_length=8 \
    video.train_horizon=3 \
    algorithm.adv_estimator=codec_seg_grpo \
    reward.method=codec_fused \
    reward.token_reward_type=codebook_l2 \
    segment.mode=token_chunk \
    segment.grid_h=2 \
    segment.grid_w=2 \
    algorithm.codec_seg_grpo.lambda_seg=0.7 \
    algorithm.codec_seg_grpo.gamma_floor=0.2 \
    processor.tokenizer.path={path_to_pretrained_compressive_tokenizer} \
    actor_rollout_ref.model.path={path_to_pretrained_multi_step_pred_transformer} \
    data.video.dataset_path={path_to_preprocessed_data} \
    actor_rollout_ref.rollout.n=16 \
    trainer.experiment_name='ctx_vgpt_codec_seg_grpo_h3' \
    $@ | tee verl_ctx_msp_codec_seg_grpo.log
```

---

## 15. Agent 执行 checklist

### 15.1 必须实现

- [ ] 找到当前视频 reward_fn / `self.reward_fn` / `self.msp_reward_fn` 的定义位置。
- [ ] 确认 response token 到 visual token 的解析方式。
- [ ] 实现 codec floor：`metric(D(E(GT)), GT)`。
- [ ] 实现 calibrated decoded reward：`-[metric(D(pred), GT) - floor]_+`。
- [ ] 实现 token reward：exact match v1，codebook L2 v2。
- [ ] 实现 fused reward：`(1-q) * r_tok + q * r_dec`。
- [ ] 写入 `segment_rewards`、`segment_floors`、`token_segment_ids`。
- [ ] 新增 `codec_seg_grpo` advantage estimator。
- [ ] 保证最终 `advantages` shape 为 `[BG,T]`。
- [ ] 保证 `algorithm.adv_estimator=grpo` 原始路径不变。

### 15.2 强烈建议实现

- [ ] frame-only v0。
- [ ] token_chunk v1。
- [ ] LPIPS frame-level broadcast。
- [ ] 多步 `H=3` 支持。
- [ ] wandb/console metrics。
- [ ] unit tests。

### 15.3 暂时不要做

- [ ] 不要一开始做 object-level mask segment。
- [ ] 不要一开始改 rollout engine。
- [ ] 不要一开始做很长 horizon，例如 H=8 或 H=16。
- [ ] 不要把 LPIPS 直接用于很小 patch。

---

## 16. 最小验收标准

代码 agent 完成后，至少应满足：

1. 原始脚本仍可运行：

```bash
bash examples/grpo_trainer/run_vgpt.sh ...
```

2. 新脚本可在小 batch 上跑通 1-2 step：

```bash
bash examples/grpo_trainer/run_vgpt_codec_seg_grpo.sh \
    data.train_batch_size=2 \
    actor_rollout_ref.rollout.n=2 \
    trainer.total_training_steps=2
```

3. 打印或记录以下 shape：

```text
responses: [BG, T]
segment_rewards: [BG, H, K]
segment_floors: [BG, H, K]
token_segment_ids: [BG, T]
advantages: [BG, T]
```

4. 所有关键 tensor 没有 NaN/Inf：

```python
assert torch.isfinite(segment_rewards).all()
assert torch.isfinite(segment_floors).all()
assert torch.isfinite(advantages).all()
```

5. `codec_floor_mean` 与你观测的 GT reconstruction LPIPS floor 同量级，例如全局 LPIPS 约 0.112；局部或融合后的数值允许不同，但必须可解释。

---

## 17. 可能的坑

1. **response 中不全是 visual token**  
   必须确认 BOS/EOS/PAD/action/special token 的位置。

2. **tokens_per_frame=80 不能直接做 2D grid**  
   如果 tokenizer 不是规则二维网格，先用 `token_chunk`。

3. **LPIPS patch 太小不稳定**  
   第一版 LPIPS 整帧算，然后 broadcast 到 segment。

4. **floor 尺度和 std 尺度不一致**  
   `segment_floors` 必须 normalize 后再进 advantage denominator。

5. **group size 太小导致 std 不稳定**  
   使用 `unbiased=False`，并设置 `eps`。

6. **多步 reward 方差更高**  
   多步从 `H=3` 开始，必要时用 horizon discount。

7. **token reward 与 decoded reward 符号相反**  
   统一约定：reward 越大越好；distance 越小越好，因此 distance reward 要取负。

---

## 18. 预计最终核心改动 diff

```text
新增:
  vid_wm/verl/verl/utils/video_reward/__init__.py
  vid_wm/verl/verl/utils/video_reward/segments.py
  vid_wm/verl/verl/utils/video_reward/metrics.py
  vid_wm/verl/verl/utils/video_reward/codec_calibrated_reward.py
  vid_wm/verl/examples/grpo_trainer/run_vgpt_codec_seg_grpo.sh
  vid_wm/verl/examples/grpo_trainer/run_ctx_msp_vgpt_codec_seg_grpo.sh
  vid_wm/verl/tests/test_codec_reward_shapes.py
  vid_wm/verl/tests/test_codec_seg_grpo_advantage.py

修改:
  vid_wm/verl/verl/trainer/ppo/core_algos.py
  vid_wm/verl/verl/trainer/ppo/ray_trainer.py
  vid_wm/verl/verl/trainer/config/vgpt_ppo_trainer.yaml
```

---

## 19. 最终方法对应公式

Reward：

```text
r_{i,h,k} = (1 - q_{h,k}) r^tok_{i,h,k} + q_{h,k} r^dec_{i,h,k}
q_{h,k} = exp(-alpha * b_tilde_{h,k})
r^dec_{i,h,k} = -[d(D(z_hat_{i,h}), x_{t+h})_k - b_{h,k}]_+
b_{h,k} = d(D(E(x_{t+h}))_k, x_{t+h,k})
```

Advantage：

```text
A^seg_{i,h,k} = (r_{i,h,k} - mu_{h,k}) / (sigma_{h,k} + gamma * b_tilde_{h,k} + eps)
A^global_i = (R_i - mu_R) / (sigma_R + eps)
A_hat_{i,h,k} = lambda * A^seg_{i,h,k} + (1 - lambda) * A^global_i
```

Loss 仍走原来的 actor policy loss，只是传入的 `advantages` 从原始 GRPO 的 rollout-level broadcast advantage 变成我们映射后的 token-level segment advantage。

