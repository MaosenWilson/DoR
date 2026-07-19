"""VP2-RoboSuite adapter for the public action-conditioned iVideoGPT checkpoint.

The upstream checkpoint uses two 64x64 context frames. Its token stream contains
``2 * 257 - 1`` context tokens followed by one ``[SDF] + 16 dynamics`` block per
future frame. The generation prompt includes the first future ``[SDF]`` token; this
is intentional and matches the upstream inference script exactly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


CONTEXT_LENGTH = 2
CONTEXT_GRID_TOKENS = 16 * 16
DYNAMICS_GRID_TOKENS = 4 * 4
CONTEXT_BLOCK_TOKENS = 1 + CONTEXT_GRID_TOKENS
FUTURE_BLOCK_TOKENS = 1 + DYNAMICS_GRID_TOKENS


@dataclass(frozen=True)
class VP2Window:
    """One action-aligned RoboSuite context and future segment."""

    episode: str
    start: int
    frames: torch.Tensor  # [C + H, 3, 64, 64], float in [0, 1]
    actions: torch.Tensor  # [C + H, action_dim], float32; final row is unused

    @property
    def horizon(self) -> int:
        return int(self.frames.shape[0] - CONTEXT_LENGTH)


@dataclass(frozen=True)
class VP2Rollout:
    """Sampled iVideoGPT rollouts with their decoded frames."""

    full_tokens: torch.Tensor  # [K, C*257 - 1 + H*17]
    dynamics_tokens: torch.Tensor  # [K, H, 16], original vocabulary IDs
    decoded: torch.Tensor  # [K, C + H, 3, 64, 64]


def _import_upstream(upstream_root: str | Path) -> None:
    root = Path(upstream_root).expanduser().resolve()
    marker = root / "ivideogpt" / "transformer" / "action_model.py"
    if not marker.is_file():
        raise FileNotFoundError(f"not an iVideoGPT checkout: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def list_vp2_episodes(hdf5_path: str | Path) -> list[str]:
    """List VP2 episode IDs in numeric order without loading image data."""
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        episodes = list(handle["data"].keys())
    return sorted(episodes, key=lambda value: int(value.rsplit("_", 1)[1]))


def load_vp2_window(
    hdf5_path: str | Path,
    episode: str,
    start: int,
    horizon: int,
    *,
    image_key: str = "agentview_shift_2_image",
    action_dim: int | None = 4,
    resolution: int = 64,
    device: torch.device | str = "cpu",
) -> VP2Window:
    """Load one contiguous, action-aligned VP2 RoboSuite/RoboDesk segment.

    iVideoGPT predicts future frame ``t`` with action index ``t-1`` when the
    context length is two. Returning frames and actions with the same start offset
    preserves that upstream convention: context ``[s, s+1]`` and future frame
    ``s+2`` use ``actions[s+1]``.

    ``image_key`` selects the observation stream (``agentview_shift_2_image`` for
    RoboSuite PushCenter, ``camera_image`` for RoboDesk). ``action_dim`` validates
    the action width (4 for RoboSuite, 5 for RoboDesk); pass ``None`` to accept
    whatever the file stores. RoboDesk's ``camera_image`` is written under a
    compression filter that fails on sliced reads, so the frame stream is read in
    full and then sliced -- matching the upstream ``preprocess_vp2`` loader.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    import h5py

    length = CONTEXT_LENGTH + int(horizon)
    with h5py.File(hdf5_path, "r", libver="latest") as handle:
        group_path = f"data/{episode}"
        if group_path not in handle:
            raise KeyError(f"episode {episode!r} not found")
        group = handle[group_path]
        image_path = f"obs/{image_key}"
        if image_path not in group:
            raise KeyError(f"{episode!r} has no image key {image_key!r}")
        total = int(group["actions"].shape[0])
        if start < 0 or start + length > total:
            raise ValueError(
                f"window [{start}, {start + length}) exceeds {episode!r} length {total}"
            )
        # Full read then slice: compressed camera_image can reject partial reads.
        images = np.asarray(group[image_path][()])[start:start + length]
        actions = np.asarray(group["actions"][start:start + length])

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB images, got {images.shape}")
    if actions.ndim != 2 or (action_dim is not None and actions.shape[1] != action_dim):
        raise ValueError(
            f"expected {action_dim}-dimensional actions, got {actions.shape}"
        )

    frames = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(frames.shape[-2:]) != (resolution, resolution):
        frames = F.interpolate(
            frames,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return VP2Window(
        episode=episode,
        start=int(start),
        frames=frames.to(device),
        actions=torch.from_numpy(actions).float().to(device),
    )


def load_vp2_window_npz(
    path: str | Path,
    *,
    action_dim: int | None = 4,
    resolution: int = 64,
    device: torch.device | str = "cpu",
) -> VP2Window:
    """Load a frozen window exported by ``export_vp2_window.py``.

    The released VP2 file needs HDF5 1.12.x for reliable scale-offset decoding on
    the current server. Exporting windows in that isolated reader makes the actual
    model/reward environment independent of an HDF5 ABI detail. ``action_dim``
    validates the action width (4 RoboSuite, 5 RoboDesk); ``None`` accepts any.
    """
    with np.load(path, allow_pickle=False) as payload:
        images = np.asarray(payload["image"])
        actions = np.asarray(payload["action"])
        episode = str(payload["episode"].item())
        start = int(payload["start"].item())
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB images, got {images.shape}")
    if actions.ndim != 2 or actions.shape[0] != images.shape[0] or (
        action_dim is not None and actions.shape[1] != action_dim
    ):
        raise ValueError(f"actions {actions.shape} do not match images {images.shape}")
    frames = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(frames.shape[-2:]) != (resolution, resolution):
        frames = F.interpolate(
            frames,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return VP2Window(
        episode=episode,
        start=start,
        frames=frames.to(device),
        actions=torch.from_numpy(actions).float().to(device),
    )


def load_ivideogpt(
    upstream_root: str | Path,
    checkpoint_dir: str | Path,
    *,
    horizon: int,
    action_dim: int = 4,
    device: torch.device | str = "cuda",
):
    """Load a public two-context-frame action-conditioned iVideoGPT checkpoint."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    _import_upstream(upstream_root)
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    from ivideogpt.transformer import HeadModelWithAction
    from ivideogpt.vq_model import CompressiveVQModel

    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    tokenizer = CompressiveVQModel.from_pretrained(
        checkpoint, subfolder="tokenizer", low_cpu_mem_usage=False
    ).to(device).eval()
    if tokenizer.context_length != CONTEXT_LENGTH:
        raise ValueError(f"expected context_length={CONTEXT_LENGTH}, got {tokenizer.context_length}")
    # The released VP2 tokenizer uses separate 8192-entry VQ and dynamics
    # codebooks.  The layout below depends on token *blocks*, not a presumed
    # codebook size; the transformer-vocabulary check validates compatibility.
    if tokenizer.num_vq_embeddings <= 0 or tokenizer.num_dyn_embeddings <= 0:
        raise ValueError("VP2 tokenizer must expose non-empty VQ and dynamics codebooks")

    config = AutoConfig.from_pretrained(checkpoint, subfolder="transformer")
    llm = AutoModelForCausalLM.from_config(config)
    model = HeadModelWithAction(
        llm,
        action_dim=int(action_dim),
        prelude_tokens_num=CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1,
        tokens_num_per_dyna=DYNAMICS_GRID_TOKENS,
        context=CONTEXT_LENGTH,
        segment_length=CONTEXT_LENGTH + int(horizon),
    ).to(device)
    state_path = checkpoint / "transformer" / "model.safetensors"
    model.load_state_dict(load_file(str(state_path)), strict=True)
    if model.llm.config.vocab_size != tokenizer.num_vq_embeddings + tokenizer.num_dyn_embeddings + 2:
        raise ValueError("transformer/tokenizer vocabularies are incompatible")
    return tokenizer, model.eval()


@torch.no_grad()
def tokenize_ground_truth(tokenizer, window: VP2Window) -> torch.Tensor:
    """Encode all context and future frames using the frozen upstream tokenizer."""
    tokens, _ = tokenizer.tokenize(window.frames.unsqueeze(0), CONTEXT_LENGTH)
    expected = CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1 + window.horizon * FUTURE_BLOCK_TOKENS
    if tokens.shape != (1, expected):
        raise RuntimeError(f"unexpected VP2 token shape {tuple(tokens.shape)}, expected (1, {expected})")
    return tokens


def prompt_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Keep context plus the first future SDF token, exactly as upstream inference."""
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"expected [1, T] tokens, got {tuple(tokens.shape)}")
    prompt_length = CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS
    if tokens.shape[1] < prompt_length:
        raise ValueError("token sequence does not contain the first future SDF token")
    return tokens[:, :prompt_length]


def _future_block_start() -> int:
    return CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1


def future_dynamics_tokens(full_tokens: torch.Tensor, horizon: int) -> torch.Tensor:
    """Extract the 16 sampled dynamics tokens from each `[SDF] + 16` future block."""
    if full_tokens.ndim != 2:
        raise ValueError(f"expected [K, T] tokens, got {tuple(full_tokens.shape)}")
    begin = _future_block_start()
    expected = begin + int(horizon) * FUTURE_BLOCK_TOKENS
    if full_tokens.shape[1] != expected:
        raise ValueError(f"unexpected full token length {full_tokens.shape[1]}, expected {expected}")
    blocks = full_tokens[:, begin:].reshape(full_tokens.shape[0], horizon, FUTURE_BLOCK_TOKENS)
    return blocks[:, :, 1:]


@torch.no_grad()
def future_dynamics_latents(
    tokenizer,
    dynamics_tokens: torch.Tensor,
    *,
    projected: bool = True,
) -> torch.Tensor:
    """Map global dynamics-token IDs to the continuous pre-decoder latents.

    iVideoGPT assigns categorical IDs to a learned dynamics codebook.  Numeric
    token differences and Hamming distance therefore do not describe the
    geometry seen by the video decoder.  ``projected=True`` follows the public
    ``detokenize`` path through ``post_quant_linear`` and returns the exact
    latent vectors consumed before de-patchification and conditional decoding.
    """
    if dynamics_tokens.ndim < 2 or dynamics_tokens.shape[-1] != DYNAMICS_GRID_TOKENS:
        raise ValueError(
            f"expected dynamics tokens ending in {DYNAMICS_GRID_TOKENS}, "
            f"got {tuple(dynamics_tokens.shape)}"
        )
    local = dynamics_tokens.long() - int(tokenizer.num_vq_embeddings)
    if torch.any(local < 0) or torch.any(local >= int(tokenizer.num_dyn_embeddings)):
        raise ValueError("dynamics token IDs fall outside the tokenizer dynamics codebook")
    latent = tokenizer.dynamics_quantize.embedding(local)
    if projected:
        latent = tokenizer.post_quant_linear(latent)
    return latent


@torch.no_grad()
def future_dynamics_latent_reward(
    tokenizer,
    candidate_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    projected: bool = True,
) -> torch.Tensor:
    """Return negative latent RMS per candidate and future frame, ``[K, H]``."""
    if candidate_tokens.ndim != 3:
        raise ValueError(f"candidate tokens must be [K,H,16], got {tuple(candidate_tokens.shape)}")
    if target_tokens.ndim == 2:
        target_tokens = target_tokens.unsqueeze(0)
    if target_tokens.shape != (1, *candidate_tokens.shape[1:]):
        raise ValueError(
            f"target tokens must be [H,16] or [1,H,16], got {tuple(target_tokens.shape)}"
        )
    candidate = future_dynamics_latents(tokenizer, candidate_tokens, projected=projected)
    target = future_dynamics_latents(tokenizer, target_tokens, projected=projected)
    return -(candidate.float() - target.float()).square().mean(dim=(-2, -1)).sqrt()


@torch.no_grad()
def sample_rollout(
    tokenizer,
    model,
    ground_truth_tokens: torch.Tensor,
    actions: torch.Tensor,
    *,
    horizon: int,
    group_size: int,
    seed: int,
    temperature: float = 1.0,
    top_k: int = 100,
) -> VP2Rollout:
    """Sample one candidate group with the public iVideoGPT generation semantics."""
    if actions.ndim != 2 or actions.shape[0] < CONTEXT_LENGTH + horizon:
        raise ValueError("actions must include context and all future action slots")
    if group_size < 2:
        raise ValueError("GRPO requires at least two candidates")
    torch.manual_seed(int(seed))
    prompt = prompt_from_tokens(ground_truth_tokens)
    sampled = model.generate(
        prompt.expand(group_size, -1).contiguous(),
        do_sample=True,
        temperature=float(temperature),
        top_k=int(top_k),
        max_new_tokens=FUTURE_BLOCK_TOKENS * int(horizon) - 1,
        pad_token_id=50256,
        action=actions.unsqueeze(0).expand(group_size, -1, -1).contiguous(),
    )
    expected = _future_block_start() + int(horizon) * FUTURE_BLOCK_TOKENS
    if sampled.shape != (group_size, expected):
        raise RuntimeError(f"unexpected sampled token shape {tuple(sampled.shape)}, expected {(group_size, expected)}")
    decoded = tokenizer.detokenize(sampled, CONTEXT_LENGTH).clamp(0.0, 1.0)
    return VP2Rollout(
        full_tokens=sampled,
        dynamics_tokens=future_dynamics_tokens(sampled, horizon),
        decoded=decoded,
    )


def prefix_tokens_through_frame(full_tokens: torch.Tensor, prefix_frames: int) -> torch.Tensor:
    """Return a generated prefix ending with the next frame's SDF separator.

    ``prefix_frames`` complete future blocks are retained.  The returned token
    stream also contains the following SDF marker, which is the exact position
    from which the public action-conditioned generator predicts the next block.
    """
    if full_tokens.ndim != 2:
        raise ValueError(f"expected [batch,tokens], got {tuple(full_tokens.shape)}")
    horizon = (full_tokens.shape[1] - _future_block_start()) // FUTURE_BLOCK_TOKENS
    if full_tokens.shape[1] != _future_block_start() + horizon * FUTURE_BLOCK_TOKENS:
        raise ValueError("full token sequence is not composed of complete future blocks")
    if prefix_frames < 1 or prefix_frames >= horizon:
        raise ValueError("prefix_frames must retain at least one but not all future frames")
    stop = _future_block_start() + prefix_frames * FUTURE_BLOCK_TOKENS + 1
    return full_tokens[:, :stop].clone()


@torch.no_grad()
def sample_continuations_from_prefixes(
    tokenizer,
    model,
    prefix_tokens: torch.Tensor,
    actions: torch.Tensor,
    *,
    prefix_frames: int,
    horizon: int,
    continuations: int,
    seed: int,
    temperature: float = 1.0,
    top_k: int = 100,
) -> VP2Rollout:
    """Branch independent continuations from fixed generated frame prefixes.

    This reproduces ``HeadModelWithAction.generate`` frame by frame.  Rebuilding
    embeddings for a prefix requires restoring action additions at every existing
    SDF separator; omitting those additions would silently turn the branch audit
    into a differently conditioned model.
    """
    if prefix_tokens.ndim != 2:
        raise ValueError(f"expected [prefix,tokens], got {tuple(prefix_tokens.shape)}")
    if prefix_frames < 1 or prefix_frames >= horizon:
        raise ValueError("prefix_frames must lie in [1,horizon-1]")
    if continuations < 2:
        raise ValueError("at least two continuations are required")
    if actions.ndim != 2 or actions.shape[0] < CONTEXT_LENGTH + horizon:
        raise ValueError("actions must include context and all future action slots")
    expected_prefix = _future_block_start() + prefix_frames * FUTURE_BLOCK_TOKENS + 1
    if prefix_tokens.shape[1] != expected_prefix:
        raise ValueError(
            f"prefix length {prefix_tokens.shape[1]} does not match "
            f"prefix_frames={prefix_frames} (expected {expected_prefix})"
        )
    if not hasattr(model, "llm") or not hasattr(model, "action_linear"):
        raise TypeError("model does not expose the public iVideoGPT action interface")

    torch.manual_seed(int(seed))
    tokens = prefix_tokens.repeat_interleave(int(continuations), dim=0).clone()
    batch = tokens.shape[0]
    action = actions.unsqueeze(0).expand(batch, -1, -1).contiguous()
    action_embeds = model.action_linear(action)
    inputs_embeds = model.get_input_embeddings(tokens)

    # Restore action conditioning for all separators already represented in the
    # fixed prefix, including the separator for the first resampled frame.
    for frame in range(prefix_frames + 1):
        position = model.prelude_tokens_num + frame * (model.tokens_num_per_dyna + 1)
        inputs_embeds[:, position, :] += action_embeds[:, frame + model.context - 1, :]

    sdf = int(model.token_for_sdf)
    for frame in range(prefix_frames, horizon):
        predicted = model.llm.generate(
            inputs_embeds=inputs_embeds,
            do_sample=True,
            temperature=float(temperature),
            pad_token_id=50256,
            top_k=int(top_k),
            use_cache=True,
            max_new_tokens=DYNAMICS_GRID_TOKENS,
            return_dict_in_generate=False,
        )
        separator = torch.full(
            (batch, 1), sdf, device=tokens.device, dtype=predicted.dtype
        )
        tokens = torch.cat([tokens, predicted, separator], dim=1)
        new_embeds = model.get_input_embeddings(torch.cat([predicted, separator], dim=1))
        inputs_embeds = torch.cat([inputs_embeds, new_embeds], dim=1)
        next_frame = frame + 1
        if next_frame < horizon:
            inputs_embeds[:, -1, :] += action_embeds[
                :, next_frame + model.context - 1, :
            ]

    tokens = tokens[:, :-1]
    expected = _future_block_start() + int(horizon) * FUTURE_BLOCK_TOKENS
    if tokens.shape != (batch, expected):
        raise RuntimeError(
            f"unexpected branched token shape {tuple(tokens.shape)}, expected {(batch, expected)}"
        )
    decoded = tokenizer.detokenize(tokens, CONTEXT_LENGTH).clamp(0.0, 1.0)
    return VP2Rollout(
        full_tokens=tokens,
        dynamics_tokens=future_dynamics_tokens(tokens, horizon),
        decoded=decoded,
    )


def teacher_forced_dynamics_logp(model, rollout: VP2Rollout, actions: torch.Tensor) -> torch.Tensor:
    """Return sampled dynamics-token log-probabilities `[K, H, 16]` with gradients.

    SDF markers are deterministic framing tokens and are deliberately excluded. This
    is the exact policy quantity to which frame-block GRPO advantages are attached.
    """
    # Rollouts are normally sampled under inference_mode to avoid storing a
    # generation graph.  A tensor created there cannot be saved by the embedding
    # backward kernel, so materialise an ordinary discrete-token tensor before
    # the teacher-forced policy forward.  This leaves sampled IDs unchanged.
    full_tokens = rollout.full_tokens.clone()
    horizon = int(rollout.dynamics_tokens.shape[1])
    group_size = int(full_tokens.shape[0])
    if actions.ndim != 2 or actions.shape[0] < CONTEXT_LENGTH + horizon:
        raise ValueError("actions are shorter than the rollout horizon")
    output = model(
        input_ids=full_tokens,
        action=actions.unsqueeze(0).expand(group_size, -1, -1).contiguous(),
    )
    logits = output.logits.float()
    begin = _future_block_start()
    positions = []
    for frame in range(horizon):
        first_dynamic = begin + frame * FUTURE_BLOCK_TOKENS + 1
        positions.extend(range(first_dynamic, first_dynamic + DYNAMICS_GRID_TOKENS))
    pos = torch.tensor(positions, device=full_tokens.device, dtype=torch.long)
    predicted = logits[:, pos - 1, :]
    target = full_tokens[:, pos]
    logp = torch.log_softmax(predicted, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return logp.reshape(group_size, horizon, DYNAMICS_GRID_TOKENS)


@torch.no_grad()
def decoded_ground_truth(tokenizer, ground_truth_tokens: torch.Tensor) -> torch.Tensor:
    """Return the codec-reachable target `D(E(s'))` for every context/future frame."""
    return tokenizer.detokenize(ground_truth_tokens, CONTEXT_LENGTH).clamp(0.0, 1.0)


def frame_rewards(metrics, rollout: VP2Rollout, window: VP2Window, reachable: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute raw and RC reward matrices `[K, H]` against a shared candidate group."""
    horizon = window.horizon
    if reachable.shape != (1, CONTEXT_LENGTH + horizon, *window.frames.shape[1:]):
        raise ValueError("reachable target shape does not match the window")
    predicted = rollout.decoded[:, CONTEXT_LENGTH:]
    raw_target = window.frames[CONTEXT_LENGTH:]
    rc_target = reachable[0, CONTEXT_LENGTH:]
    raw, rc = [], []
    for frame in range(horizon):
        raw_metric = metrics.eval_batch(predicted[:, frame], raw_target[frame])
        rc_metric = metrics.eval_batch(predicted[:, frame], rc_target[frame])
        raw.append(-(np.asarray(raw_metric["mse"]) + np.asarray(raw_metric["lpips"])))
        rc.append(-(np.asarray(rc_metric["mse"]) + np.asarray(rc_metric["lpips"])))
    return {"raw": np.stack(raw, axis=1), "rc": np.stack(rc, axis=1)}


def windows_from_manifest(
    entries: Iterable[dict],
    hdf5_path: str | Path,
    horizon: int,
    *,
    image_key: str = "agentview_shift_2_image",
    action_dim: int | None = 4,
    device="cpu",
) -> list[VP2Window]:
    """Load a frozen manifest of `{"episode": ..., "start": ...}` entries."""
    return [
        load_vp2_window(
            hdf5_path, str(entry["episode"]), int(entry["start"]), horizon,
            image_key=image_key, action_dim=action_dim, device=device,
        )
        for entry in entries
    ]
