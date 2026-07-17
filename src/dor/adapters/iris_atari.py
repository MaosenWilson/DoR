"""Adapter for the public IRIS Atari world-model checkpoints."""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical


TOKENS_PER_FRAME = 16
VOCAB_SIZE = 512


@dataclass(frozen=True)
class IRISWindow:
    episode: int
    step: int
    frames: torch.Tensor  # [C+1,3,64,64], current context plus next-frame GT
    actions: torch.Tensor  # [C], action taking each context frame to the next

    @property
    def context_length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class IRISRollout:
    tokens: torch.Tensor  # [K,16]
    decoded: torch.Tensor  # [K,3,64,64]
    sample_logp: torch.Tensor | None = None  # [K,16], exact cached-generation policy


def _import_upstream(upstream_root: str | Path) -> None:
    root = Path(upstream_root).expanduser().resolve()
    source = root / "src"
    if not (source / "models" / "world_model.py").is_file():
        raise FileNotFoundError(f"not an IRIS checkout: {root}")
    # IRIS targets legacy gym.  Its model modules only need gym for type imports;
    # gymnasium provides the compatible namespace on the current Python runtime.
    import gymnasium as gym

    sys.modules.setdefault("gym", gym)
    source_s = str(source)
    if source_s not in sys.path:
        sys.path.insert(0, source_s)


def _strip_prefix(state: dict, prefix: str) -> OrderedDict:
    marker = prefix + "."
    return OrderedDict((key[len(marker):], value) for key, value in state.items() if key.startswith(marker))


def load_iris(
    upstream_root: str | Path,
    checkpoint: str | Path,
    *,
    action_vocab_size: int | None = None,
    device: torch.device | str = "cuda",
):
    """Instantiate the public IRIS architecture and load tokenizer/world model."""
    _import_upstream(upstream_root)
    from models.tokenizer import Decoder, Encoder, EncoderDecoderConfig, Tokenizer
    from models.transformer import TransformerConfig
    from models.world_model import WorldModel

    codec_config = EncoderDecoderConfig(
        resolution=64,
        in_channels=3,
        z_channels=512,
        ch=64,
        ch_mult=[1, 1, 1, 1, 1],
        num_res_blocks=2,
        attn_resolutions=[8, 16],
        out_ch=3,
        dropout=0.0,
    )
    tokenizer = Tokenizer(
        vocab_size=VOCAB_SIZE,
        embed_dim=512,
        encoder=Encoder(codec_config),
        decoder=Decoder(codec_config),
        with_lpips=False,
    )
    state = torch.load(Path(checkpoint), map_location="cpu")
    inferred_actions = int(state["world_model.embedder.embedding_tables.0.weight"].shape[0])
    if action_vocab_size is None:
        action_vocab_size = inferred_actions
    elif int(action_vocab_size) != inferred_actions:
        raise ValueError(
            f"action vocabulary {action_vocab_size} does not match checkpoint {inferred_actions}"
        )
    world_config = TransformerConfig(
        tokens_per_block=17,
        max_blocks=20,
        attention="causal",
        num_layers=10,
        num_heads=4,
        embed_dim=256,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    world_model = WorldModel(VOCAB_SIZE, int(action_vocab_size), world_config)
    missing, unexpected = tokenizer.load_state_dict(_strip_prefix(state, "tokenizer"), strict=False)
    if missing or any(not key.startswith("lpips.") for key in unexpected):
        raise RuntimeError(f"IRIS tokenizer checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    world_model.load_state_dict(_strip_prefix(state, "world_model"), strict=True)
    return tokenizer.to(device).eval(), world_model.to(device).eval()


def load_iris_actor(
    upstream_root: str | Path,
    checkpoint: str | Path,
    *,
    action_vocab_size: int,
    device: torch.device | str = "cuda",
):
    """Load the checkpoint actor used by the public IRIS evaluation protocol."""
    _import_upstream(upstream_root)
    from models.actor_critic import ActorCritic

    actor = ActorCritic(
        act_vocab_size=int(action_vocab_size),
        use_original_obs=False,
    )
    state = torch.load(Path(checkpoint), map_location="cpu")
    actor.load_state_dict(_strip_prefix(state, "actor_critic"), strict=True)
    return actor.to(device).eval()


@torch.no_grad()
def sample_iris_actor_action(
    tokenizer,
    actor,
    frame: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> tuple[int, float]:
    """Sample one action exactly from the reconstruction-input actor policy."""
    if frame.ndim == 3:
        frame = frame.unsqueeze(0)
    if frame.shape[1:] != (3, 64, 64):
        raise ValueError(f"expected [B,3,64,64] frame, got {tuple(frame.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    reconstructed = tokenizer.encode_decode(
        frame, should_preprocess=True, should_postprocess=True
    ).clamp(0.0, 1.0)
    logits = actor(reconstructed).logits_actions[:, -1] / float(temperature)
    distribution = Categorical(logits=logits)
    action = distribution.sample()
    return int(action.item()), float(distribution.entropy().mean().cpu())


def load_iris_window_npz(path: str | Path, *, device: torch.device | str = "cpu") -> IRISWindow:
    with np.load(path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frames"])
        actions = np.asarray(payload["actions"])
        episode = int(payload["episode"].item())
        step = int(payload["step"].item())
    if frames.ndim != 4 or frames.shape[1:] != (64, 64, 3):
        raise ValueError(f"expected [C+1,64,64,3] frames, got {frames.shape}")
    if actions.shape != (frames.shape[0] - 1,):
        raise ValueError(f"actions {actions.shape} do not match frames {frames.shape}")
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    return IRISWindow(
        episode=episode,
        step=step,
        frames=tensor.to(device),
        actions=torch.from_numpy(actions).long().to(device),
    )


@torch.no_grad()
def encode_frames(tokenizer, frames: torch.Tensor) -> torch.Tensor:
    tokens = tokenizer.encode(frames, should_preprocess=True).tokens
    if tokens.shape != (frames.shape[0], TOKENS_PER_FRAME):
        raise RuntimeError(f"unexpected IRIS token shape {tuple(tokens.shape)}")
    return tokens


def interleave_context_tokens(frame_tokens: torch.Tensor, history_actions: torch.Tensor) -> torch.Tensor:
    """Build ``obs(16), action, ..., current_obs(16)`` prompt tokens."""
    if frame_tokens.ndim != 2 or frame_tokens.shape[1] != TOKENS_PER_FRAME:
        raise ValueError("frame_tokens must be [C,16]")
    if history_actions.shape != (frame_tokens.shape[0] - 1,):
        raise ValueError("history_actions must contain C-1 actions")
    pieces = []
    for index, tokens in enumerate(frame_tokens):
        pieces.append(tokens)
        if index < len(history_actions):
            pieces.append(history_actions[index:index + 1])
    return torch.cat(pieces).unsqueeze(0)


def teacher_forced_next_frame_inputs(
    context_tokens: torch.Tensor,
    actions: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> torch.Tensor:
    """Build inputs whose final 16 observation logits score one candidate frame.

    IRIS lays out each block as 16 observation tokens followed by the action that
    leads to the next block.  The current action predicts candidate token zero;
    candidate tokens 0--14 predict tokens 1--15.
    """
    if context_tokens.ndim != 2 or context_tokens.shape[1] != TOKENS_PER_FRAME:
        raise ValueError("context_tokens must be [C,16]")
    if actions.shape != (context_tokens.shape[0],):
        raise ValueError("actions must contain one transition action per context frame")
    if candidate_tokens.ndim != 2 or candidate_tokens.shape[1] != TOKENS_PER_FRAME:
        raise ValueError("candidate_tokens must be [K,16]")
    prompt = interleave_context_tokens(context_tokens, actions[:-1])
    prefix = torch.cat((prompt, actions[-1:].reshape(1, 1)), dim=1)
    prefix = prefix.expand(candidate_tokens.shape[0], -1)
    return torch.cat((prefix, candidate_tokens[:, :-1]), dim=1)


@torch.no_grad()
def sample_next_frame(
    tokenizer,
    world_model,
    window: IRISWindow,
    *,
    group_size: int,
    seed: int,
    temperature: float = 1.0,
) -> tuple[IRISRollout, torch.Tensor]:
    """Sample a candidate group and return ground-truth next-frame tokens."""
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    torch.manual_seed(int(seed))
    frame_tokens = encode_frames(tokenizer, window.frames)
    prompt = interleave_context_tokens(frame_tokens[:-1], window.actions[:-1])
    prompt = prompt.expand(group_size, -1).contiguous()
    cache = world_model.transformer.generate_empty_keys_values(
        n=group_size, max_tokens=world_model.config.max_tokens
    )
    world_model(prompt, past_keys_values=cache)
    token = window.actions[-1].expand(group_size, 1)
    sampled, sampled_logp = [], []
    for _ in range(TOKENS_PER_FRAME):
        output = world_model(token, past_keys_values=cache)
        distribution = Categorical(
            logits=output.logits_observations / float(temperature)
        )
        token = distribution.sample()
        sampled.append(token)
        sampled_logp.append(distribution.log_prob(token))
    candidate_tokens = torch.cat(sampled, dim=1)
    candidate_logp = torch.cat(sampled_logp, dim=1)
    embeddings = tokenizer.embedding(candidate_tokens)
    latent = rearrange(embeddings, "b (h w) e -> b e h w", h=4, w=4)
    decoded = tokenizer.decode(latent, should_postprocess=True).clamp(0.0, 1.0)
    return IRISRollout(candidate_tokens, decoded, candidate_logp), frame_tokens[-1]


def teacher_forced_next_frame_logp(
    tokenizer,
    world_model,
    window: IRISWindow,
    rollout: IRISRollout,
) -> torch.Tensor:
    """Log-probability of sampled next-frame tokens under the current policy.

    The caller controls ``world_model.train/eval``.  External GRPO keeps the
    model in eval mode for both sampling and scoring so IRIS's 0.1 dropout does
    not turn the update off-policy; eval mode does not disable gradients.
    """
    # Sampling runs under inference_mode.  Materialize ordinary index tensors
    # before embedding/gather so autograd may save them for the policy backward.
    candidate_tokens = rollout.tokens.detach().clone()
    with torch.no_grad():
        context_tokens = encode_frames(tokenizer, window.frames[:-1])
    inputs = teacher_forced_next_frame_inputs(
        context_tokens,
        window.actions,
        candidate_tokens,
    )
    output = world_model(inputs)
    logits = output.logits_observations
    if logits.shape[:2] != (rollout.tokens.shape[0], TOKENS_PER_FRAME):
        # A full prompt also produces logits for historical blocks.  The final
        # 16 entries are exactly current-action plus candidate tokens 0--14.
        if logits.shape[0] != rollout.tokens.shape[0] or logits.shape[1] < TOKENS_PER_FRAME:
            raise RuntimeError(f"unexpected IRIS observation logits {tuple(logits.shape)}")
        logits = logits[:, -TOKENS_PER_FRAME:]
    return F.log_softmax(logits.float(), dim=-1).gather(
        -1, candidate_tokens.unsqueeze(-1)
    ).squeeze(-1)


@torch.no_grad()
def reachable_target(tokenizer, target_tokens: torch.Tensor) -> torch.Tensor:
    embeddings = tokenizer.embedding(target_tokens.unsqueeze(0))
    latent = rearrange(embeddings, "b (h w) e -> b e h w", h=4, w=4)
    return tokenizer.decode(latent, should_postprocess=True).clamp(0.0, 1.0)[0]


@torch.no_grad()
def post_quant_latent_reward(tokenizer, candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate_z = rearrange(tokenizer.embedding(candidate), "b (h w) e -> b e h w", h=4, w=4)
    target_z = rearrange(tokenizer.embedding(target.unsqueeze(0)), "b (h w) e -> b e h w", h=4, w=4)
    candidate_z = tokenizer.post_quant_conv(candidate_z)
    target_z = tokenizer.post_quant_conv(target_z)
    return -(candidate_z.float() - target_z.float()).square().mean(dim=(1, 2, 3)).sqrt()
