import torch
from torch import nn

from dor.adapters.ivideogpt_vp2 import (
    CONTEXT_BLOCK_TOKENS,
    CONTEXT_LENGTH,
    FUTURE_BLOCK_TOKENS,
    future_dynamics_tokens,
    future_dynamics_latent_reward,
    prefix_tokens_through_frame,
    prompt_from_tokens,
)


class _ToyTokenizer:
    num_vq_embeddings = 10
    num_dyn_embeddings = 4

    def __init__(self):
        self.dynamics_quantize = type("Quantizer", (), {})()
        self.dynamics_quantize.embedding = nn.Embedding.from_pretrained(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        )
        self.post_quant_linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.post_quant_linear.weight.copy_(torch.diag(torch.tensor([2.0, 0.5])))


def test_prompt_keeps_first_future_separator():
    horizon = 3
    full_length = CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1 + horizon * FUTURE_BLOCK_TOKENS
    tokens = torch.arange(full_length).reshape(1, -1)

    prompt = prompt_from_tokens(tokens)

    assert prompt.shape == (1, CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS)
    assert prompt[0, -1].item() == CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1


def test_future_dynamics_excludes_only_frame_separators():
    horizon = 2
    start = CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1
    full_length = start + horizon * FUTURE_BLOCK_TOKENS
    tokens = torch.arange(full_length).reshape(1, -1)

    dynamics = future_dynamics_tokens(tokens, horizon)

    assert dynamics.shape == (1, horizon, FUTURE_BLOCK_TOKENS - 1)
    assert dynamics[0, 0].tolist() == list(range(start + 1, start + FUTURE_BLOCK_TOKENS))
    second = start + FUTURE_BLOCK_TOKENS
    assert dynamics[0, 1].tolist() == list(range(second + 1, second + FUTURE_BLOCK_TOKENS))


def test_latent_reward_uses_codebook_geometry_not_token_hamming():
    tokenizer = _ToyTokenizer()
    target = torch.full((1, 16), 10, dtype=torch.long)
    near = target.clone()
    far = target.clone()
    near[0, 0] = 11
    far[0, 0] = 13
    candidates = torch.stack([near, far], dim=0)

    reward = future_dynamics_latent_reward(tokenizer, candidates, target)

    assert reward.shape == (2, 1)
    assert reward[0, 0] > reward[1, 0]


def test_prefix_tokens_end_at_next_frame_separator():
    horizon = 4
    start = CONTEXT_LENGTH * CONTEXT_BLOCK_TOKENS - 1
    full_length = start + horizon * FUTURE_BLOCK_TOKENS
    tokens = torch.arange(full_length).reshape(1, -1)

    prefix = prefix_tokens_through_frame(tokens, prefix_frames=2)

    assert prefix.shape[1] == start + 2 * FUTURE_BLOCK_TOKENS + 1
    assert prefix[0, -1].item() == tokens[0, prefix.shape[1] - 1].item()
