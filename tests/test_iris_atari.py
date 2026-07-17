from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from dor.adapters.iris_atari import (
    interleave_context_tokens,
    post_quant_latent_reward,
    sample_iris_actor_action,
    teacher_forced_next_frame_inputs,
)


def test_cluster_bootstrap_confidence_is_explicit():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "external" / "gate_iris_breakout_rank.py"
    spec = importlib.util.spec_from_file_location("gate_iris_breakout_rank", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module._cluster_bootstrap(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0, 0, 1, 1]),
        rounds=100,
        seed=7,
        confidence=0.95,
    )

    assert report["confidence"] == 0.95
    assert "ci" in report
    assert "ci90" not in report


def test_interleave_context_uses_history_actions_but_not_target_action():
    frames = torch.arange(3 * 16).reshape(3, 16)
    actions = torch.tensor([7, 8])
    prompt = interleave_context_tokens(frames, actions)
    assert prompt.shape == (1, 50)
    assert prompt[0, 16].item() == 7
    assert prompt[0, 33].item() == 8
    assert prompt[0, -16:].tolist() == frames[-1].tolist()


class _Tokenizer:
    def __init__(self):
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
        )
        self.post_quant_conv = nn.Conv2d(2, 2, 1, bias=False)
        with torch.no_grad():
            self.post_quant_conv.weight.zero_()
            self.post_quant_conv.weight[0, 0, 0, 0] = 1.0
            self.post_quant_conv.weight[1, 1, 0, 0] = 1.0


def test_post_quant_reward_respects_codebook_geometry():
    tokenizer = _Tokenizer()
    target = torch.zeros(16, dtype=torch.long)
    near, far = target.clone(), target.clone()
    near[0], far[0] = 1, 2
    reward = post_quant_latent_reward(tokenizer, torch.stack([near, far]), target)
    assert reward[0] > reward[1]


def test_teacher_forced_inputs_follow_iris_block_layout():
    context = torch.arange(4 * 16).reshape(4, 16)
    actions = torch.tensor([70, 71, 72, 73])
    candidates = torch.stack((torch.arange(100, 116), torch.arange(200, 216)))
    inputs = teacher_forced_next_frame_inputs(context, actions, candidates)
    assert inputs.shape == (2, 83)
    assert inputs[:, 16].tolist() == [70, 70]
    assert inputs[:, 33].tolist() == [71, 71]
    assert inputs[:, 50].tolist() == [72, 72]
    assert inputs[:, 67].tolist() == [73, 73]
    assert inputs[0, 68:].tolist() == candidates[0, :-1].tolist()
    assert inputs[1, 68:].tolist() == candidates[1, :-1].tolist()


def test_inference_candidate_can_be_materialized_for_teacher_forcing():
    context = torch.arange(4 * 16).reshape(4, 16)
    actions = torch.tensor([70, 71, 72, 73])
    with torch.inference_mode():
        sampled = torch.arange(100, 116).reshape(1, 16)
    ordinary = sampled.detach().clone()
    inputs = teacher_forced_next_frame_inputs(context, actions, ordinary)
    assert not ordinary.is_inference()
    assert not inputs.is_inference()


def test_actor_action_uses_reconstructed_frame_and_temperature():
    class IdentityTokenizer:
        def encode_decode(self, frame, **_kwargs):
            return frame

    class FixedActor:
        def __call__(self, frame):
            assert frame.shape == (1, 3, 64, 64)
            return SimpleNamespace(logits_actions=torch.tensor([[[0.0, 2.0]]]))

    torch.manual_seed(7)
    action, entropy = sample_iris_actor_action(
        IdentityTokenizer(), FixedActor(), torch.zeros(3, 64, 64), temperature=0.5
    )
    assert action in (0, 1)
    assert entropy > 0.0
