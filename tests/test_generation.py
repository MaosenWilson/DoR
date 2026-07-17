from types import SimpleNamespace

import pytest
import torch

from dor.constants import EOS, TPF
from dor.generation import generate_candidates
from dor.tokenization import decode_tokens


class _RecordingModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=False)
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        total = input_ids.shape[1] + kwargs["max_new_tokens"]
        return torch.zeros(
            kwargs["num_return_sequences"], total, dtype=torch.long
        )


def test_candidate_generation_explicitly_enables_kv_cache_without_mutating_config():
    model = _RecordingModel()
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)

    candidates = generate_candidates(model, prompt, K=2, seed=7)

    assert candidates.shape[0] == 2
    assert model.kwargs["use_cache"] is True
    assert model.kwargs["eos_token_id"] == EOS + 1
    assert "min_new_tokens" not in model.kwargs
    assert model.kwargs["max_new_tokens"] == TPF
    assert model.config.use_cache is False


def test_candidate_generation_can_disable_cache_for_equivalence_audits():
    model = _RecordingModel()
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)

    generate_candidates(model, prompt, K=2, seed=7, use_kv_cache=False)

    assert model.kwargs["use_cache"] is False


class _ShortGenerationModel(_RecordingModel):
    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        total = input_ids.shape[1] + 90
        return torch.zeros(
            kwargs["num_return_sequences"], total, dtype=torch.long
        )


def test_candidate_generation_rejects_early_stopping_before_decode():
    model = _ShortGenerationModel()
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(RuntimeError, match="90 new tokens"):
        generate_candidates(model, prompt, K=2, seed=7)


def test_decoder_rejects_incomplete_visual_token_grids():
    with pytest.raises(ValueError, match=r"shape \[N, 320\].*\(16, 90\)"):
        decode_tokens(None, torch.zeros(16, 90, dtype=torch.long))
