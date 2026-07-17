import torch

from dor.multistep import DYN_PER_FRAME, msp_continue


class _ToyGenerate:
    def generate(self, input_ids, max_new_tokens, **_kwargs):
        token = torch.full(
            (input_ids.shape[0], max_new_tokens),
            input_ids.shape[1] % 17,
            dtype=torch.long,
        )
        return torch.cat([input_ids, token], dim=1)


def test_continue_preserves_prefix_and_samples_only_future_frames():
    model = _ToyGenerate()
    ctx = torch.zeros((1, 3), dtype=torch.long)
    actions = torch.ones((4, 13), dtype=torch.long)
    prefix = torch.full((2, DYN_PER_FRAME), 7, dtype=torch.long)
    out = msp_continue(model, ctx, actions, prefix, n_future=2, K=3, seed=1)
    assert out.shape == (3, 2, DYN_PER_FRAME)
    # The toy's generated value depends on the fully appended fixed prefix/actions.
    assert torch.all(out[:, 0] == (3 + 2 * (DYN_PER_FRAME + 13)) % 17)


def test_continue_rejects_action_horizon_overflow():
    model = _ToyGenerate()
    ctx = torch.zeros((1, 3), dtype=torch.long)
    actions = torch.ones((2, 13), dtype=torch.long)
    prefix = torch.zeros((2, DYN_PER_FRAME), dtype=torch.long)
    try:
        msp_continue(model, ctx, actions, prefix, n_future=1, K=1, seed=1)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("expected prefix/action horizon validation")
