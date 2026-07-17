"""Candidate next-frame sampling via HF .generate() (bypasses vllm/verl on Blackwell)."""
import torch

from dor.constants import EOS, TPF, VTOK


# Passing None can fall back to model.generation_config.eos_token_id in some
# Transformers versions. EOS + 1 lies just beyond this model's vocabulary, so
# it cannot be sampled and implements the official ignore-EOS protocol.
_IGNORE_EOS_ID = EOS + 1


@torch.no_grad()
def generate_candidates(
    model,
    prompt,
    K,
    temperature=1.0,
    top_k=100,
    seed=0,
    use_kv_cache=True,
):
    """prompt [PROMPT_LEN] -> candidate visual tokens [K, TPF] long.

    top_k=100 aligns with the RLVR-World official eval default
    (eval_vgpt.py: --topk default 100, temperature 1.0, ignore_eos) so the
    generator stays comparable to the RLVR-World baseline.
    """
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=prompt.unsqueeze(0),
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=K,
        max_new_tokens=TPF,      # only the visual tokens
        eos_token_id=_IGNORE_EOS_ID,
        pad_token_id=EOS,
        # Training keeps model.config.use_cache=False because teacher-forced
        # backward does not need KV states. Autoregressive sampling does: without
        # this explicit override, every new token recomputes the 1,333-token
        # prompt and all preceding visual tokens.
        use_cache=bool(use_kv_cache),
    )
    if not isinstance(out, torch.Tensor) or out.ndim != 2:
        raise RuntimeError(
            "candidate generation must return a rank-2 token tensor; "
            f"got {type(out).__name__} with shape={getattr(out, 'shape', None)}"
        )
    expected_shape = (int(K), int(prompt.numel() + TPF))
    if tuple(out.shape) != expected_shape:
        generated = int(out.shape[1] - prompt.numel()) if out.shape[1] >= prompt.numel() else -1
        raise RuntimeError(
            "fixed-length candidate generation violated the video-frame contract: "
            f"expected output shape {expected_shape} ({TPF} new tokens), got "
            f"{tuple(out.shape)} ({generated} new tokens). Check EOS/stopping settings."
        )
    gen = out[:, prompt.shape[0]:prompt.shape[0] + TPF]  # [K, TPF]
    return gen.clamp(0, VTOK - 1).long()
