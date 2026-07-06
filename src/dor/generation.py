"""Candidate next-frame sampling via HF .generate() (bypasses vllm/verl on Blackwell)."""
import torch

from dor.constants import EOS, TPF, VTOK


@torch.no_grad()
def generate_candidates(model, prompt, K, temperature=1.0, top_k=100, seed=0):
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
        min_new_tokens=TPF,
        eos_token_id=None,       # ignore EOS like the official eval
        pad_token_id=EOS,
    )
    gen = out[:, prompt.shape[0]:prompt.shape[0] + TPF]  # [K, TPF]
    return gen.clamp(0, VTOK - 1).long()
