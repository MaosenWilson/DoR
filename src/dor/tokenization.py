"""Visual tokenization helpers and prompt construction."""
import torch

from dor.constants import ABINS, BOS, CTX, GRID, TPF, VTOK


@torch.no_grad()
def encode_indices(tok, frames):
    """frames [T,3,256,320] in [0,1] -> visual token indices [T,16,20] long."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        idx = tok.encode(frames.unsqueeze(0))  # [1,T,16,20]
    return idx[0].long()


@torch.no_grad()
def encode_features(tok, frames):
    """Continuous pre-quant features (frozen encoder phi), GAP-pooled.
    frames [N,3,256,320] in [0,1] -> [N, C'] float32."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        d = tok.encoder(frames)        # [N,C,h,w]
        d = tok.quant_linear(d)        # [N,C',h,w]
    return d.float().mean(dim=(-1, -2))  # GAP -> [N,C']


@torch.no_grad()
def encode_feature_map(tok, frames):
    """Continuous pre-quant feature MAP (frozen encoder phi), spatial kept.
    frames [N,3,256,320] in [0,1] -> [N, C'*h*w] float32 (flattened, NO GAP).

    Matches code_rms's full-spatial granularity so the phi arm (A5) and the code
    arm (A6) live in the same space and isolate exactly the quantization step
    (A5 vs A6). The GAP-pooled `encode_features` above is only for the consensus
    global descriptor and must NOT be used as the phi reward (it discards the
    spatial structure that distinguishes candidates)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        d = tok.encoder(frames)        # [N,C,h,w]
        d = tok.quant_linear(d)        # [N,C',h,w]
    return d.float().reshape(frames.shape[0], -1)  # [N, C'*h*w]


@torch.no_grad()
def decode_tokens(tok, vis_tokens):
    """vis_tokens [N,320] long -> images [N,3,256,320] in [0,1]."""
    idx = vis_tokens.clamp(0, VTOK - 1).long().reshape(-1, 1, GRID[0], GRID[1])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dec = tok.decode(idx)          # [N,1,3,256,320]
    return dec.float().reshape(-1, 3, 256, 320).clamp(0.0, 1.0)


def discretize_actions(actions, ar, bins=ABINS):
    lo, hi = ar[:, 0], ar[:, 1]
    a = torch.clip((actions - lo) / (hi - lo + 1e-8), 0, 1)
    return torch.floor(a * bins).to(torch.long).clip(0, bins - 1)


@torch.no_grad()
def build_prompt(tok, frames, actions, ar):
    """frames [>=CTX+1,...], actions [>=CTX+1,13] -> prompt ids [PROMPT_LEN] long."""
    vis = encode_indices(tok, frames[:CTX]).reshape(CTX, -1)   # [CTX,320]
    act = discretize_actions(actions[:CTX], ar) + VTOK         # [CTX,13]
    blocks = torch.cat([vis, act], dim=1).reshape(-1)          # [CTX*333]
    bos = torch.tensor([BOS], device=blocks.device, dtype=blocks.dtype)
    return torch.cat([blocks, bos]).long()
