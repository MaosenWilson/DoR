"""Legal FSQ-neighborhood utilities for reachable-target audits."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from dor.tokenization import decode_tokens


def fsq_basis(levels, *, device=None):
    levels = [int(value) for value in levels]
    if not levels or any(value < 2 for value in levels):
        raise ValueError("FSQ levels must contain integers >= 2")
    basis = [1]
    for value in levels[:-1]:
        basis.append(basis[-1] * value)
    return torch.tensor(basis, device=device, dtype=torch.long)


def indices_to_levels(indices, levels):
    """Convert combined FSQ indices to one integer coordinate per level."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    basis = fsq_basis(levels, device=indices.device)
    level_tensor = torch.tensor(levels, device=indices.device, dtype=torch.long)
    return (indices[..., None] // basis) % level_tensor


def levels_to_indices(level_indices, levels):
    level_indices = torch.as_tensor(level_indices, dtype=torch.long)
    if level_indices.shape[-1] != len(levels):
        raise ValueError("last dimension must match the number of FSQ levels")
    basis = fsq_basis(levels, device=level_indices.device)
    return (level_indices * basis).sum(dim=-1)


def legal_adjacent_neighbors(indices, levels, positions):
    """Enumerate one-level legal moves at selected flattened spatial positions."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    if indices.ndim != 2:
        raise ValueError(f"indices must be [H,W], got {tuple(indices.shape)}")
    level_indices = indices_to_levels(indices, levels)
    flat = level_indices.reshape(-1, len(levels))
    candidates = []
    for position in positions:
        position = int(position)
        if position < 0 or position >= len(flat):
            raise IndexError(f"position {position} is outside the token grid")
        for channel, maximum in enumerate(levels):
            value = int(flat[position, channel])
            for delta in (-1, 1):
                moved = value + delta
                if 0 <= moved < int(maximum):
                    proposal = flat.clone()
                    proposal[position, channel] = moved
                    candidates.append(levels_to_indices(proposal, levels).reshape_as(indices))
    if not candidates:
        return torch.empty((0,) + tuple(indices.shape), device=indices.device, dtype=torch.long)
    return torch.stack(candidates)


def highest_error_positions(target, reconstruction, latent_hw, topk):
    """Select latent cells whose corresponding image regions have largest MSE."""
    if target.shape != reconstruction.shape or target.ndim != 3:
        raise ValueError("target and reconstruction must be matching [C,H,W] tensors")
    error = (target.float() - reconstruction.float()).square().mean(dim=0, keepdim=True)
    pooled = F.adaptive_avg_pool2d(error.unsqueeze(0), latent_hw).flatten()
    count = min(int(topk), pooled.numel())
    if count <= 0:
        raise ValueError("topk must be positive")
    return torch.topk(pooled, count, largest=True, sorted=True).indices


def hamming_fraction(left, right):
    left = torch.as_tensor(left)
    right = torch.as_tensor(right)
    if left.shape != right.shape:
        raise ValueError("Hamming inputs must have matching shapes")
    return float((left != right).float().mean().item())


@torch.no_grad()
def decode_metric_score(tok, metrics, indices, target, batch_size=8):
    """Score reachable token grids with the verifier's MSE+LPIPS objective."""
    scores, parts = [], []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        images = decode_tokens(tok, batch.reshape(len(batch), -1))
        quality = metrics.eval_batch(images, target)
        mse = torch.as_tensor(quality["mse"], dtype=torch.float64).cpu().numpy()
        lpips = torch.as_tensor(quality["lpips"], dtype=torch.float64).cpu().numpy()
        scores.extend((mse + lpips).tolist())
        parts.extend(zip(mse.tolist(), lpips.tolist()))
    return torch.tensor(scores, dtype=torch.float64).numpy(), torch.tensor(parts).numpy()


@torch.no_grad()
def greedy_metric_refine(
    tok,
    metrics,
    base,
    target,
    levels,
    *,
    positions=8,
    rounds=2,
    batch_size=8,
    min_improvement=1e-7,
):
    """Apply a fixed number of legal FSQ coordinate-refinement moves."""
    current = base.clone()
    current_image = decode_tokens(tok, current.reshape(1, -1))[0]
    current_scores, _ = decode_metric_score(
        tok, metrics, current.unsqueeze(0), target, batch_size
    )
    current_score = float(current_scores[0])
    trace, evaluated = [], 0
    for _ in range(int(rounds)):
        selected = highest_error_positions(
            target, current_image, tuple(current.shape), positions
        )
        proposals = legal_adjacent_neighbors(current, levels, selected)
        if not len(proposals):
            break
        scores, _ = decode_metric_score(tok, metrics, proposals, target, batch_size)
        evaluated += len(proposals)
        best = int(scores.argmin())
        improvement = current_score - float(scores[best])
        if improvement <= min_improvement:
            break
        current = proposals[best]
        current_score = float(scores[best])
        current_image = decode_tokens(tok, current.reshape(1, -1))[0]
        trace.append({
            "objective": current_score,
            "improvement": improvement,
            "candidate_positions": [int(value) for value in selected.cpu().tolist()],
        })
    return current, trace, evaluated


def matched_random_legal_target(base, levels, candidate_positions, changed_cells, seed):
    """Random legal target with an exact token-cell Hamming budget."""
    base = torch.as_tensor(base, dtype=torch.long)
    positions = torch.as_tensor(candidate_positions, dtype=torch.long).flatten()
    changed_cells = int(changed_cells)
    if changed_cells < 0 or changed_cells > len(positions):
        raise ValueError("changed_cells must fit within candidate_positions")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(positions), generator=generator)[:changed_cells]
    chosen = positions.cpu()[order].tolist()
    coordinates = indices_to_levels(base, levels).reshape(-1, len(levels)).clone()
    for position in chosen:
        options = []
        for channel, maximum in enumerate(levels):
            value = int(coordinates[position, channel])
            for delta in (-1, 1):
                if 0 <= value + delta < int(maximum):
                    options.append((channel, delta))
        pick = int(torch.randint(len(options), (1,), generator=generator))
        channel, delta = options[pick]
        coordinates[position, channel] += delta
    return levels_to_indices(coordinates, levels).reshape_as(base)


def load_mrrt_cache(path):
    """Load cached targets keyed by ``(episode basename, start index)``."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MRRT target cache not found: {path!r}")
    payload = np.load(path, allow_pickle=False)
    required = {"episodes", "starts", "mrrt", "mrrt_random"}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"MRRT cache is missing arrays: {sorted(missing)}")
    count = len(payload["episodes"])
    if any(len(payload[name]) != count for name in required):
        raise ValueError("MRRT cache arrays have inconsistent lengths")
    cache = {}
    for index in range(count):
        key = (str(payload["episodes"][index]), int(payload["starts"][index]))
        if key in cache:
            raise ValueError(f"duplicate MRRT cache key: {key}")
        cache[key] = {
            "mrrt": np.asarray(payload["mrrt"][index], dtype=np.int64),
            "mrrt_random": np.asarray(payload["mrrt_random"][index], dtype=np.int64),
        }
    return cache
