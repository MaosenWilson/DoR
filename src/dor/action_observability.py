"""Diagnostics for action observability in frozen visual-token transitions.

This module deliberately stops before candidate generation or policy training.
It asks which temporal alignment and spatial resolution, if any, make actions
recoverable from real transitions under increasingly strict data splits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dor.action_verifier import ARM_MOTION_DIMS, split_episodes, transition_features


ALPHA_GRID = tuple(10.0**p for p in range(-4, 5))


def episode_folds(n_episodes, n_folds=5, seed=2027):
    """Deterministic complete episode partition for outer cross-fitting."""
    if n_folds < 2 or n_episodes < n_folds:
        raise ValueError("cross-fitting requires 2 <= n_folds <= n_episodes")
    order = np.random.default_rng(seed).permutation(int(n_episodes))
    folds = [np.sort(x.astype(int)) for x in np.array_split(order, int(n_folds))]
    joined = np.concatenate(folds)
    if not np.array_equal(np.sort(joined), np.arange(n_episodes)):
        raise AssertionError("outer folds must cover every episode exactly once")
    return folds


def effect_indices(episode_id, step, horizon):
    """Return common frame pairs valid for offsets -1, 0 and +1.

    The returned starts exclude the first frame and the final ``horizon``
    frames, so all pre-registered action targets are defined on identical rows.
    """
    episode_id = np.asarray(episode_id, dtype=int)
    step = np.asarray(step, dtype=int)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    starts, ends, episodes, steps = [], [], [], []
    for episode in np.unique(episode_id):
        loc = np.flatnonzero(episode_id == episode)
        loc = loc[np.argsort(step[loc])]
        local_steps = step[loc]
        if len(loc) < horizon + 2:
            continue
        if not np.array_equal(local_steps, np.arange(len(loc))):
            raise ValueError(f"episode {episode} does not contain contiguous frame steps")
        t = np.arange(1, len(loc) - horizon, dtype=int)
        starts.extend(loc[t])
        ends.extend(loc[t + horizon])
        episodes.extend(np.full(len(t), episode, dtype=int))
        steps.extend(t)
    return tuple(np.asarray(x, dtype=int) for x in (starts, ends, episodes, steps))


def action_targets(actions, starts, episode_id, horizon, offset, mode):
    """Build aligned first-action or interval-mean targets."""
    actions = np.asarray(actions, dtype=np.float64)
    starts = np.asarray(starts, dtype=int)
    episode_id = np.asarray(episode_id, dtype=int)
    if offset not in (-1, 0, 1):
        raise ValueError("offset must be one of -1, 0, +1")
    if mode not in ("first", "mean"):
        raise ValueError("mode must be 'first' or 'mean'")
    width = 1 if mode == "first" else int(horizon)
    rows = np.stack([starts + offset + u for u in range(width)], axis=1)
    if np.any(rows < 0) or np.any(rows >= len(actions)):
        raise ValueError("an aligned action target falls outside the cache")
    if not np.all(episode_id[rows] == episode_id[starts, None]):
        raise ValueError("an aligned action target crosses an episode boundary")
    return actions[rows].mean(axis=1)[:, np.asarray(ARM_MOTION_DIMS, dtype=int)]


def pooled_effect_features(codes, starts, ends, pool_hw, batch_size=512):
    """Build state/signed-delta/absolute-delta features in bounded batches."""
    codes = np.asarray(codes)
    chunks = []
    for first in range(0, len(starts), int(batch_size)):
        sl = slice(first, first + int(batch_size))
        z0 = torch.from_numpy(codes[starts[sl]]).float()
        zh = torch.from_numpy(codes[ends[sl]]).float()
        chunks.append(transition_features(z0, zh, pool_hw).numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def motion_oracle_features(current_frames, flow, pool_hw):
    """Return pooled RGB-state and signed/magnitude flow features.

    Args:
        current_frames: float tensor ``[N,3,H,W]`` in ``[0,1]``.
        flow: frozen optical flow ``[N,2,h,w]``.
    """
    if current_frames.ndim != 4 or current_frames.shape[1] != 3:
        raise ValueError("current_frames must have shape [N,3,H,W]")
    if flow.ndim != 4 or flow.shape[1] != 2 or len(flow) != len(current_frames):
        raise ValueError("flow must have shape [N,2,h,w] with matching batch size")
    state = torch.nn.functional.adaptive_avg_pool2d(
        current_frames.float(), pool_hw
    ).flatten(1)
    magnitude = flow.float().square().sum(1, keepdim=True).sqrt()
    motion_map = torch.cat((flow.float(), flow.float().abs(), magnitude), dim=1)
    motion = torch.nn.functional.adaptive_avg_pool2d(motion_map, pool_hw).flatten(1)
    return torch.cat((state, motion), dim=1), int(state.shape[1])


def split_masks(protocol, episodes, steps, horizon, episode_names, seed=2027):
    """Create random, temporally blocked, or episode-disjoint masks."""
    episodes = np.asarray(episodes, dtype=int)
    steps = np.asarray(steps, dtype=int)
    n = len(episodes)
    if protocol == "random":
        order = np.random.default_rng(seed).permutation(n)
        n_train = int(np.floor(0.60 * n))
        n_cal = int(np.floor(0.20 * n))
        masks = {name: np.zeros(n, dtype=bool) for name in ("train", "calibration", "test")}
        masks["train"][order[:n_train]] = True
        masks["calibration"][order[n_train:n_train + n_cal]] = True
        masks["test"][order[n_train + n_cal:]] = True
        return masks
    if protocol == "blocked":
        masks = {name: np.zeros(n, dtype=bool) for name in ("train", "calibration", "test")}
        gap = int(horizon)
        for episode in np.unique(episodes):
            loc = np.flatnonzero(episodes == episode)
            loc = loc[np.argsort(steps[loc])]
            b1 = int(np.floor(0.60 * len(loc)))
            b2 = int(np.floor(0.80 * len(loc)))
            masks["train"][loc[:max(0, b1 - gap)]] = True
            masks["calibration"][loc[min(len(loc), b1 + gap):max(b1 + gap, b2 - gap)]] = True
            masks["test"][loc[min(len(loc), b2 + gap):]] = True
        return masks
    if protocol == "episode":
        row_names = np.asarray(episode_names, dtype=str)[episodes]
        split = split_episodes(row_names, seed)
        return {name: np.isin(row_names, values) for name, values in split.items()}
    raise ValueError(f"unknown split protocol: {protocol}")


@dataclass
class RidgeDesign:
    """Reusable standardized SVD design for a ridge path."""

    x_mean: np.ndarray
    x_scale: np.ndarray
    u: torch.Tensor
    singular: torch.Tensor
    vh: torch.Tensor
    device: str

    @classmethod
    def from_array(cls, x, device="cpu"):
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(0)
        scale = np.maximum(x.std(0), 1e-6)
        xn = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
        u, singular, vh = torch.linalg.svd(xn, full_matrices=False)
        return cls(mean, scale, u, singular, vh, str(device))

    def fit_parameters(self, y, alpha):
        y = np.asarray(y, dtype=np.float64)
        y_mean = y.mean(0)
        y_scale = np.maximum(y.std(0), 1e-6)
        yn = torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float32, device=self.u.device)
        uy = self.u.T @ yn
        # Keep alpha invariant when the final fit adds calibration rows.
        shrink = self.singular / (self.singular.square() + len(y) * float(alpha))
        coef = self.vh.T @ (shrink[:, None] * uy)
        return {
            "x_mean": self.x_mean.copy(),
            "x_scale": self.x_scale.copy(),
            "y_mean": y_mean,
            "y_scale": y_scale,
            "coef": coef.cpu().numpy().astype(np.float64),
            "alpha": float(alpha),
        }

    def fit_predict(self, y, x_eval, alpha):
        parameters = self.fit_parameters(y, alpha)
        xe = np.asarray(x_eval, dtype=np.float64)
        xen = torch.as_tensor(
            (xe - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.u.device
        )
        coef = torch.as_tensor(parameters["coef"], dtype=torch.float32, device=self.u.device)
        pred = (xen @ coef).cpu().numpy().astype(np.float64)
        return pred * parameters["y_scale"] + parameters["y_mean"]


def r2_per_dim(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def direction_balanced_accuracy(y_train, y, pred):
    """Macro sign balanced accuracy outside a train-derived dead zone."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    thresholds = np.quantile(np.abs(y_train), 0.25, axis=0)
    scores = []
    for dim in range(y.shape[1]):
        active = np.abs(y[:, dim]) > thresholds[dim]
        truth = y[active, dim] >= 0
        guess = pred[active, dim] >= 0
        if not np.any(truth) or np.all(truth):
            continue
        tpr = np.mean(guess[truth])
        tnr = np.mean(~guess[~truth])
        scores.append(0.5 * (tpr + tnr))
    return float(np.mean(scores)) if scores else float("nan")


def retrieval_rows(y_train, y, pred, episodes):
    """Pair matched actions against deterministic within-episode alternatives."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    episodes = np.asarray(episodes, dtype=int)
    scale = np.maximum(y_train.std(0), 1e-6)
    true_error = np.mean(((pred - y) / scale) ** 2, axis=1)
    wins = np.full(len(y), np.nan, dtype=np.float64)
    for episode in np.unique(episodes):
        loc = np.flatnonzero(episodes == episode)
        if len(loc) < 2:
            continue
        alternatives = []
        for shift in sorted(set((1, max(1, len(loc) // 4), max(1, len(loc) // 2)))):
            y_neg = np.roll(y[loc], shift, axis=0)
            neg_error = np.mean(((pred[loc] - y_neg) / scale) ** 2, axis=1)
            alternatives.append(true_error[loc] < neg_error)
        wins[loc] = np.mean(np.stack(alternatives, axis=1), axis=1)
    return wins


def episode_bootstrap(values, episodes, n_boot=2000, seed=2027):
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes, dtype=int)
    means = []
    for episode in np.unique(episodes):
        current = values[episodes == episode]
        current = current[np.isfinite(current)]
        if len(current):
            means.append(float(np.mean(current)))
    means = np.asarray(means, dtype=np.float64)
    if not len(means):
        return {"mean": None, "q05": None, "q95": None, "episode_means": []}
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(int(n_boot), len(means)))].mean(1)
    return {
        "mean": float(means.mean()),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
        "episode_means": means.tolist(),
    }


def evaluate_prediction(y_train, y, pred, episodes, n_boot=2000, seed=2027):
    r2 = r2_per_dim(y, pred)
    retrieval = retrieval_rows(y_train, y, pred, episodes)
    return {
        "mean_r2": float(np.mean(r2)),
        "r2_per_dim": r2.tolist(),
        "positive_dims": int(np.sum(r2 > 0)),
        "direction_balanced_accuracy": direction_balanced_accuracy(y_train, y, pred),
        "retrieval": episode_bootstrap(retrieval, episodes, n_boot, seed),
    }


def select_alpha(design, y_train, x_cal, y_cal, alpha_grid=ALPHA_GRID):
    rows = []
    for alpha in alpha_grid:
        pred = design.fit_predict(y_train, x_cal, alpha)
        rows.append({"alpha": float(alpha), "mean_r2": float(np.mean(r2_per_dim(y_cal, pred)))})
    best = max(rows, key=lambda row: (row["mean_r2"], -row["alpha"]))
    return best, rows
