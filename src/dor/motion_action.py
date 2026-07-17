"""Cross-fitted decoded-motion action verifier payload utilities."""
from __future__ import annotations

import json

import numpy as np

from dor.action_verifier import ARM_MOTION_DIMS


def predict_model(payload, fold, features):
    features = np.asarray(features, dtype=np.float64)
    xn = (features - payload["x_mean"][fold]) / payload["x_scale"][fold]
    yn = xn @ payload["coef"][fold]
    return yn * payload["y_scale"][fold] + payload["y_mean"][fold]


def action_score(payload, fold, features, action):
    """Reliability-weighted negative action residual; higher is better."""
    pred = predict_model(payload, fold, features)
    target = np.asarray(action, dtype=np.float64)
    if target.ndim == 1:
        target = target[None]
    target = target[:, np.asarray(payload["action_dims"], dtype=int)]
    if len(target) == 1 and len(pred) != 1:
        target = np.broadcast_to(target, pred.shape)
    residual = (pred - target) / np.maximum(payload["residual_scale"], 1e-6)
    weights = np.asarray(payload["dimension_weights"], dtype=np.float64)
    return -np.sum(weights[None] * residual**2, axis=1) / np.maximum(weights.sum(), 1e-12)


def episode_fold(payload, episode_name):
    name = str(episode_name)
    for fold, names in enumerate(payload["fold_test_episode_names"]):
        if name in set(str(x) for x in names):
            return fold
    raise KeyError(f"episode {name!r} is absent from cross-fit payload")


def save_payload(path, models, residual_scale, dimension_weights, fold_names, metadata):
    np.savez_compressed(
        path,
        x_mean=np.stack([m["x_mean"] for m in models]).astype(np.float32),
        x_scale=np.stack([m["x_scale"] for m in models]).astype(np.float32),
        y_mean=np.stack([m["y_mean"] for m in models]).astype(np.float32),
        y_scale=np.stack([m["y_scale"] for m in models]).astype(np.float32),
        coef=np.stack([m["coef"] for m in models]).astype(np.float32),
        alpha=np.asarray([m["alpha"] for m in models], dtype=np.float32),
        residual_scale=np.asarray(residual_scale, dtype=np.float32),
        dimension_weights=np.asarray(dimension_weights, dtype=np.float32),
        action_dims=np.asarray(ARM_MOTION_DIMS, dtype=np.int64),
        fold_test_episode_names=np.asarray(fold_names, dtype=str),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_payload(path):
    data = np.load(path, allow_pickle=False)
    return {
        "x_mean": data["x_mean"].astype(np.float64),
        "x_scale": data["x_scale"].astype(np.float64),
        "y_mean": data["y_mean"].astype(np.float64),
        "y_scale": data["y_scale"].astype(np.float64),
        "coef": data["coef"].astype(np.float64),
        "alpha": data["alpha"].astype(np.float64),
        "residual_scale": data["residual_scale"].astype(np.float64),
        "dimension_weights": data["dimension_weights"].astype(np.float64),
        "action_dims": data["action_dims"].astype(int),
        "fold_test_episode_names": data["fold_test_episode_names"].astype(str),
        "metadata": json.loads(str(data["metadata"].item())),
    }
