"""Reconstruction-calibrated Energy verifier utilities.

The NumPy functions implement the score and admission statistics without a GPU.
``EnergyFeatureGeometry`` lazily imports torch and reuses the frozen LPIPS-VGG
backbone already loaded by :class:`dor.metrics.Metrics`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np


CONFIG_VERSION = 1


def _as_float_array(value, name):
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def combine_block_distances(block_distances, scales, beta=1.0):
    """Combine per-block RMS distances into one Hilbert distance.

    ``block_distances`` has feature blocks on the last axis. The division by the
    number of blocks is equivalent to concatenating equally scaled feature blocks
    with a global ``1/sqrt(B)`` factor.
    """
    blocks = _as_float_array(block_distances, "block_distances")
    scales = _as_float_array(scales, "scales")
    if blocks.ndim < 1 or blocks.shape[-1] != len(scales):
        raise ValueError("block distance/scales shape mismatch")
    if np.any(scales <= 0):
        raise ValueError("all feature block scales must be positive")
    beta = float(beta)
    if not 0.0 < beta < 2.0:
        raise ValueError("Energy exponent beta must lie in (0, 2)")
    distance = np.sqrt(np.mean((blocks / scales) ** 2, axis=-1))
    return distance**beta


def energy_objective(target_distance, pairwise_distance):
    """Finite-ensemble estimate of the higher-is-better Energy objective."""
    target = _as_float_array(target_distance, "target_distance")
    pairwise = _as_float_array(pairwise_distance, "pairwise_distance")
    if target.ndim != 1 or pairwise.shape != (len(target), len(target)):
        raise ValueError("expected target [G] and pairwise [G,G]")
    if len(target) < 2:
        raise ValueError("Energy score requires at least two candidates")
    off_diag = ~np.eye(len(target), dtype=bool)
    return float(-target.mean() + 0.5 * pairwise[off_diag].mean())


def energy_influence(target_distance, pairwise_distance, pairwise_sign=1.0):
    """Per-candidate score-function reward for the Energy objective.

    The population objective contains ``1/2 E d(X,X')``. Differentiating its two
    symmetric policy samples yields the full conditional expectation in this
    candidate influence, hence no extra ``1/2`` appears here.
    """
    target = _as_float_array(target_distance, "target_distance")
    pairwise = _as_float_array(pairwise_distance, "pairwise_distance")
    if target.ndim != 1 or pairwise.shape != (len(target), len(target)):
        raise ValueError("expected target [G] and pairwise [G,G]")
    if len(target) < 2:
        raise ValueError("RC-Energy requires G >= 2")
    if not np.allclose(np.diag(pairwise), 0.0, atol=1e-6):
        raise ValueError("pairwise distance diagonal must be zero")
    return -target + float(pairwise_sign) * pairwise.sum(axis=1) / (len(target) - 1)


def cross_group_utility(raw_target_distance, cross_group_distance):
    """Out-of-group Energy influence using an independent candidate group."""
    target = _as_float_array(raw_target_distance, "raw_target_distance")
    cross = _as_float_array(cross_group_distance, "cross_group_distance")
    if target.ndim != 1 or cross.ndim != 2 or cross.shape[0] != len(target):
        raise ValueError("expected raw target [G] and cross-group distances [G,G2]")
    if cross.shape[1] < 1:
        raise ValueError("independent candidate group cannot be empty")
    return -target + cross.mean(axis=1)


def pair_uncertainty_threshold(raw_distance, reachable_distance, quantile=0.95):
    """Estimate a pairwise rank-corruption radius from a disjoint scale split.

    The candidate-specific target perturbation is ``eta=d(x,y)-d(x,D(E(y)))``.
    Its pairwise difference is exactly the part that can alter candidate order.
    """
    raw = _as_float_array(raw_distance, "raw_distance")
    reachable = _as_float_array(reachable_distance, "reachable_distance")
    if raw.shape != reachable.shape or raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("expected matching raw/reachable distances with shape [N,K]")
    quantile = float(quantile)
    if not 0.5 < quantile < 1.0:
        raise ValueError("uncertainty quantile must lie in (0.5, 1)")
    eta = raw - reachable
    upper = np.triu_indices(eta.shape[1], k=1)
    differences = np.abs(eta[:, :, None] - eta[:, None, :])[:, upper[0], upper[1]]
    return float(np.quantile(differences, quantile))


def _project_order_mask(energy_reward, constraint_mask, *, max_iter=2000, tol=1e-9):
    """Euclidean projection onto row-wise pair-order halfspaces."""
    energy = _as_float_array(energy_reward, "energy_reward")
    constraints_array = np.asarray(constraint_mask, dtype=bool)
    expected = (len(energy), energy.shape[1], energy.shape[1])
    if energy.ndim != 2 or constraints_array.shape != expected:
        raise ValueError("constraint mask must have shape [N,K,K]")
    projected = energy.copy()
    for row in range(len(projected)):
        constraints = [tuple(pair) for pair in np.argwhere(constraints_array[row])]
        if not constraints:
            continue
        corrections = np.zeros((len(constraints), 2), dtype=np.float64)
        values = projected[row]
        for _ in range(int(max_iter)):
            previous = values.copy()
            for ci, (i, j) in enumerate(constraints):
                yi = values[i] + corrections[ci, 0]
                yj = values[j] + corrections[ci, 1]
                if yi < yj:
                    midpoint = 0.5 * (yi + yj)
                    zi = zj = midpoint
                else:
                    zi, zj = yi, yj
                corrections[ci] = (yi - zi, yj - zj)
                values[i], values[j] = zi, zj
            if np.max(np.abs(values - previous)) <= tol:
                break
        else:
            raise RuntimeError("reliable-order projection did not converge")
        violation = max((values[j] - values[i] for i, j in constraints), default=0.0)
        if violation > 1e-7:
            raise RuntimeError(f"reliable-order projection violation: {violation}")
    return projected


def project_reliable_order(energy_reward, rc_reward, threshold, *, max_iter=2000, tol=1e-9):
    """Project Energy rewards onto global high-confidence RC pair constraints."""
    energy = _as_float_array(energy_reward, "energy_reward")
    rc = _as_float_array(rc_reward, "rc_reward")
    if energy.shape != rc.shape or energy.ndim != 2:
        raise ValueError("energy_reward and rc_reward must have matching shape [N,K]")
    threshold = float(threshold)
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    margin = rc[:, :, None] - rc[:, None, :]
    return _project_order_mask(
        energy, margin > threshold, max_iter=max_iter, tol=tol
    )


def project_certified_order(
    energy_reward,
    rc_reward,
    raw_distance,
    reachable_distance,
    *,
    max_iter=2000,
    tol=1e-9,
):
    """Preserve pairs certified against their observed decoder interaction.

    For ``eta_i=d(x_i,y)-d(x_i,D(E(y)))``, a positive RC margin larger
    than ``|eta_i-eta_j|`` cannot be flipped by that target perturbation.
    """
    energy = _as_float_array(energy_reward, "energy_reward")
    rc = _as_float_array(rc_reward, "rc_reward")
    raw = _as_float_array(raw_distance, "raw_distance")
    reachable = _as_float_array(reachable_distance, "reachable_distance")
    if not (energy.shape == rc.shape == raw.shape == reachable.shape) or energy.ndim != 2:
        raise ValueError("all certified-order inputs must have matching shape [N,K]")
    eta = raw - reachable
    margin = rc[:, :, None] - rc[:, None, :]
    interaction = np.abs(eta[:, :, None] - eta[:, None, :])
    return _project_order_mask(
        energy, margin > interaction, max_iter=max_iter, tol=tol
    )


def radial_residual_reward(rc_reward, pair_contribution, target_distance):
    """Remove the radial target-distance component from ensemble spread."""
    rc = _as_float_array(rc_reward, "rc_reward")
    pair = _as_float_array(pair_contribution, "pair_contribution")
    target = _as_float_array(target_distance, "target_distance")
    if rc.shape != pair.shape or rc.shape != target.shape or rc.ndim != 2:
        raise ValueError("rc, pair contribution and target distance must match [N,K]")
    centered_pair = pair - pair.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    denominator = np.sum(centered_target**2, axis=1, keepdims=True)
    slope = np.divide(
        np.sum(centered_pair * centered_target, axis=1, keepdims=True),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    residual = centered_pair - slope * centered_target
    return rc + residual


def top_safe_energy_reward(rc_reward, pair_contribution, max_coefficient=1.0):
    """Diagnostic upper bound: largest per-group coefficient preserving RC top-1."""
    rc = _as_float_array(rc_reward, "rc_reward")
    pair = _as_float_array(pair_contribution, "pair_contribution")
    if rc.shape != pair.shape or rc.ndim != 2:
        raise ValueError("rc_reward and pair_contribution must match [N,K]")
    maximum = float(max_coefficient)
    if maximum < 0.0 or not np.isfinite(maximum):
        raise ValueError("max_coefficient must be finite and non-negative")
    coefficients = np.full(len(rc), maximum, dtype=np.float64)
    for row in range(len(rc)):
        top = int(np.argmax(rc[row]))
        for candidate in range(rc.shape[1]):
            pair_gap = pair[row, candidate] - pair[row, top]
            if candidate != top and pair_gap > 0.0:
                bound = (rc[row, top] - rc[row, candidate]) / pair_gap
                coefficients[row] = min(coefficients[row], max(0.0, bound))
        if coefficients[row] < maximum:
            coefficients[row] = np.nextafter(coefficients[row], 0.0)
    reward = rc + coefficients[:, None] * pair
    # At the analytical crossing point, floating-point multiplication can leave
    # an exact tie whose argmax depends on candidate order. Use a one-ULP tie
    # break so this diagnostic fulfils its stated top-preservation contract.
    for row in range(len(reward)):
        top = int(np.argmax(rc[row]))
        other_max = np.max(np.delete(reward[row], top))
        if reward[row, top] <= other_max:
            reward[row, top] = np.nextafter(other_max, np.inf)
    return reward, coefficients


def rowwise_correlation(a, b, method="pearson"):
    """Correlation for corresponding rows; constant rows return NaN."""
    a = _as_float_array(a, "a")
    b = _as_float_array(b, "b")
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("rowwise correlation expects matching [N,G] arrays")
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    if method == "spearman":
        a = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
        b = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.full(len(a), np.nan, dtype=np.float64)
    valid = den > 1e-12
    out[valid] = (a[valid] * b[valid]).sum(axis=1) / den[valid]
    return out


def episode_bootstrap(values, episodes, rounds=2000, seed=2027):
    """Bootstrap episode means so repeated windows are not treated as independent."""
    values = np.asarray(values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("values contains infinite entries")
    episodes = np.asarray(episodes)
    if values.ndim != 1 or len(values) != len(episodes):
        raise ValueError("values and episodes must be aligned one-dimensional arrays")
    episode_means = []
    for episode in np.unique(episodes):
        current = values[episodes == episode]
        current = current[np.isfinite(current)]
        if len(current):
            episode_means.append(float(current.mean()))
    if not episode_means:
        raise ValueError("no finite episode means for bootstrap")
    episode_means = np.asarray(episode_means, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = episode_means[
        rng.integers(0, len(episode_means), size=(int(rounds), len(episode_means)))
    ].mean(axis=1)
    return {
        "mean": float(episode_means.mean()),
        "q05": float(np.quantile(sampled, 0.05)),
        "q95": float(np.quantile(sampled, 0.95)),
        "episode_means": episode_means.tolist(),
    }


def make_energy_config(block_names, block_scales, *, beta=1.0, metadata=None):
    names = [str(name) for name in block_names]
    scales = _as_float_array(block_scales, "block_scales")
    if len(names) != len(scales) or len(set(names)) != len(names):
        raise ValueError("feature block names/scales are invalid")
    if np.any(scales <= 0):
        raise ValueError("feature block scales must be positive")
    beta = float(beta)
    if not 0.0 < beta < 2.0:
        raise ValueError("Energy exponent beta must lie in (0, 2)")
    return {
        "version": CONFIG_VERSION,
        "geometry": "rgb_plus_normalized_lpips_vgg_blocks",
        "block_names": names,
        "block_scales": scales.tolist(),
        "energy_beta": beta,
        "scale_estimator": "median_base_candidate_to_reachable_target_rms",
        "metadata": dict(metadata or {}),
    }


@lru_cache(maxsize=8)
def load_energy_config(path):
    if not path:
        raise ValueError("RC-Energy reward requires --energy_config")
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"RC-Energy config not found: {resolved}")
    with open(resolved) as handle:
        config = json.load(handle)
    if config.get("version") != CONFIG_VERSION:
        raise ValueError(f"unsupported RC-Energy config version: {config.get('version')}")
    make_energy_config(
        config.get("block_names", []),
        config.get("block_scales", []),
        beta=config.get("energy_beta", 1.0),
        metadata=config.get("metadata", {}),
    )
    return config


class EnergyFeatureGeometry:
    """RGB + normalized LPIPS-VGG feature geometry.

    Torch is imported lazily so all score/statistics tests remain runnable in a
    lightweight local environment.
    """

    def __init__(self, lpips_model):
        self.lpips_model = lpips_model
        self.block_names = ("rgb", "vgg0", "vgg1", "vgg2", "vgg3", "vgg4")

    @staticmethod
    def _channel_normalize(value, eps=1e-10):
        return value / (value.square().sum(dim=1, keepdim=True) + eps).sqrt()

    def extract(self, images):
        """Return feature blocks with shape ``[N,D_b]`` for images in [0,1]."""
        import torch

        images = images.clamp(0.0, 1.0)
        lpips_input = images * 2.0 - 1.0
        if getattr(self.lpips_model, "version", "0.1") == "0.1":
            lpips_input = self.lpips_model.scaling_layer(lpips_input)
        with torch.no_grad():
            outputs = self.lpips_model.net.forward(lpips_input)
        blocks = [images.float().flatten(1)]
        blocks.extend(self._channel_normalize(value.float()).flatten(1) for value in outputs)
        if len(blocks) != len(self.block_names):
            raise RuntimeError(
                f"LPIPS-VGG returned {len(blocks)-1} blocks; expected {len(self.block_names)-1}"
            )
        return tuple(blocks)

    def block_distances(self, left_blocks, right_blocks):
        """Per-block RMS distances ``[N_left,N_right,B]``."""
        import torch

        if len(left_blocks) != len(self.block_names) or len(right_blocks) != len(self.block_names):
            raise ValueError("feature block count mismatch")
        rows = []
        for left, right in zip(left_blocks, right_blocks):
            if left.shape[1] != right.shape[1]:
                raise ValueError("feature block dimensions do not match")
            rows.append(torch.cdist(left, right) / np.sqrt(left.shape[1]))
        return torch.stack(rows, dim=-1)

    def distances_from_config(self, left_blocks, right_blocks, config):
        """Combined Energy distance tensor using a validated frozen config."""
        import torch

        if tuple(config["block_names"]) != self.block_names:
            raise ValueError("RC-Energy config feature blocks do not match LPIPS-VGG geometry")
        block = self.block_distances(left_blocks, right_blocks)
        scales = torch.as_tensor(
            config["block_scales"], device=block.device, dtype=block.dtype
        )
        distance = ((block / scales) ** 2).mean(dim=-1).sqrt()
        return distance.pow(float(config.get("energy_beta", 1.0)))


def energy_candidate_reward(lpips_model, candidates, target, config, *, pairwise=True):
    """Compute pointwise or Energy rewards in the same frozen geometry."""
    geometry = EnergyFeatureGeometry(lpips_model)
    candidate_blocks = geometry.extract(candidates)
    target_blocks = geometry.extract(target.unsqueeze(0) if target.ndim == 3 else target)
    target_distance = geometry.distances_from_config(candidate_blocks, target_blocks, config)[:, 0]
    reward = -target_distance
    if pairwise:
        if len(candidates) < 2:
            raise ValueError("RC-Energy training requires K >= 2")
        pairwise_distance = geometry.distances_from_config(
            candidate_blocks, candidate_blocks, config
        )
        reward = reward + pairwise_distance.sum(dim=1) / (len(candidates) - 1)
    return reward.detach().float().cpu().numpy()


def rc_energy_reward(lpips_model, candidates, target, config):
    """Backward-compatible entry point for the group Energy reward."""
    return energy_candidate_reward(
        lpips_model, candidates, target, config, pairwise=True
    )


def certified_energy_reward(lpips_model, candidates, raw_target, reachable_target, config):
    """RC-Energy projected onto exact decoder-interaction rank certificates."""
    geometry = EnergyFeatureGeometry(lpips_model)
    candidate_blocks = geometry.extract(candidates)
    raw_blocks = geometry.extract(
        raw_target.unsqueeze(0) if raw_target.ndim == 3 else raw_target
    )
    reachable_blocks = geometry.extract(
        reachable_target.unsqueeze(0) if reachable_target.ndim == 3 else reachable_target
    )
    if len(candidates) < 2:
        raise ValueError("certified RC-Energy training requires K >= 2")
    raw_distance = geometry.distances_from_config(
        candidate_blocks, raw_blocks, config
    )[:, 0]
    reachable_distance = geometry.distances_from_config(
        candidate_blocks, reachable_blocks, config
    )[:, 0]
    pairwise_distance = geometry.distances_from_config(
        candidate_blocks, candidate_blocks, config
    )
    rc = -reachable_distance
    energy = rc + pairwise_distance.sum(dim=1) / (len(candidates) - 1)
    projected = project_certified_order(
        energy.detach().float().cpu().numpy()[None, :],
        rc.detach().float().cpu().numpy()[None, :],
        raw_distance.detach().float().cpu().numpy()[None, :],
        reachable_distance.detach().float().cpu().numpy()[None, :],
    )[0]
    return projected.astype(np.float32, copy=False)
