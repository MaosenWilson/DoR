"""Offline admission gate for reachability-consistent temporal ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dor.rank_stats import rowwise_spearman
from dor.temporal_credit import discounted_returns, reachability_consistent_temporal_scores


def _returns(values: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(f"expected [context,draw,horizon,candidate], got {values.shape}")
    result = np.empty_like(values)
    for context in range(values.shape[0]):
        for draw in range(values.shape[1]):
            result[context, draw] = discounted_returns(
                values[context, draw].T, gamma
            ).T
    return result


def _scores(raw: np.ndarray, rc: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty_like(raw, dtype=np.float64)
    coverage = np.empty(raw.shape[:-1], dtype=np.float64)
    for context in range(raw.shape[0]):
        for draw in range(raw.shape[1]):
            value, kept = reachability_consistent_temporal_scores(
                raw[context, draw].T, rc[context, draw].T, gamma
            )
            scores[context, draw] = value.T
            coverage[context, draw] = kept
    return scores, coverage


def _pair_precision(score: np.ndarray, reference: np.ndarray, selector=None) -> np.ndarray:
    gap = score[..., :, None] - score[..., None, :]
    ref_gap = reference[..., :, None] - reference[..., None, :]
    group = score.shape[-1]
    valid = np.triu(np.ones((group, group), dtype=bool), k=1)
    valid = valid & (gap != 0.0) & (ref_gap != 0.0)
    if selector is not None:
        valid = valid & selector
    correct = valid & (gap * ref_gap > 0.0)
    axes = (-2, -1)
    return correct.sum(axis=axes) / np.maximum(valid.sum(axis=axes), 1)


def _episode_bootstrap(values, episodes, rounds, seed) -> dict:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes).astype(str)
    unique = np.unique(episodes)
    per_episode = np.asarray([
        np.nanmean(values[episodes == episode]) for episode in unique
    ])
    rng = np.random.default_rng(seed)
    samples = per_episode[
        rng.integers(0, len(per_episode), size=(rounds, len(per_episode)))
    ].mean(axis=1)
    return {
        "mean": float(np.nanmean(per_episode)),
        "ci90": [float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))],
        "episodes": int(len(unique)),
    }


def _evaluate(raw, rc, reference, episodes, gamma, rounds, seed) -> dict:
    raw_return = _returns(raw, gamma)
    rc_return = _returns(rc, gamma)
    reference_return = _returns(reference, gamma)
    score, coverage = _scores(raw, rc, gamma)
    raw_gap = raw_return[..., :, None] - raw_return[..., None, :]
    rc_gap = rc_return[..., :, None] - rc_return[..., None, :]
    concordant = raw_gap * rc_gap > 0.0
    retained_precision = _pair_precision(raw_return, reference_return, concordant)
    raw_precision = _pair_precision(raw_return, reference_return)
    rc_precision = _pair_precision(rc_return, reference_return)
    score_rho = rowwise_spearman(score, reference_return)
    raw_rho = rowwise_spearman(raw_return, reference_return)
    rc_rho = rowwise_spearman(rc_return, reference_return)
    raw_top = np.argmax(raw_return, axis=-1)
    score_top = np.argmax(score, axis=-1)
    raw_max = np.max(raw_return, axis=-1)
    chosen_raw = np.take_along_axis(raw_return, score_top[..., None], axis=-1)[..., 0]
    normalized_regret = (raw_max - chosen_raw) / (
        np.std(raw_return, axis=-1) + 1e-6
    )
    return {
        "coverage": _episode_bootstrap(coverage, episodes, rounds, seed),
        "retained_precision": _episode_bootstrap(
            retained_precision, episodes, rounds, seed + 1
        ),
        "retained_minus_raw_precision": _episode_bootstrap(
            retained_precision - raw_precision, episodes, rounds, seed + 2
        ),
        "retained_minus_rc_precision": _episode_bootstrap(
            retained_precision - rc_precision, episodes, rounds, seed + 3
        ),
        "score_minus_raw_spearman": _episode_bootstrap(
            score_rho - raw_rho, episodes, rounds, seed + 4
        ),
        "score_minus_rc_spearman": _episode_bootstrap(
            score_rho - rc_rho, episodes, rounds, seed + 5
        ),
        "raw_spearman": _episode_bootstrap(raw_rho, episodes, rounds, seed + 6),
        "rc_spearman": _episode_bootstrap(rc_rho, episodes, rounds, seed + 7),
        "score_spearman": _episode_bootstrap(score_rho, episodes, rounds, seed + 8),
        "raw_top_same_fraction": float(np.mean(raw_top == score_top)),
        "normalized_raw_top_regret_q95": float(np.quantile(normalized_regret, 0.95)),
        "score_std_mean": float(np.mean(np.std(score, axis=-1))),
    }


def _shuffle_candidates(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    for index in np.ndindex(values.shape[:-1]):
        values[index] = values[index][rng.permutation(values.shape[-1])]
    return values


def _core_green(report, coverage_min, regret_max) -> bool:
    return bool(
        report["coverage"]["ci90"][0] > coverage_min
        and report["retained_minus_raw_precision"]["ci90"][0] > 0.0
        and report["retained_minus_rc_precision"]["ci90"][0] > 0.0
        and report["score_minus_raw_spearman"]["ci90"][0] > 0.0
        and report["normalized_raw_top_regret_q95"] <= regret_max
        and report["score_std_mean"] > 1e-3
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=16271)
    parser.add_argument("--coverage_min", type=float, default=0.75)
    parser.add_argument("--regret_q95_max", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        raw = np.asarray(cache["raw_reward"], dtype=np.float64)
        rc = np.asarray(cache["rc_reward"], dtype=np.float64)
        reference = np.asarray(cache["post_quant_reward"], dtype=np.float64)
        episodes = np.asarray(cache["episode"]).astype(str)
    if raw.shape != rc.shape or raw.shape != reference.shape:
        raise ValueError("raw, RC, and post-quant rewards must share shape")

    main_report = _evaluate(
        raw, rc, reference, episodes, args.gamma, args.bootstrap, args.seed
    )
    shuffled_report = _evaluate(
        raw, _shuffle_candidates(rc, args.seed + 100), reference, episodes,
        args.gamma, args.bootstrap, args.seed + 200,
    )
    reversed_report = _evaluate(
        raw, -rc, reference, episodes, args.gamma, args.bootstrap, args.seed + 300
    )
    main_green = _core_green(main_report, args.coverage_min, args.regret_q95_max)
    shuffled_green = _core_green(
        shuffled_report, args.coverage_min, args.regret_q95_max
    )
    reversed_green = _core_green(
        reversed_report, args.coverage_min, args.regret_q95_max
    )
    green = main_green and not shuffled_green and not reversed_green
    report = {
        "protocol": "reachability-consistent temporal-ranking offline gate v1",
        "cache": str(Path(args.cache).resolve()),
        "shape": list(raw.shape),
        "gamma": args.gamma,
        "cluster": "episode",
        "thresholds": {
            "coverage_ci90_lower": args.coverage_min,
            "normalized_raw_top_regret_q95_max": args.regret_q95_max,
            "note": "pilot-informed non-inferiority thresholds",
        },
        "main": main_report,
        "controls": {"shuffled_rc": shuffled_report, "reversed_rc": reversed_report},
        "component_verdicts": {
            "main": "GREEN" if main_green else "RED",
            "shuffled_rc": "GREEN" if shuffled_green else "RED",
            "reversed_rc": "GREEN" if reversed_green else "RED",
        },
        "verdict": "GREEN" if green else "RED",
        "scope": "offline admission only; GREEN admits paired GRPO training",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("=== Reachability-Consistent Temporal Ranking Gate ===")
    for name, value in (("main", main_report), ("shuffled", shuffled_report),
                        ("reversed", reversed_report)):
        print(
            f"[{name}] coverage={value['coverage']['mean']:.3f} "
            f"dPrecRaw={value['retained_minus_raw_precision']['mean']:+.4f} "
            f"dPrecRC={value['retained_minus_rc_precision']['mean']:+.4f} "
            f"dRhoRaw={value['score_minus_raw_spearman']['mean']:+.4f} "
            f"regretQ95={value['normalized_raw_top_regret_q95']:.3f}"
        )
    print(f"[verdict] {report['verdict']}")
    print(f"saved {output}\nRCTR_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
