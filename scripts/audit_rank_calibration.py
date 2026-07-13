"""Unified zero-training audit for Codec-Conditioned Rank Calibration (C1-a/b/c).

Reads the single-step CNN-FSQ calibration cache and the multi-step compressive-FSQ
temporal cache and reports, with EPISODE-CLUSTER bootstrap (groups sharing an episode
are not independent), how RC calibration repairs the rank channel GRPO consumes:

  * Delta-rho  = mean within-group Pearson gain of the reward against the pre-decode
    code reference when the raw target is swapped for the reachable target;
  * Delta-flip = mean pairwise rank-disagreement reduction against the same reference,
    with the arccos(rho)/pi law checked before and after calibration;
  * RIR (residual-interaction ratio, C1-b) severity quartiles and the pre-registered
    dose-response trend test: cluster-bootstrapped Spearman(RIR, per-group rho gain).

Zero training; every input is an existing audited cache.
"""
import argparse
import json
import os

import numpy as np


def pearson_rows(a, b):
    """Row-wise Pearson correlation for [M,K] arrays."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.zeros(len(a))
    ok = den > 0
    out[ok] = (a * b).sum(axis=1)[ok] / den[ok]
    return out


def flip_rows(r, ref):
    """Row-wise pairwise rank-disagreement fraction vs the reference (ties excluded)."""
    dr = r[:, :, None] - r[:, None, :]
    de = ref[:, :, None] - ref[:, None, :]
    upper = np.triu(np.ones(r.shape[1], dtype=bool), 1)
    valid = upper & (dr != 0) & (de != 0)
    flips = valid & (dr * de < 0)
    return flips.sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_boot(episodes, stat_fn, rounds, seed):
    """Bootstrap over unique episodes; stat_fn(group_mask_indices) -> float."""
    uniq = np.unique(episodes)
    index = {e: np.nonzero(episodes == e)[0] for e in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(rounds)):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[e] for e in chosen])
        stats.append(stat_fn(idx))
    return np.asarray(stats, dtype=np.float64)


def audit_stratum(name, raw, rc, ref, episodes, rounds, seed, report):
    """raw/rc/ref [M,K]; episodes [M] strings."""
    rho_raw = pearson_rows(raw, ref)
    rho_rc = pearson_rows(rc, ref)
    flip_raw = flip_rows(raw, ref)
    flip_rc = flip_rows(rc, ref)
    drho = rho_rc - rho_raw
    dflip = flip_rc - flip_raw
    theory = lambda rho: np.arccos(np.clip(rho, -1, 1)) / np.pi

    boot_drho = cluster_boot(episodes, lambda i: float(drho[i].mean()), rounds, seed)
    boot_dflip = cluster_boot(episodes, lambda i: float(dflip[i].mean()), rounds, seed + 1)

    rir = np.std(raw - rc, axis=1) / (np.std(rc, axis=1) + 1e-8)
    edges = np.quantile(rir, [0.25, 0.5, 0.75])
    quart = np.digitize(rir, edges)
    quart_drho = [float(drho[quart == q].mean()) for q in range(4)]
    boot_trend = cluster_boot(episodes, lambda i: spearman(rir[i], drho[i]),
                              rounds, seed + 2)

    stats = {
        "groups": int(len(raw)),
        "episodes": int(len(np.unique(episodes))),
        "rho_raw": float(rho_raw.mean()), "rho_rc": float(rho_rc.mean()),
        "delta_rho": float(drho.mean()),
        "delta_rho_ci": [float(np.quantile(boot_drho, 0.025)),
                         float(np.quantile(boot_drho, 0.975))],
        "delta_rho_boot_p": float(np.mean(boot_drho <= 0)),
        "flip_raw": float(flip_raw.mean()), "flip_rc": float(flip_rc.mean()),
        "delta_flip": float(dflip.mean()),
        "delta_flip_ci": [float(np.quantile(boot_dflip, 0.025)),
                          float(np.quantile(boot_dflip, 0.975))],
        "delta_flip_boot_p": float(np.mean(boot_dflip >= 0)),
        "theory_gap_raw": float(np.abs(flip_raw - theory(rho_raw)).mean()),
        "theory_gap_rc": float(np.abs(flip_rc - theory(rho_rc)).mean()),
        "rir_quartile_delta_rho": quart_drho,
        "rir_trend_spearman": spearman(rir, drho),
        "rir_trend_ci": [float(np.quantile(boot_trend, 0.025)),
                         float(np.quantile(boot_trend, 0.975))],
        "rir_trend_boot_p": float(np.mean(boot_trend <= 0)),
    }
    report[name] = stats
    print(f"[{name}] groups={stats['groups']} eps={stats['episodes']} | "
          f"rho {stats['rho_raw']:.3f}->{stats['rho_rc']:.3f} "
          f"drho={stats['delta_rho']:+.4f} CI[{stats['delta_rho_ci'][0]:+.4f},"
          f"{stats['delta_rho_ci'][1]:+.4f}] p={stats['delta_rho_boot_p']:.4f} | "
          f"flip {stats['flip_raw']:.3f}->{stats['flip_rc']:.3f} "
          f"dflip={stats['delta_flip']:+.4f} p={stats['delta_flip_boot_p']:.4f} | "
          f"theory {stats['theory_gap_raw']:.4f}/{stats['theory_gap_rc']:.4f}", flush=True)
    print(f"[{name}] RIR quartile drho: "
          + " ".join(f"{v:+.4f}" for v in quart_drho)
          + f" | trend rho={stats['rir_trend_spearman']:+.3f} "
          f"CI[{stats['rir_trend_ci'][0]:+.3f},{stats['rir_trend_ci'][1]:+.3f}] "
          f"p={stats['rir_trend_boot_p']:.4f}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_cache", default="outputs/rankcal/calibration_spatial.npz")
    ap.add_argument("--multi_cache", default="outputs/analysis/temporal_reliability_cache.npz")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3701)
    ap.add_argument("--out", default="outputs/analysis/rank_calibration_audit.json")
    args = ap.parse_args()

    report = {"bootstrap": args.bootstrap, "cluster": "episode"}

    s = np.load(args.single_cache, allow_pickle=False)
    audit_stratum(
        "single_cnnfsq",
        -(s["raw_mse"] + s["raw_lpips"]).astype(np.float64),
        -(s["reach_mse"] + s["reach_lpips"]).astype(np.float64),
        s["code"].astype(np.float64),
        np.asarray(s["episode"]),
        args.bootstrap, args.seed, report,
    )

    m = np.load(args.multi_cache, allow_pickle=False)
    raw, rc, ref = (m["raw_reward"].astype(np.float64), m["rc_reward"].astype(np.float64),
                    m["code_reward"].astype(np.float64))
    g, n, h, k = raw.shape
    eps = np.asarray(m["episode"])
    horizons = m["horizon"].tolist()
    ep_flat = np.broadcast_to(eps[None, :, None], (g, n, h)).reshape(-1)
    audit_stratum(
        "multi_compressive_all",
        raw.reshape(-1, k), rc.reshape(-1, k), ref.reshape(-1, k), ep_flat,
        args.bootstrap, args.seed + 100, report,
    )
    for hi, hz in enumerate(horizons):
        audit_stratum(
            f"multi_h{hz}",
            raw[:, :, hi].reshape(-1, k), rc[:, :, hi].reshape(-1, k),
            ref[:, :, hi].reshape(-1, k),
            np.broadcast_to(eps[None, :], (g, n)).reshape(-1),
            args.bootstrap, args.seed + 200 + hi, report,
        )

    core = [report["single_cnnfsq"], report["multi_compressive_all"]]
    verdict = all(x["delta_rho_boot_p"] < 0.05 and x["delta_flip_boot_p"] < 0.05
                  for x in core)
    trend = all(x["rir_trend_boot_p"] < 0.05 for x in core)
    report["verdict"] = {
        "rank_repair_both_codecs": bool(verdict),
        "rir_dose_response": bool(trend),
    }
    print(f"[verdict] rank repair on both codecs (cluster-boot p<0.05): {verdict}; "
          f"RIR dose-response trend: {trend}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[done] {args.out}", flush=True)
    print("AUDIT_RANK_CALIBRATION_OK", flush=True)


if __name__ == "__main__":
    main()
