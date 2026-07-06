"""Phase 0 (offline): compute the floor-aware reward weights for the DoR arm.

Reward design (docs/aaai2027/reward_design.md, advisor_meeting_20260627.md):
  R_i = -sum_m  w_m * d_m(i) / s_m
  w_m = sigma_star^2 / (sigma_star^2 + (phi_m/s_m)^2)     # reliability / Wiener weight
where components m in {code (pre-decode), recon=MSE, perc=LPIPS}.

Inputs (all from already-cached, eval-only data; NO training):
  outputs/analysis/reward_spaces_s*.npz   per-window[N] x per-candidate[K] metrics
                                          (keys: lpips, mse, code_rms, ...)
  outputs/analysis/floor_metrics.json     cross-metric reward-noise floor phi_m
                                          (from scripts/probe_floor_metrics.py)

Outputs configs/aaai2027/reward_weights.json with, per component:
  s_m (dataset std), phi_m, phi_norm=phi_m/s_m, w_m
plus sigma_star2 (shared signal var, from the floor-free code component, normalized)
and sigma_eta_norm (floor scale for the GRPO group filter; = LPIPS normalized floor).

This is purely offline numpy. Run on the server where the npz/json live, OR --dry
to self-test the logic on synthetic data locally.
"""
import argparse
import glob
import json
import os

import numpy as np

# component -> npz metric key. code has ~zero floor (pre-decode); recon/perc are post-decode.
COMP = {"code": "code_rms", "recon": "mse", "perc": "lpips"}
FLOOR_KEY = {"code": None, "recon": "mse", "perc": "lpips"}  # key in floor_metrics.json


def load_cache(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    out = {}
    for k in COMP.values():
        arrs = [np.load(f)[k] for f in files if k in np.load(f)]
        out[k] = np.concatenate(arrs, axis=0)  # [sum_N, K]
    return out


def synth():
    """Synthetic [N,K] dry-run: code = clean signal; mse/lpips = signal + floor noise."""
    rng = np.random.default_rng(0)
    N, K = 200, 16
    q = rng.normal(0, 1, (N, K))                 # latent true quality (per candidate)
    code = 0.30 + 0.02 * q                        # clean, small floor
    mse = 0.0035 + 0.0008 * q + 0.0004 * rng.normal(0, 1, (N, K))   # some floor
    lpips = 0.085 + 0.015 * q + 0.03 * rng.normal(0, 1, (N, K))     # high floor
    return {"code_rms": np.abs(code), "mse": np.abs(mse), "lpips": np.abs(lpips)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/analysis/reward_spaces_s*.npz")
    ap.add_argument("--floor", default="outputs/analysis/floor_metrics.json")
    ap.add_argument("--out", default="configs/aaai2027/reward_weights.json")
    ap.add_argument("--dry", action="store_true", help="self-test on synthetic data")
    args = ap.parse_args()

    cache = synth() if args.dry else load_cache(args.cache)
    if cache is None:
        raise SystemExit(f"no cache matched {args.cache!r} (use --dry to self-test)")
    floor = {} if args.dry else json.load(open(args.floor)).get("metrics", {})

    # scales s_m = dataset std of each component distance
    s = {c: float(cache[COMP[c]].std()) + 1e-12 for c in COMP}
    # phi_m: floor in metric m (code ~ 0; recon/perc from floor_metrics.json or synth-known)
    phi = {}
    for c in COMP:
        if FLOOR_KEY[c] is None:
            phi[c] = 0.0
        elif args.dry:
            phi[c] = {"recon": 0.0004, "perc": 0.03}[c]  # known synth noise std
        else:
            phi[c] = float(floor[FLOOR_KEY[c]]["floor_mean"])

    # sigma_star^2: shared within-group signal variance, estimated from the floor-free
    # code component in the SCALE-NORMALIZED space (code/s_code), averaged over windows.
    code_norm = cache[COMP["code"]] / s["code"]
    sigma_star2 = float(np.mean(code_norm.var(axis=1)))

    # reliability / Wiener weight per component (in normalized units)
    phi_norm = {c: phi[c] / s[c] for c in COMP}
    w = {c: sigma_star2 / (sigma_star2 + phi_norm[c] ** 2) for c in COMP}

    payload = {
        "components": {c: {"metric": COMP[c], "s": s[c], "phi": phi[c],
                           "phi_norm": phi_norm[c], "w": w[c]} for c in COMP},
        "sigma_star2": sigma_star2,
        "sigma_eta_norm": phi_norm["perc"],   # floor scale for the GRPO group filter
        "s_code": s["code"],
        "source": "synthetic(--dry)" if args.dry else args.cache,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)

    print(f"sigma_star^2 = {sigma_star2:.5f}   sigma_eta_norm = {phi_norm['perc']:.4f}")
    print(f"{'comp':6s}{'s_m':>12s}{'phi_m':>12s}{'phi/s':>10s}{'w_m':>9s}")
    for c in COMP:
        print(f"{c:6s}{s[c]:12.5f}{phi[c]:12.5f}{phi_norm[c]:10.4f}{w[c]:9.4f}")
    print(f"[done] saved {args.out}")
    print("REWARD_WEIGHTS_OK")


if __name__ == "__main__":
    main()
