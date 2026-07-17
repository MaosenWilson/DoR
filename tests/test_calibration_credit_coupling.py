import numpy as np

from scripts.audit_calibration_credit_coupling import audit


def test_coupling_audit_detects_accumulated_residual(tmp_path):
    rng = np.random.default_rng(17)
    repetitions, windows, horizons, group = 2, 12, 4, 10
    reference = rng.normal(size=(repetitions, windows, horizons, group))
    rc = reference + rng.normal(scale=0.05, size=reference.shape)
    raw = rc + rng.normal(scale=0.45, size=reference.shape)
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        raw_reward=raw.astype(np.float32),
        rc_reward=rc.astype(np.float32),
        code_reward=reference.astype(np.float32),
        episode=np.asarray([f"ep{index // 2}" for index in range(windows)]),
        horizon=np.arange(2, 2 + horizons),
    )

    report = audit(str(cache), gamma=0.95, bootstrap=300, seed=4)
    assert report["aggregate"]["delta_rho"]["mean"] > 0
    assert report["aggregate"]["delta_flip"]["mean"] < 0
    assert report["aggregate"]["early_minus_late_residual_dispersion"]["mean"] > 0
    assert report["shape"]["horizons"] == horizons
