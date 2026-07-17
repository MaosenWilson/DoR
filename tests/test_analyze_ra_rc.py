import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_ra_rc.py"
SPEC = importlib.util.spec_from_file_location("analyze_ra_rc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _series(value):
    return [value + 1.0, value]


def _metrics(latent_key):
    return {
        "eval_lpips": _series(0.2),
        "eval_mse": _series(0.01),
        "eval_psnr": _series(25.0),
        "eval_ssim": _series(0.8),
        latent_key: _series(0.3),
    }


def test_loads_rt1_nested_run_payload(tmp_path):
    names = {
        "a0faithful": "a0faithful-gt_only",
        "a0faithful_tok": "a0faithful_tok-gt_only",
        "ra_rc": "ra_rc-gt_only",
    }
    for filename_arm, run_arm in names.items():
        payload = {"run": {run_arm: _metrics("eval_code_rms")}}
        (tmp_path / f"sweep_{filename_arm}_gt_only_s0.json").write_text(json.dumps(payload))

    runs = MODULE._load(tmp_path, "rt1", "seq")

    assert sorted(runs) == ["ra_rc", "raw", "rc"]
    assert runs["raw"][0]["latent"] == 0.3


def test_loads_external_flat_run_payload_and_intersects_complete_seeds(tmp_path):
    for arm in ("raw", "rc", "ra_rc"):
        for seed in (0, 1):
            payload = {"run": _metrics("eval_latent_rms")}
            (tmp_path / f"sweep_{arm}_seq_s{seed}.json").write_text(json.dumps(payload))
    (tmp_path / "sweep_raw_seq_s2.json").write_text(
        json.dumps({"run": _metrics("eval_latent_rms")})
    )

    runs = MODULE._load(tmp_path, "vp2", "seq")

    assert sorted(runs["raw"]) == [0, 1]
    assert runs["ra_rc"][1]["lpips"] == 0.2


def test_paired_summary_respects_metric_direction():
    lower = MODULE._paired_summary(
        MODULE.np.asarray([-0.2, -0.1, 0.1]), True, rounds=100, seed=1
    )
    higher = MODULE._paired_summary(
        MODULE.np.asarray([0.2, 0.1, -0.1]), False, rounds=100, seed=1
    )

    assert lower["wins"] == 2
    assert higher["wins"] == 2
