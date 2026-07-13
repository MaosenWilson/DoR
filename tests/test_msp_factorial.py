import json
import sys

from scripts.analyze_msp_factorial import main


PROTOCOL = {
    "T": 8,
    "K": 16,
    "steps": 30,
    "batch_windows": 2,
    "train_windows": 24,
    "eval_windows": 8,
    "lr": 1e-5,
    "kl": 0.001,
    "kl_type": "low_var_kl",
    "temporal_gamma": 0.95,
    "return_horizon": 0,
    "horizon_kl_alpha": 0.0,
    "eval_every": 10,
    "deterministic": True,
    "which": "rlvr",
}


def _write_arm(root, arm, values, adv, reward):
    folder = root / arm
    folder.mkdir()
    for seed, value in enumerate(values):
        payload = {
            "args": {**PROTOCOL, "adv_temporal": adv, "rewards": reward},
            "run": {f"{reward}-msp": {
                "eval_lpips": [value],
                "eval_lpips_last": [value + 0.01],
                "eval_mse": [value / 10],
            }},
        }
        (folder / f"sweep_{arm}_msp_s{seed}.json").write_text(json.dumps(payload))
    return str(folder / "*.json")


def test_factorial_analysis_end_to_end(tmp_path, monkeypatch):
    patterns = {
        "seq_raw": _write_arm(tmp_path, "seq_raw", [0.22, 0.23, 0.24], "seq", "raw"),
        "seq_rc": _write_arm(tmp_path, "seq_rc", [0.21, 0.22, 0.23], "seq", "rc"),
        "return_raw": _write_arm(
            tmp_path, "return_raw", [0.215, 0.225, 0.235], "return", "raw"
        ),
        "return_rc": _write_arm(
            tmp_path, "return_rc", [0.20, 0.21, 0.22], "return", "rc"
        ),
    }
    output = tmp_path / "factorial.json"
    argv = ["analyze_msp_factorial.py"]
    for name, pattern in patterns.items():
        argv.extend([f"--{name}", pattern])
    argv.extend(["--expected_n", "3", "--out", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    main()

    report = json.loads(output.read_text())
    lpips = report["metrics"]["eval_lpips"]
    assert report["paired_seeds"] == [0, 1, 2]
    assert lpips["rc_effect_under_seq"]["mean"] < 0
    assert lpips["return_effect_under_rc"]["mean"] < 0
    assert lpips["interaction"]["mean"] < 0
