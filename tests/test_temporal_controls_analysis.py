import json
import sys

from scripts.analyze_temporal_controls import main


BASE = {
    "T": 8, "K": 16, "steps": 30, "batch_windows": 2,
    "train_windows": 24, "eval_windows": 8, "lr": 1e-5,
    "kl": 0.001, "kl_type": "low_var_kl", "temporal_gamma": 0.95,
    "horizon_kl_alpha": 0.0, "eval_every": 10,
    "deterministic": True, "which": "rlvr", "rewards": "rc",
}


def _arm(root, name, values, mode, horizon):
    folder = root / name
    folder.mkdir()
    for seed, value in enumerate(values):
        payload = {
            "args": {**BASE, "adv_temporal": mode, "return_horizon": horizon},
            "run": {"rc-msp": {
                "eval_lpips": [value], "eval_lpips_last": [value + 0.01],
                "eval_mse": [value / 10],
            }},
        }
        (folder / f"sweep_rc_msp_s{seed}.json").write_text(json.dumps(payload))
    return str(folder / "*.json")


def test_temporal_control_analysis(tmp_path, monkeypatch):
    arms = {
        "trunc1": _arm(tmp_path, "trunc1", [.23, .24, .25], "return", 1),
        "trunc3": _arm(tmp_path, "trunc3", [.22, .23, .24], "return", 3),
        "full": _arm(tmp_path, "full", [.20, .21, .22], "return", 0),
        "shuffled": _arm(tmp_path, "shuffled", [.24, .25, .26], "shuffled_return", 0),
    }
    output = tmp_path / "report.json"
    argv = ["analyze_temporal_controls.py"]
    for name, path in arms.items():
        argv.extend([f"--{name}", path])
    argv.extend(["--expected_n", "3", "--out", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    report = json.loads(output.read_text())
    assert report["metrics"]["eval_lpips"]["full_minus_shuffled"]["mean"] < 0
