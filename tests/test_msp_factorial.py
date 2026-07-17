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
    assert lpips["interaction"]["bootstrap95"][1] < 0
    assert report["c3_training_gate"]["provisional_n5"] is False


def _write_vp2_arm(root, arm, values, credit, reward):
    folder = root / arm
    folder.mkdir()
    protocol = {
        "checkpoint": "/tmp/model", "train_manifest": "/tmp/train.json",
        "eval_manifest": "/tmp/eval.json", "horizon": 8, "K": 16, "eval_K": 4,
        "steps": 30, "batch_windows": 2, "lr": 1e-6, "kl": 0.001,
        "kl_type": "low_var_kl", "gamma": 0.95, "data_seed": 7304,
        "grad_clip": 1.0, "eval_every": 10, "eval_seed": 9917,
        "deterministic": True,
    }
    for seed, value in enumerate(values):
        payload = {
            "args": {**protocol, "credits": credit, "rewards": reward},
            "run": {
                "eval_lpips": [value], "eval_lpips_last": [value + 0.01],
                "eval_mse": [value / 10],
            },
        }
        (folder / f"sweep_{reward}_{credit}_s{seed}.json").write_text(json.dumps(payload))
    return str(folder / "*.json")


def test_vp2_factorial_pilot_gate(tmp_path, monkeypatch):
    patterns = {
        "seq_raw": _write_vp2_arm(tmp_path, "seq_raw", [0.030, 0.031, 0.032], "seq", "raw"),
        "seq_rc": _write_vp2_arm(tmp_path, "seq_rc", [0.029, 0.030, 0.031], "seq", "rc"),
        "return_raw": _write_vp2_arm(
            tmp_path, "return_raw", [0.0295, 0.0305, 0.0315], "return", "raw"
        ),
        "return_rc": _write_vp2_arm(
            tmp_path, "return_rc", [0.027, 0.028, 0.029], "return", "rc"
        ),
    }
    output = tmp_path / "vp2_factorial.json"
    argv = ["analyze_msp_factorial.py", "--platform", "vp2"]
    for name, pattern in patterns.items():
        argv.extend([f"--{name}", pattern])
    argv.extend(["--expected_n", "3", "--bootstrap", "1000", "--out", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    main()

    report = json.loads(output.read_text())
    assert report["platform"] == "vp2"
    assert report["c3_training_gate"]["pilot"] is True


def test_factorial_formal_coupling_gate(tmp_path, monkeypatch):
    seeds = list(range(10))
    seq_raw = [0.220 + 0.001 * seed for seed in seeds]
    seq_rc = [value - 0.002 for value in seq_raw]
    return_raw = [value - 0.001 for value in seq_raw]
    return_rc = [value - 0.008 for value in seq_raw]
    patterns = {
        "seq_raw": _write_arm(tmp_path, "seq_raw", seq_raw, "seq", "raw"),
        "seq_rc": _write_arm(tmp_path, "seq_rc", seq_rc, "seq", "rc"),
        "return_raw": _write_arm(tmp_path, "return_raw", return_raw, "return", "raw"),
        "return_rc": _write_arm(tmp_path, "return_rc", return_rc, "return", "rc"),
    }
    output = tmp_path / "factorial_n10.json"
    argv = ["analyze_msp_factorial.py"]
    for name, pattern in patterns.items():
        argv.extend([f"--{name}", pattern])
    argv.extend(["--expected_n", "10", "--bootstrap", "2000", "--out", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    main()

    report = json.loads(output.read_text())
    assert report["c3_training_gate"]["formal_n10"] is True
    assert report["metrics"]["eval_lpips"]["interaction"]["negative"] == 10
