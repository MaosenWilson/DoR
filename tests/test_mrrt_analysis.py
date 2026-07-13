import json
import sys

from scripts.analyze_mrrt_training import ARM_REWARD, main


BASE = {
    "rewards": "a0faithful,a0faithful_tok,mrrt,mrrt_random",
    "modes": "gt_only", "steps": 150, "K": 16, "batch_windows": 2,
    "train_windows": 24, "eval_windows": 12, "lr": 1e-5,
    "eval_every": 10, "deterministic": True,
}


def _arm(root, arm, values):
    reward = ARM_REWARD[arm]
    folder = root / arm
    folder.mkdir()
    for seed, value in enumerate(values):
        run = {metric: [value] for metric in (
            "eval_lpips", "eval_mse", "eval_psnr", "eval_ssim",
            "eval_flow", "eval_dmotion",
        )}
        payload = {"args": BASE, "run": {f"{reward}-gt_only": run}}
        (folder / f"sweep_{reward}_gt_only_s{seed}.json").write_text(json.dumps(payload))
    return str(folder / "*.json")


def test_mrrt_analysis(tmp_path, monkeypatch):
    patterns = {
        "raw": _arm(tmp_path, "raw", [.22, .23, .24]),
        "encoder_rc": _arm(tmp_path, "encoder_rc", [.21, .22, .23]),
        "mrrt": _arm(tmp_path, "mrrt", [.19, .20, .21]),
        "random": _arm(tmp_path, "random", [.23, .24, .25]),
    }
    output = tmp_path / "report.json"
    argv = ["analyze_mrrt_training.py"]
    for arm, pattern in patterns.items():
        argv.extend([f"--{arm}", pattern])
    argv.extend(["--expected_n", "3", "--out", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    report = json.loads(output.read_text())
    assert report["metrics"]["eval_lpips"]["mrrt_minus_encoder_rc"]["mean"] < 0
