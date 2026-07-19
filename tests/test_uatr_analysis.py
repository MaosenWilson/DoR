import json
import subprocess
import sys


def _args(reward, credit):
    return {
        "checkpoint": "checkpoint",
        "upstream": "upstream",
        "train_manifest": "train.json",
        "eval_manifest": "eval.json",
        "rewards": reward,
        "credits": credit,
        "horizon": 8,
        "K": 16,
        "eval_K": 8,
        "eval_draws": 2,
        "eval_aggregation": "episode_macro",
        "steps": 20,
        "batch_windows": 2,
        "lr": 1e-6,
        "kl": 0.001,
        "kl_type": "low_var_kl",
        "gamma": 0.95,
        "data_seed": 7304,
        "eval_seed": 9917,
    }


def _write(path, reward, credit, metrics, *, projected=None):
    run = {
        "step": [20],
        "protocol": {
            "adaptive_coefficients": [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "adaptive_config_sha256": "adaptive-sha",
            "train_manifest_sha256": "train-sha",
            "eval_manifest_sha256": "eval-sha",
            "context_schedule_sha256": "schedule-sha",
            "train_episodes": ["train-0", "train-1"],
            "eval_episodes": ["eval-0", "eval-1"],
        },
    }
    for name, value in metrics.items():
        run[f"eval_{name}"] = [value]
    if projected is not None:
        run["train_projected_progress_ratio"] = projected
    path.write_text(json.dumps({"args": _args(reward, credit), "run": run}))


def test_uatr_analysis_requires_final_three_arm_gain(tmp_path):
    for seed in range(3):
        raw = {
            "lpips": 0.0200,
            "lpips_last": 0.0400,
            "mse": 0.0020,
            "psnr": 29.0,
            "ssim": 0.95,
            "latent_rms": 0.05,
        }
        aligned = {
            "lpips": 0.0197,
            "lpips_last": 0.0385,
            "mse": 0.00198,
            "psnr": 29.1,
            "ssim": 0.951,
            "latent_rms": 0.049,
        }
        shuffled = {
            "lpips": 0.0199,
            "lpips_last": 0.0395,
            "mse": 0.00199,
            "psnr": 29.05,
            "ssim": 0.9505,
            "latent_rms": 0.0495,
        }
        _write(tmp_path / f"sweep_raw_seq_s{seed}.json", "raw", "seq", raw)
        _write(
            tmp_path / f"sweep_uatr_adaptive_s{seed}.json",
            "uatr,uatr_shuffled", "adaptive", aligned,
            projected=[1.0, 1.02],
        )
        _write(
            tmp_path / f"sweep_uatr_shuffled_adaptive_s{seed}.json",
            "uatr,uatr_shuffled", "adaptive", shuffled,
            projected=[1.0, 1.01],
        )

    output = tmp_path / "gate.json"
    subprocess.run([
        sys.executable,
        "scripts/analyze_uatr_vp2.py",
        "--raw", str(tmp_path / "sweep_raw_seq_s*.json"),
        "--uatr", str(tmp_path / "sweep_uatr_adaptive_s*.json"),
        "--shuffled", str(tmp_path / "sweep_uatr_shuffled_adaptive_s*.json"),
        "--bootstrap", "100",
        "--out", str(output),
    ], check=True)

    report = json.loads(output.read_text())
    assert report["verdict"] == "PROVISIONAL-GREEN"
    assert report["active_blocks"] == 2
    assert report["uatr_minus_seq_raw"]["eval_lpips"]["wins"] == 3
    assert report["uatr_minus_shuffled"]["eval_lpips_last"]["wins"] == 3
