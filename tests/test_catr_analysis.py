import json
import subprocess
import sys
from pathlib import Path


def _write(path, reward, credit, seeds, lpips, lpips_last, mse):
    run = {
        "eval_lpips": [lpips],
        "eval_lpips_last": [lpips_last],
        "eval_mse": [mse],
        "eval_psnr": [30.0],
        "eval_ssim": [0.95],
        "eval_latent_rms": [0.05],
    }
    path.write_text(json.dumps({
        "args": {"rewards": reward, "credits": credit, "seeds": str(seeds)},
        "run": run,
    }))


def test_catr_analysis_accepts_noninferior_full_and_better_terminal(tmp_path):
    for seed in range(3):
        _write(tmp_path / f"sweep_raw_seq_s{seed}.json", "raw", "seq", "0,1,2", 0.0200, 0.0400, 0.0020)
        _write(
            tmp_path / f"sweep_catr_adaptive_s{seed}.json", "catr", "adaptive", "0,1,2",
            0.02005, 0.0390, 0.00201,
        )
    output = tmp_path / "report.json"
    subprocess.run([
        sys.executable, "scripts/analyze_catr.py",
        "--catr", str(tmp_path / "sweep_catr_adaptive_s*.json"),
        "--seq_raw", str(tmp_path / "sweep_raw_seq_s*.json"),
        "--bootstrap", "100", "--out", str(output),
    ], check=True)
    assert json.loads(output.read_text())["verdict"] == "PROVISIONAL-GREEN"
