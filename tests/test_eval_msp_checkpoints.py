import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_msp_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("eval_msp_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _episode(path, length):
    np.savez(
        path,
        image=np.zeros((length, 2, 2, 3), dtype=np.uint8),
        action=np.zeros((length, 13), dtype=np.float32),
    )


def test_episode_disjoint_windows_exclude_training_episodes(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"episode_{index}.npz"
        _episode(path, 24)
        paths.append(str(path))
    train, evaluation = MODULE.build_episode_disjoint_windows(
        paths, T=8, train_windows=2, split_seed=1, stride=8
    )
    assert train
    assert evaluation
    assert {path for path, _ in train}.isdisjoint(
        {path for path, _ in evaluation}
    )


def test_aggregate_rows_reports_window_and_episode_macro():
    rows = [
        {
            "episode": "a",
            "lpips": 1.0,
            "lpips_last": 2.0,
            "mse": 3.0,
            "psnr": 4.0,
            "ssim": 5.0,
        },
        {
            "episode": "a",
            "lpips": 3.0,
            "lpips_last": 4.0,
            "mse": 5.0,
            "psnr": 6.0,
            "ssim": 7.0,
        },
        {
            "episode": "b",
            "lpips": 9.0,
            "lpips_last": 10.0,
            "mse": 11.0,
            "psnr": 12.0,
            "ssim": 13.0,
        },
    ]
    summary = MODULE.aggregate_rows(rows)
    assert np.isclose(summary["window_macro"]["lpips"], 13.0 / 3.0)
    assert np.isclose(summary["episode_macro"]["lpips"], 5.5)


class _FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(use_cache=False)
        self.device = None
        self.training = True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self


def test_load_model_falls_back_for_missing_safetensors_metadata(
    tmp_path, monkeypatch
):
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"placeholder")
    model = _FakeModel()
    loaded = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, torch_dtype):
            raise AttributeError("'NoneType' object has no attribute 'get'")

        @staticmethod
        def from_config(config):
            loaded["config"] = config
            return model

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path):
            return {"path": str(path)}

    def fake_load_model(target, filename, strict, device):
        loaded.update(
            target=target, filename=filename, strict=strict, device=device
        )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoConfig=FakeAutoConfig, AutoModelForCausalLM=FakeAutoModel
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        types.SimpleNamespace(load_model=fake_load_model),
    )
    monkeypatch.setitem(sys.modules, "dor.compat", types.SimpleNamespace())

    result = MODULE._load_model(str(tmp_path), "cuda")

    assert result is model
    assert loaded["target"] is model
    assert loaded["filename"] == str(weight_path)
    assert loaded["strict"] is True
    assert loaded["device"] == "cpu"
    assert model.device == "cuda"
    assert model.config.use_cache is True
    assert model.training is False


def test_load_model_does_not_mask_unrelated_attribute_error(tmp_path, monkeypatch):
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, torch_dtype):
            raise AttributeError("unrelated failure")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoConfig=object, AutoModelForCausalLM=FakeAutoModel
        ),
    )
    monkeypatch.setitem(sys.modules, "dor.compat", types.SimpleNamespace())

    try:
        MODULE._load_model(str(tmp_path), "cuda")
    except AttributeError as error:
        assert str(error) == "unrelated failure"
    else:
        raise AssertionError("unrelated AttributeError was unexpectedly masked")
