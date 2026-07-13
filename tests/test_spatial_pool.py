import pytest
import torch

from dor.spatial_pool import RATIO_CAP, block_pool, floor_weights, weighted_pool


def test_block_pool_preserves_mean_exactly():
    m = torch.rand(256, 320, dtype=torch.float64)
    assert block_pool(m).mean() == pytest.approx(m.mean().item(), abs=1e-12)
    k = torch.rand(4, 256, 320, dtype=torch.float64)
    torch.testing.assert_close(block_pool(k).mean(dim=(1, 2)), k.mean(dim=(1, 2)))


def test_flat_floor_degenerates_to_uniform_mean():
    phi = torch.full((32, 40), 0.05, dtype=torch.float64)
    for scheme in ("iv", "im"):
        w = floor_weights(phi, 0.5, scheme)
        torch.testing.assert_close(w, torch.full_like(w, 1.0 / w.numel()))
        maps = torch.rand(3, 32, 40, dtype=torch.float64)
        torch.testing.assert_close(weighted_pool(maps, w), maps.mean(dim=(1, 2)))


def test_weight_ratio_is_capped_and_normalized():
    phi = torch.zeros(32, 40, dtype=torch.float64)
    phi[:16] = 100.0  # extreme dynamic range
    for scheme in ("iv", "im"):
        w = floor_weights(phi, 0.5, scheme)
        assert w.sum() == pytest.approx(1.0)
        assert (w.max() / w.min()).item() <= RATIO_CAP + 1e-9
        assert torch.all(w > 0)


def test_high_floor_regions_are_downweighted():
    phi = torch.full((32, 40), 0.01, dtype=torch.float64)
    phi[:, 20:] = 0.2  # right half: high reconstruction floor
    err = torch.zeros(1, 32, 40, dtype=torch.float64)
    err[0, :, 20:] = 1.0  # candidate error concentrated where the floor is high
    w = floor_weights(phi, 0.5, "iv")
    assert weighted_pool(err, w)[0] < err.mean().item()  # discounted vs uniform pooling
