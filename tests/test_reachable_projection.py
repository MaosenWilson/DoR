import numpy as np
import torch

from dor.reachable_projection import (
    fsq_basis,
    hamming_fraction,
    indices_to_levels,
    legal_adjacent_neighbors,
    levels_to_indices,
    load_mrrt_cache,
    matched_random_legal_target,
)


def test_fsq_index_roundtrip():
    levels = [3, 2]
    coordinates = torch.tensor([[[0, 0], [2, 1]], [[1, 0], [0, 1]]])
    indices = levels_to_indices(coordinates, levels)
    torch.testing.assert_close(indices_to_levels(indices, levels), coordinates)
    torch.testing.assert_close(fsq_basis(levels), torch.tensor([1, 3]))


def test_neighbors_are_legal_single_level_moves():
    levels = [3, 2]
    indices = levels_to_indices(torch.tensor([[[1, 0], [2, 1]]]), levels)
    neighbors = legal_adjacent_neighbors(indices, levels, positions=[0])
    assert len(neighbors) == 3
    base_coordinates = indices_to_levels(indices, levels)
    for neighbor in neighbors:
        coordinates = indices_to_levels(neighbor, levels)
        assert int((coordinates != base_coordinates).sum()) == 1
        assert hamming_fraction(indices, neighbor) == 0.5


def test_matched_random_target_has_exact_cell_hamming_budget():
    levels = [3, 2]
    coordinates = torch.tensor([
        [[1, 0], [2, 1]],
        [[0, 0], [1, 1]],
    ])
    base = levels_to_indices(coordinates, levels)
    random_target = matched_random_legal_target(
        base, levels, candidate_positions=[0, 1, 2, 3], changed_cells=2, seed=9
    )
    assert int((base != random_target).sum()) == 2
    decoded = indices_to_levels(random_target, levels)
    assert int((decoded != coordinates).sum()) == 2


def test_mrrt_cache_loader_keys_targets(tmp_path):
    path = tmp_path / "targets.npz"
    np.savez_compressed(
        path,
        episodes=np.asarray(["ep0.npz", "ep1.npz"]),
        starts=np.asarray([5, 10]),
        mrrt=np.asarray([[[1]], [[2]]]),
        mrrt_random=np.asarray([[[3]], [[4]]]),
    )
    cache = load_mrrt_cache(path)
    assert cache[("ep0.npz", 5)]["mrrt"].item() == 1
    assert cache[("ep1.npz", 10)]["mrrt_random"].item() == 4
