import numpy as np

from dor.rank_stats import average_ranks, pair_flip_fraction, rowwise_spearman


def test_average_ranks_is_tie_aware():
    ranks = average_ranks(np.asarray([[1.0, 1.0, 3.0]]))
    np.testing.assert_allclose(ranks, [[0.5, 0.5, 2.0]])


def test_pair_flip_fraction_identifies_reversal():
    reference = np.asarray([[0.0, 1.0, 2.0]])
    assert pair_flip_fraction(reference, reference)[0] == 0.0
    assert pair_flip_fraction(reference[:, ::-1], reference)[0] == 1.0
    assert rowwise_spearman(reference, reference)[0] == 1.0
