import numpy as np

from scripts.gate_rc_safe_variants import _variants, _zscore


def test_snr_shrink_stays_between_raw_and_rc():
    raw = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    rc = np.array([[0.5, 1.5, 3.0], [1.0, 0.5, 2.0]])
    variants, _, weight = _variants(raw, rc)
    expected = (1.0 - weight[:, None]) * raw + weight[:, None] * rc
    np.testing.assert_allclose(variants["snr_shrink"], expected)
    assert np.all((weight >= 0.0) & (weight < 1.0))


def test_orthogonal_rc_is_group_normalized_and_preserves_raw_component():
    raw = np.array([[0.0, 1.0, 3.0, 2.0]])
    rc = np.array([[1.0, 3.0, 0.0, 2.0]])
    variants, _, _ = _variants(raw, rc)
    safe = variants["orthogonal_rc"]
    primary = _zscore(raw)
    np.testing.assert_allclose(safe.mean(axis=1), 0.0, atol=1e-8)
    np.testing.assert_allclose(safe.std(axis=1), 1.0, atol=1e-7)
    assert float((safe * primary).sum()) > 0.0
