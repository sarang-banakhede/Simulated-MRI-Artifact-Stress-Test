import numpy as np
import pytest

from simulated_mri_artifact_stress_test.analysis.common import first_cross, holm, rauc_arr
from simulated_mri_artifact_stress_test.analysis.operating import effects_ss, icc21


def test_holm_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_first_cross_interpolates():
    x = np.array([1.0, 0.95, 0.90, 0.85])
    y = np.array([1.0, 0.99, 0.97, 0.90])
    assert first_cross(x, y, 0.98) == pytest.approx(0.925)
    assert np.isnan(first_cross(x, y, 0.5))


def test_icc_perfect_and_independent():
    rng = np.random.default_rng(1)
    v = rng.uniform(10, 100, 200)
    assert icc21(v, v) == pytest.approx(1.0)
    assert abs(icc21(v, rng.permutation(v))) < 0.2


def test_variance_decomposition_sums_to_total():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((6, 3, 3, 4))
    ss = effects_ss(Y)
    assert sum(ss.values()) == pytest.approx(((Y - Y.mean()) ** 2).sum())


def test_rauc_is_one_without_loss():
    arr = np.full((3, 6, 5), 0.8)
    assert rauc_arr(arr, 0.1, 5) == pytest.approx(np.ones((3, 5)))
