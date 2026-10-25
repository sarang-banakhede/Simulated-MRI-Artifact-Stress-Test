import numpy as np
import pytest

from simulated_mri_artifact_stress_test.evaluation.metrics import case_metrics


def ball(shape, center, r):
    g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return sum((a - c) ** 2 for a, c in zip(g, center)) <= r**2


def test_perfect_prediction():
    gt = ball((40, 40, 40), (20, 20, 20), 8)
    m = case_metrics(gt, gt, (1, 1, 1), 100.0, 1.0)
    assert m["dsc"] == 1.0 and m["nsd"] == 1.0 and m["hd95"] == 0.0 and m["empty_pred"] == 0


def test_shifted_prediction():
    gt = ball((40, 40, 40), (20, 20, 20), 8)
    pred = ball((40, 40, 40), (20, 20, 23), 8)
    m = case_metrics(pred, gt, (1, 1, 1), 100.0, 1.0)
    assert 0.6 < m["dsc"] < 0.9 and m["hd95"] == pytest.approx(3.0, abs=1.0) and m["nsd"] < 0.7


def test_empty_prediction_and_volumes():
    gt = ball((30, 30, 30), (15, 15, 15), 5)
    m = case_metrics(np.zeros_like(gt), gt, (1, 1, 2), 77.0, 1.0)
    assert m["dsc"] == 0.0 and m["empty_pred"] == 1 and m["hd95"] == 77.0
    assert m["gt_ml"] == pytest.approx(gt.sum() * 2 / 1000)
