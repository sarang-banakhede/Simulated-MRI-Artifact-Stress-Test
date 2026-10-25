import numpy as np
import pytest

from simulated_mri_artifact_stress_test.artifacts.simulate import ArtifactSimulator
from simulated_mri_artifact_stress_test.data.nifti import bbox_of, preprocess, to_slices


@pytest.fixture(scope="module")
def sim(cfg):
    return ArtifactSimulator.from_config(cfg)


@pytest.fixture(scope="module")
def volume():
    rng = np.random.default_rng(0)
    x = np.zeros((40, 44, 36), np.float32)
    z, y, w = np.meshgrid(*[np.linspace(-1, 1, n) for n in x.shape], indexing="ij")
    brain = (z**2 + y**2 + w**2) < 0.7
    x[brain] = 100 + 20 * rng.standard_normal(brain.sum())
    x[brain & (z > 0.2)] += 60
    return x, brain


def test_pattern_is_deterministic_per_case(sim, volume):
    x, _ = volume
    a = sim.pattern("BRATS_001", "noise", x.shape)
    b = sim.pattern("BRATS_001", "noise", x.shape)
    c = sim.pattern("BRATS_002", "noise", x.shape)
    assert np.array_equal(a["n1"], b["n1"]) and not np.array_equal(a["n1"], c["n1"])


@pytest.mark.parametrize("artifact,strength", [("motion", 2.0), ("noise", 8.0), ("bias", 0.2)])
def test_degradation_reduces_ssim_monotonically(sim, volume, artifact, strength):
    x, brain = volume
    sl = to_slices(*bbox_of(brain))
    clean = preprocess(x, brain, sl)
    shp = sim.pattern("BRATS_010", artifact, x.shape)
    sims = []
    for s in (0.0, strength, 2 * strength):
        d = sim.degrade(x, artifact, s, shp, brain, (1.0, 1.0, 1.0))
        sims.append(sim.ssim(clean, preprocess(d, brain, sl), brain[sl]))
    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert sims[0] > sims[1] > sims[2]


def test_zero_strength_is_identity(sim, volume):
    x, brain = volume
    for a in ("motion", "noise", "bias"):
        assert sim.degrade(x, a, 0.0, sim.pattern("c", a, x.shape), brain, (1, 1, 1)) is x
