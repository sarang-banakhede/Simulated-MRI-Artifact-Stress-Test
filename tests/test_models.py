import pytest

from simulated_mri_artifact_stress_test.models import build_model

PARAMS = {"dynunet": 31181697, "unetr": 93029585, "swinunetr": 62186659}


@pytest.mark.parametrize("name", list(PARAMS))
def test_parameter_counts_match_the_trained_models(cfg, name):
    net = build_model(cfg, name)
    assert sum(p.numel() for p in net.parameters()) == PARAMS[name]
