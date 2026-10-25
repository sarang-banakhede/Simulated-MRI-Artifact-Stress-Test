from pathlib import Path

import pytest

from simulated_mri_artifact_stress_test.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "config.yaml")
