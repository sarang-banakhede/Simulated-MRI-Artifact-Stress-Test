import json
from pathlib import Path

import nibabel as nib
import numpy as np

from simulated_mri_artifact_stress_test.config import load_config
from simulated_mri_artifact_stress_test.data.splits import get_splits, make_splits

ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(root: Path, n_train=24, n_test=6):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        for kind in ("images", "labels"):
            (root / split / kind).mkdir(parents=True)
        for i in range(n):
            name = f"BRATS_{split[:2]}{i:03d}.nii.gz"
            img = rng.uniform(1, 2, (8, 8, 8, 4)).astype(np.float32)
            lab = np.zeros((8, 8, 8), np.uint8)
            lab[: 2 + i % 5, :3, :3] = 1
            nib.save(nib.Nifti1Image(img, np.eye(4)), str(root / split / "images" / name))
            nib.save(nib.Nifti1Image(lab, np.eye(4)), str(root / split / "labels" / name))


def test_split_is_patient_level_and_reproducible(tmp_path):
    _write_dataset(tmp_path / "data")
    over = [
        f"paths.data_root={tmp_path / 'data'}",
        f"paths.output_root={tmp_path / 'out'}",
        "data.split.val_cases=6",
        "data.split.probe_cases=3",
        "data.split.expected_sha256_prefix=null",
        "analysis.workers=1",
    ]
    cfg = load_config(ROOT / "configs" / "config.yaml", over)
    a = make_splits(cfg)
    assert a == make_splits(cfg)
    assert len(a["val"]) == 6 and not set(a["train"]) & set(a["val"]) and len(a["test"]) == 6
    assert set(a["probe"]) <= set(a["val"])
    cfg.paths.out.mkdir(parents=True)
    assert get_splits(cfg)["val"] == a["val"]
    assert json.load(open(cfg.paths.splits))["train"] == a["train"]
