"""Whole pipeline on a tiny synthetic dataset (CPU, about a minute). Run with: pytest -m slow"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from simulated_mri_artifact_stress_test.cli import main

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def _blob(shape, center, radius):
    g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return sum((a - c) ** 2 for a, c in zip(g, center)) <= radius**2


def _dataset(root: Path, n_train=14, n_test=6, size=48):
    rng = np.random.default_rng(0)
    shape = (size,) * 3
    for split, n in (("train", n_train), ("test", n_test)):
        for kind in ("images", "labels"):
            (root / split / kind).mkdir(parents=True)
        for i in range(n):
            brain = _blob(shape, (size / 2,) * 3, size * 0.42)
            c = rng.uniform(size * 0.35, size * 0.65, 3)
            tumor = _blob(shape, c, rng.uniform(4, 8)) & brain
            flair = np.where(brain, 100 + 10 * rng.standard_normal(shape), 0) + 80 * tumor
            img = np.stack([flair, flair * 0.9, flair * 1.1, flair * 0.8], -1).astype(np.float32)
            name = f"BRATS_{split[:2]}{i:03d}.nii.gz"
            nib.save(nib.Nifti1Image(img, np.eye(4)), str(root / split / "images" / name))
            nib.save(nib.Nifti1Image(tumor.astype(np.uint8), np.eye(4)), str(root / split / "labels" / name))


def test_pipeline(tmp_path):
    _dataset(tmp_path / "data")
    out = tmp_path / "out"
    overrides = [
        "data.split.expected_sha256_prefix=null",
        "data.split.val_cases=4",
        "data.split.probe_cases=2",
        "experiment.models=[dynunet, unetr, swinunetr]",
        "experiment.seeds=[0]",
        "training.patch_size=[64,64,64]",
        "training.epochs=2",
        "training.iters_per_epoch=2",
        "training.val_every_epochs=1",
        "training.sw_batch_size=1",
        "training.num_workers=0",
        "models.dynunet.batch_size=1",
        "models.unetr.batch_size=1",
        "models.swinunetr.batch_size=1",
        "models.swinunetr.kwargs.use_checkpoint=false",
        "artifacts.grid_points=6",
        "artifacts.frozen_rel_tol=1.0e+9",
        "inference.sw_batch_size=1",
        "inference.loader_workers=0",
        "inference.metric_workers=1",
        "analysis.workers=2",
        "analysis.bootstrap.h2=60",
        "analysis.bootstrap.h1_rho=40",
        "analysis.bootstrap.curves=40",
        "analysis.bootstrap.pooled=40",
        "analysis.bootstrap.envelope=40",
        "analysis.bootstrap.envelope_passrate=20",
        "analysis.bootstrap.volumetry_icc=20",
        "analysis.bootstrap.sensitivity=40",
        "analysis.bootstrap.sensitivity_rho=40",
        "analysis.bootstrap.post_masks=40",
        "reader_study.n_cases=4",
    ]
    args = [
        "-c",
        str(ROOT / "configs" / "config.yaml"),
        "--set",
        f"paths.data_root={tmp_path / 'data'}",
        "--set",
        f"paths.output_root={out}",
    ]
    for o in overrides:
        args += ["--set", o]

    def run(*cmd):
        with pytest.raises(SystemExit) as e:
            main(list(cmd) + args)
        assert e.value.code == 0

    run("prepare")
    run("train")
    run("status")
    run("calibrate")
    run("infer")
    results = sorted((out / "inference" / "results").glob("*/seed0.csv"))
    assert len(results) == 3
    df = pd.read_csv(results[0])
    assert len(df) == 6 * 31 and df.dsc.between(0, 1).all()
    assert len(list((out / "inference" / "predictions").rglob("*.nii.gz"))) == 3 * 6 * 31
    run("analyze")
    assert (out / "analysis" / "tables" / "h2.csv").exists()
    run("reader-study")
    assert (out / "reader_study" / "radiology_slices.pdf").exists()
    assert (out / "case_split.csv").exists()
