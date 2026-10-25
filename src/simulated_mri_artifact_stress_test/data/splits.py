from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.data.nifti import case_name, label_voxels, list_nii, nii_path
from simulated_mri_artifact_stress_test.utils import atomic_json, log, sha256_file


def make_splits(cfg: Config) -> dict:
    """Patient-level split of the labeled cases, stratified by tumor-volume quartile."""
    data, sc = cfg.paths.data, cfg.data.split
    train_all = sorted(case_name(p) for p in list_nii(data / "train" / "labels"))
    test = sorted(case_name(p) for p in list_nii(data / "test" / "labels"))
    paths = [nii_path(data, "train", "labels", c) for c in train_all]
    with ProcessPoolExecutor(cfg.workers(cfg.analysis.workers)) as ex:
        vols = np.array(list(ex.map(label_voxels, paths, chunksize=4)))
    pos = vols[vols > 0]
    strata = np.where(vols == 0, -1, np.digitize(vols, np.percentile(pos, sc.volume_quantiles) if len(pos) else []))
    rng = np.random.default_rng(sc.seed)
    frac = sc.val_cases / len(train_all)
    val: list[str] = []
    keys = sorted(set(strata.tolist()))
    for i, s in enumerate(keys):
        idx = rng.permutation(np.where(strata == s)[0])
        k = round(len(idx) * frac) if i < len(keys) - 1 else sc.val_cases - len(val)
        k = int(max(0, min(k, len(idx))))
        val += [train_all[j] for j in idx[:k]]
    val = sorted(val)
    train = sorted(set(train_all) - set(val))
    probe = sorted(rng.choice(val, size=min(sc.probe_cases, len(val)), replace=False).tolist())
    return {
        "split_seed": sc.seed,
        "stratified_by": "tumour volume quartile",
        "num_train": len(train),
        "num_val": len(val),
        "num_probe": len(probe),
        "num_test": len(test),
        "train": train,
        "val": val,
        "probe": probe,
        "test": test,
    }


def write_case_split(cfg: Config, splits: dict) -> None:
    """One row per labeled case: the split it belongs to (train, val or test) and whether it is a probe case."""
    role = {c: "train" for c in splits["train"]} | {c: "val" for c in splits["val"]} | {c: "test" for c in splits["test"]}
    probe = set(splits.get("probe", []))
    df = pd.DataFrame({"case_id": sorted(role), "split": [role[c] for c in sorted(role)],
                       "probe": [c in probe for c in sorted(role)]})
    df.to_csv(cfg.paths.case_split, index=False)


def get_splits(cfg: Config) -> dict:
    path = cfg.paths.splits
    if not path.exists():
        atomic_json(make_splits(cfg), path)
        log.info("splits written to %s", path)
    sha = sha256_file(path)
    expected = cfg.data.split.expected_sha256_prefix
    if expected and not sha.startswith(str(expected)):
        raise SystemExit(
            f"splits.json sha256 {sha[:12]} differs from the published split {expected}; "
            "set data.split.expected_sha256_prefix to null to use a new split"
        )
    splits = json.load(open(path))
    test_dir = cfg.paths.data / "test" / "images"
    if test_dir.exists() and sorted(splits["test"]) != sorted(case_name(p) for p in list_nii(test_dir)):
        raise SystemExit("splits.json test list does not match test/images")
    if not cfg.paths.case_split.exists():
        write_case_split(cfg, splits)
    return splits
