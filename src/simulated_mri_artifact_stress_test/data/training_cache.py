from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.data.nifti import case_name, list_nii
from simulated_mri_artifact_stress_test.utils import atomic_json, log


def _preprocess_case(args: tuple[str, str, str, int]) -> str:
    img_path, lab_path, out_dir, channel = args
    out = Path(out_dir)
    case = case_name(img_path)
    if (out / f"{case}_properties.json").exists():
        return case
    img_nii = nib.as_closest_canonical(nib.load(img_path))
    lab_nii = nib.as_closest_canonical(nib.load(lab_path))
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    lab = (np.asarray(lab_nii.dataobj) > 0).astype(np.uint8)
    flair = img[..., channel] if img.ndim == 4 else img
    mask = (img != 0).any(-1) if img.ndim == 4 else img != 0
    if flair.shape != lab.shape:
        raise ValueError(f"{case}: image {flair.shape} vs label {lab.shape}")
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    flair_c, lab_c, mask_c = flair[sl], lab[sl], mask[sl]
    vals = flair_c[mask_c]
    mean, std = float(vals.mean()), float(vals.std())
    flair_c = ((flair_c - mean) / max(std, 1e-8)).astype(np.float32)
    flair_c[~mask_c] = 0
    np.save(out / f"{case}_img.npy", flair_c[None])
    np.save(out / f"{case}_seg.npy", lab_c[None])
    np.save(out / f"{case}_mask.npy", mask_c[None].astype(np.uint8))
    props = {
        "case": case,
        "original_shape": list(flair.shape),
        "cropped_shape": list(flair_c.shape),
        "bbox_start": lo.tolist(),
        "bbox_end": hi.tolist(),
        "spacing_mm": [float(z) for z in img_nii.header.get_zooms()[:3]],
        "norm_mean": mean,
        "norm_std": std,
        "tumour_voxels": int(lab.sum()),
        "tumour_outside_mask_voxels": int((lab.astype(bool) & ~mask).sum()),
    }
    atomic_json(props, out / f"{case}_properties.json")
    return case


def build_training_cache(cfg: Config) -> Path:
    """Crop, z-score and store every training case as .npy so training does not decode NIfTI files."""
    src, cache = cfg.paths.data / "train", cfg.paths.cache / "preprocessed"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for img in list_nii(src / "images"):
        lab = src / "labels" / img.name
        if not lab.exists():
            raise FileNotFoundError(lab)
        jobs.append((str(img), str(lab), str(cache), cfg.data.flair_channel))
    if not jobs:
        raise FileNotFoundError(f"no images in {src / 'images'}")
    todo = [j for j in jobs if not (cache / f"{case_name(j[0])}_properties.json").exists()]
    log.info("training cache: %d cases, %d to preprocess (%s)", len(jobs), len(todo), cache)
    if todo:
        with ProcessPoolExecutor(max(1, cfg.workers(cfg.training.num_workers))) as ex:
            for _ in tqdm(ex.map(_preprocess_case, todo, chunksize=2), total=len(todo), desc="preprocess"):
                pass
    return cache
