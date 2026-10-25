from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def case_name(path: str | Path) -> str:
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def list_nii(directory: str | Path) -> list[Path]:
    d = Path(directory)
    files = list(d.glob("*.nii.gz")) + list(d.glob("*.nii"))
    return sorted(p for p in files if not p.name.startswith("."))


def nii_path(data_root: Path, split: str, kind: str, case: str) -> Path:
    for ext in (".nii.gz", ".nii"):
        p = Path(data_root) / split / kind / f"{case}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(Path(data_root) / split / kind / f"{case}.nii.gz")


def load_case(img_path, lab_path=None, flair_channel: int = 0):
    """Return FLAIR (RAS, float32), brain mask, optional binary label, affine and spacing."""
    img_nii = nib.as_closest_canonical(nib.load(str(img_path)))
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    flair = img[..., flair_channel] if img.ndim == 4 else img
    mask = (img != 0).any(-1) if img.ndim == 4 else img != 0
    lab = None
    if lab_path is not None:
        lab = (np.asarray(nib.as_closest_canonical(nib.load(str(lab_path))).dataobj) > 0).astype(np.uint8)
        if lab.shape != flair.shape:
            raise ValueError(f"{img_path}: image {flair.shape} vs label {lab.shape}")
    spacing = tuple(float(z) for z in img_nii.header.get_zooms()[:3])
    return np.ascontiguousarray(flair), mask, lab, img_nii.affine, spacing


def bbox_of(mask: np.ndarray) -> tuple[list[int], list[int]]:
    idx = np.argwhere(mask)
    return idx.min(0).tolist(), (idx.max(0) + 1).tolist()


def to_slices(lo, hi) -> tuple[slice, ...]:
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def preprocess(x: np.ndarray, mask: np.ndarray, sl: tuple[slice, ...]) -> np.ndarray:
    """Crop to the brain bounding box and z-score inside the brain mask; zero outside."""
    xc = x[sl].astype(np.float32, copy=True)
    mc = mask[sl]
    vals = xc[mc]
    mean, std = float(vals.mean()), float(vals.std())
    xc = (xc - mean) / max(std, 1e-8)
    xc[~mc] = 0
    return xc.astype(np.float32)


def label_voxels(path) -> int:
    return int((np.asarray(nib.load(str(path)).dataobj) > 0).sum())
