from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from simulated_mri_artifact_stress_test.artifacts.calibration import load_degradation_config
from simulated_mri_artifact_stress_test.artifacts.simulate import ArtifactSimulator
from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.data.nifti import bbox_of, case_name, list_nii, load_case, preprocess, to_slices
from simulated_mri_artifact_stress_test.utils import log


def choose_cases(cfg: Config, n: int, seed: int) -> tuple[list[str], dict[str, float]]:
    """n random test cases spread over the four tumor-volume quartiles."""
    data = cfg.paths.data / "test"
    cases = sorted(case_name(p) for p in list_nii(data / "images"))
    vol = {}
    for c in cases:
        lab = nib.load(str(data / "labels" / f"{c}.nii.gz"))
        vol[c] = float((np.asarray(lab.dataobj) > 0).sum()) * float(np.prod(lab.header.get_zooms()[:3])) / 1000.0
    order = sorted(cases, key=lambda c: vol[c])
    quart = [order[i * len(order) // 4 : (i + 1) * len(order) // 4] for i in range(4)]
    base, extra = divmod(n, 4)
    counts = [base + (1 if i < extra else 0) for i in range(4)]
    rng = random.Random(seed)
    chosen = []
    for q, k in zip(quart, counts):
        chosen += rng.sample(q, k)
    return sorted(chosen), vol


def _save(arr: np.ndarray, affine, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def build_dataset(cfg: Config) -> list[dict]:
    """Clean and degraded FLAIR volumes (frozen parameters, same code as the inference) for the radiologist."""
    rs, out = cfg.reader_study, cfg.paths.reader
    out.mkdir(parents=True, exist_ok=True)
    sim = ArtifactSimulator.from_config(cfg)
    deg = load_degradation_config(cfg.paths.degradation_config)
    levels = [int(l) for l in rs.levels]
    cases, vol = choose_cases(cfg, rs.n_cases, rs.seed)
    log.info("reader study: %d cases %s", len(cases), cases)
    ref = {}
    if cfg.paths.ssim_test.exists():
        s = pd.read_csv(cfg.paths.ssim_test)
        ref = {(r.case_id, r.artifact, int(r.level)): float(r.ssim) for r in s.itertuples()}
    meta, max_diff = [], 0.0
    for ci, case in enumerate(cases, 1):
        flair, mask, _, affine, spacing = load_case(
            cfg.paths.data / "test" / "images" / f"{case}.nii.gz", flair_channel=cfg.data.flair_channel
        )
        sl = to_slices(*bbox_of(mask))
        clean_pp = preprocess(flair, mask, sl)
        _save(np.where(mask, flair, 0.0), affine, out / case / "clean.nii.gz")
        meta.append(
            dict(
                case=case,
                artifact="clean",
                level=0,
                param=0.0,
                ssim=1.0,
                tumor_ml=round(vol[case], 1),
                file=f"{case}/clean.nii.gz",
            )
        )
        for a in cfg.artifacts:
            shp = sim.pattern(case, a, flair.shape)
            for l in levels:
                x = np.where(mask, sim.degrade(flair, a, deg["params"][a][l], shp, mask, spacing), 0.0).astype(
                    np.float32
                )
                ssim = sim.ssim(clean_pp, preprocess(x, mask, sl), mask[sl])
                if (case, a, l) in ref:
                    max_diff = max(max_diff, abs(ssim - ref[(case, a, l)]))
                _save(x, affine, out / case / a / f"L{l}.nii.gz")
                meta.append(
                    dict(
                        case=case,
                        artifact=a,
                        level=l,
                        param=deg["params"][a][l],
                        ssim=round(ssim, 4),
                        tumor_ml=round(vol[case], 1),
                        file=f"{case}/{a}/L{l}.nii.gz",
                    )
                )
        log.info("  [%d/%d] %s", ci, len(cases), case)
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    json.dump(
        {
            "seed": rs.seed,
            "n_cases": len(cases),
            "cases": cases,
            "levels": levels,
            "levels_ssim_target": {str(l): cfg.target_ssim(l) for l in levels},
            "parameters": {a: {str(l): deg["params"][a][l] for l in levels} for a in cfg.artifacts},
            "selection": "random within the four tumor-volume quartiles of the test cases",
        },
        open(out / "selection.json", "w"),
        indent=2,
    )
    if ref:
        log.info("max |SSIM here - SSIM of the inference run| = %.2e", max_diff)
    return meta


def make_blinded(cfg: Config, meta: list[dict]) -> None:
    """Random IDs, shuffled so that no two volumes of the same case are adjacent, two sittings."""
    out = cfg.paths.reader
    rng = random.Random(cfg.reader_study.seed + 1)
    pools: dict[str, list] = {}
    for m in meta:
        pools.setdefault(m["case"], []).append(m)
    for p in pools.values():
        rng.shuffle(p)
    items, prev = [], None
    while any(pools.values()):
        avail = [p for c, p in pools.items() if c != prev and p]
        left = max(len(p) for p in avail) if avail else 0
        cand = [c for c, p in pools.items() if c != prev and len(p) == left and p] or [c for c, p in pools.items() if p]
        prev = rng.choice(cand)
        items.append(pools[prev].pop())
    bdir = out / "blinded"
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    half = (len(items) + 1) // 2
    rows = []
    for i, m in enumerate(items, 1):
        vid = f"V{i:03d}"
        nib.save(nib.load(str(out / m["file"])), str(bdir / f"{vid}.nii.gz"))
        rows.append(
            dict(
                id=vid,
                sitting=1 if i <= half else 2,
                case=m["case"],
                artifact=m["artifact"],
                level=m["level"],
                ssim=m["ssim"],
            )
        )
    with open(out / "blinded_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "blinded_answers_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "sitting",
                "Q1_problem_seen (no / yes mild / yes obvious)",
                "Q2_looks_like_real_artifact (yes / somewhat / no; only if Q1 is yes)",
                "Q3_decision (accept / rescan)",
                "comment",
            ]
        )
        for r in rows:
            w.writerow([r["id"], r["sitting"], "", "", "", ""])
    log.info("blinded copy: %d volumes in %s", len(rows), bdir)


def build_slice_pdf(cfg: Config) -> Path:
    """One A4 page per case: clean slice plus every artifact at three levels, on the slice with the largest change."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root = cfg.paths.reader
    sel = json.load(open(root / "selection.json"))
    levels = [int(l) for l in sel["levels"]]
    names = {"motion": "Motion", "noise": "Rician noise", "bias": "Bias field"}
    grades = ["mild", "moderate", "severe"]
    out = root / "radiology_slices.pdf"
    with PdfPages(out) as pdf:
        for case in sel["cases"]:
            clean = nib.load(str(root / case / "clean.nii.gz")).get_fdata()
            deg = {
                (a, l): nib.load(str(root / case / a / f"L{l}.nii.gz")).get_fdata()
                for a in cfg.artifacts
                for l in levels
            }
            m = clean > 0
            zs = np.where(m.any((0, 1)))[0]
            lo, hi = zs.min(), zs.max()
            span = hi - lo
            cand = range(int(lo + 0.3 * span), int(lo + 0.7 * span) + 1)
            best = max(
                cand,
                key=lambda z: (
                    sum(np.abs(v[:, :, z] - clean[:, :, z])[m[:, :, z]].sum() for v in deg.values())
                    / max(m[:, :, z].sum(), 1)
                ),
            )
            vmax = np.percentile(clean[m], 99.5)
            fig = plt.figure(figsize=(8.27, 11.69))

            def panel(r, col, img, title):
                ax = fig.add_axes([0.13 + col * 0.285, 0.74 - r * 0.2, 0.27, 0.17])
                ax.imshow(np.rot90(img[:, :, best]), cmap="gray", vmin=0, vmax=vmax)
                ax.axis("off")
                ax.set_title(title, fontsize=9)

            panel(0, 1, clean, "Original (clean)")
            for r, a in enumerate(cfg.artifacts, 1):
                for k, l in enumerate(levels):
                    panel(r, k, deg[(a, l)], f"{names.get(a, a)}, Level {l} ({grades[k % 3]})")
            pdf.savefig(fig)
            plt.close(fig)
    log.info("slice overview written to %s", out)
    return out


def run(cfg: Config, blind_only: bool = False) -> None:
    if blind_only:
        with open(cfg.paths.reader / "metadata.csv") as f:
            meta = [dict(r, level=int(r["level"]), ssim=float(r["ssim"])) for r in csv.DictReader(f)]
    else:
        meta = build_dataset(cfg)
    if cfg.reader_study.blinded:
        make_blinded(cfg, meta)
    build_slice_pdf(cfg)
