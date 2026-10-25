from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import fft as sfft
from tqdm.auto import tqdm

from simulated_mri_artifact_stress_test.artifacts.simulate import ArtifactSimulator
from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.data.nifti import bbox_of, load_case, nii_path, preprocess, to_slices
from simulated_mri_artifact_stress_test.data.splits import get_splits
from simulated_mri_artifact_stress_test.utils import atomic_json, log, sha256_file

_STATE: dict = {}


def load_degradation_config(path) -> dict:
    cfg = json.load(open(path))
    cfg["params"] = {a: {int(l): float(v) for l, v in d.items()} for a, d in cfg["params"].items()}
    cfg["capped"] = {a: {int(l): bool(v) for l, v in d.items()} for a, d in cfg["capped"].items()}
    return cfg


def _init(cfg: Config) -> None:
    _STATE["cfg"] = cfg
    _STATE["sim"] = ArtifactSimulator.from_config(cfg)


def _sweep_job(args):
    case, img_path, artifact, grid = args
    cfg, sim = _STATE["cfg"], _STATE["sim"]
    flair, mask, _, _, spacing = load_case(img_path, flair_channel=cfg.data.flair_channel)
    sl = to_slices(*bbox_of(mask))
    mc = mask[sl]
    clean = preprocess(flair, mask, sl)
    shp = sim.pattern(case, artifact, flair.shape)
    return [
        {
            "artifact": artifact,
            "case_id": case,
            "param": float(p),
            "ssim": sim.ssim(clean, preprocess(sim.degrade(flair, artifact, p, shp, mask, spacing), mask, sl), mc),
        }
        for p in grid
    ]


def fit_levels(cfg: Config, sweep: pd.DataFrame):
    a_cfg = cfg.art
    params, capped, curves, monotone = {}, {}, {}, {}
    for a in cfg.artifacts:
        med = sweep[sweep.artifact == a].groupby("param")["ssim"].median().sort_index()
        p = np.concatenate([[0.0], med.index.to_numpy(float)])
        s = np.concatenate([[1.0], med.to_numpy(float)])
        monotone[a] = bool(np.all(np.diff(s) <= 1e-9))
        sm = np.minimum.accumulate(s) - np.arange(len(s)) * 1e-9
        params[a], capped[a] = {}, {}
        for l in cfg.levels:
            t = cfg.target_ssim(l)
            if t < sm[-1]:
                params[a][l], capped[a][l] = float(a_cfg.caps[a]), True
            else:
                params[a][l], capped[a][l] = round(float(np.interp(t, sm[::-1], p[::-1])), 6), False
        curves[a] = {"param": p.tolist(), "median_ssim": s.tolist()}
    return params, capped, curves, monotone


def calibrate(cfg: Config, debug_cases: int | None = None) -> dict:
    """Find, per artifact, the strength whose median validation SSIM hits each target level."""
    path = cfg.paths.degradation_config
    if path.exists():
        log.info("degradation config reused (sha256 %s)", sha256_file(path)[:12])
        return load_degradation_config(path)
    cal = cfg.paths.calibration
    cal.mkdir(parents=True, exist_ok=True)
    splits = get_splits(cfg)
    val = splits["val"][:debug_cases] if debug_cases else splits["val"]
    a_cfg = cfg.art
    sweep_path = cal / "sweep_val.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        log.info("calibration sweep reused (%d rows)", len(sweep))
    else:
        grids = {a: np.linspace(0, a_cfg.caps[a], a_cfg.grid_points + 1)[1:] for a in cfg.artifacts}
        jobs = [
            (c, str(nii_path(cfg.paths.data, "train", "images", c)), a, grids[a]) for c in val for a in cfg.artifacts
        ]
        n = cfg.workers(cfg.analysis.workers)
        log.info(
            "calibration: %d val cases x %d artifacts x %d strengths, %d workers",
            len(val),
            len(cfg.artifacts),
            a_cfg.grid_points,
            n,
        )
        rows = []
        with ProcessPoolExecutor(n, mp_context=mp.get_context("fork"), initializer=_init, initargs=(cfg,)) as ex:
            for r in tqdm(ex.map(_sweep_job, jobs), total=len(jobs), desc="calibration"):
                rows += r
        sweep = pd.DataFrame(rows).sort_values(["artifact", "param", "case_id"])
        sweep.to_csv(sweep_path, index=False)
    params, capped, curves, monotone = fit_levels(cfg, sweep)
    frozen = {a: {int(l): float(v) for l, v in d.items()} for a, d in a_cfg.frozen_primary.items()}
    rel = {
        f"{a}_L{l}": abs(params[a][l] - frozen[a][l]) / frozen[a][l] for a in cfg.artifacts for l in cfg.primary_levels
    }
    worst = max(rel.values())
    log.info("frozen L1-L5 parameters reproduced by this calibration: max relative difference %.2e", worst)
    if worst > a_cfg.frozen_rel_tol and not debug_cases:
        raise SystemExit(
            f"calibration does not reproduce the frozen L1-L5 parameters (max rel. diff {worst:.2e} > "
            f"{a_cfg.frozen_rel_tol}); check the validation cases and package versions"
        )
    for a in cfg.artifacts:
        for l in cfg.primary_levels:
            params[a][l], capped[a][l] = frozen[a][l], False
    out = {
        "artifacts": cfg.artifacts,
        "levels": cfg.levels,
        "primary_levels": cfg.primary_levels,
        "exploratory_levels": [l for l in cfg.levels if l not in cfg.primary_levels],
        "target_ssim": {l: cfg.target_ssim(l) for l in cfg.levels},
        "params": params,
        "capped": capped,
        "caps": dict(a_cfg.caps),
        "grid_n": a_cfg.grid_points,
        "median_curves": curves,
        "raw_curve_monotone": monotone,
        "calibration_set": "val",
        "calibration_cases": len(val),
        "frozen_primary": {"params": frozen, "max_rel_diff_this_calibration": worst},
        "ssim": {
            "space": "preprocessed (clean-mask crop, z-score inside mask, 0 outside)",
            "window": "gaussian",
            "sigma": a_cfg.ssim_sigma,
            "data_range": "clean max-min inside mask",
            "average": "SSIM map mean inside mask",
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_json(out, path)
    for a in cfg.artifacts:
        log.info(
            "  %s: %s | monotone=%s",
            a,
            ", ".join(f"L{l}={params[a][l]:.4g}{' (capped)' if capped[a][l] else ''}" for l in cfg.levels),
            monotone[a],
        )
    return load_degradation_config(path)


def _radial(shape):
    g = np.meshgrid(*[np.fft.fftshift(np.fft.fftfreq(n)) for n in shape], indexing="ij", sparse=True)
    return np.sqrt(sum(v**2 for v in g)) / 0.5


def _band_share(diff, r, bands):
    pw = np.abs(sfft.fftshift(sfft.fftn(diff))) ** 2
    tot = max(float(pw.sum()), 1e-12)
    return [
        100.0 * float(pw[r < bands[0]].sum()) / tot,
        100.0 * float(pw[(r >= bands[0]) & (r < bands[1])].sum()) / tot,
        100.0 * float(pw[r >= bands[1]].sum()) / tot,
    ]


def _footprint_job(args):
    case, img_path, params = args
    cfg, sim = _STATE["cfg"], _STATE["sim"]
    level, bands = cfg.art.footprint.level, cfg.art.footprint.bands
    flair, mask, _, _, spacing = load_case(img_path, flair_channel=cfg.data.flair_channel)
    sl = to_slices(*bbox_of(mask))
    clean = preprocess(flair, mask, sl)
    r = _radial(clean.shape)
    rows = []
    for a in cfg.artifacts:
        d = preprocess(
            sim.degrade(flair, a, params[a][level], sim.pattern(case, a, flair.shape), mask, spacing), mask, sl
        )
        e = _band_share(d - clean, r, bands)
        for b, name in enumerate(("low", "mid", "high")):
            rows.append(
                {"case_id": case, "artifact": a, "level": level, "band": name, "artifact_energy_share_pct": e[b]}
            )
    return rows


def frequency_footprint(cfg: Config, deg_cfg: dict, debug_cases: int | None = None) -> None:
    """Share of artifact energy in the low, mid and high radial frequency band at the footprint level."""
    cal = cfg.paths.calibration
    path = cal / f"footprint_val_L{cfg.art.footprint.level}.csv"
    if path.exists():
        return
    val = get_splits(cfg)["val"]
    val = val[:debug_cases] if debug_cases else val
    jobs = [(c, str(nii_path(cfg.paths.data, "train", "images", c)), deg_cfg["params"]) for c in val]
    rows = []
    n = cfg.workers(cfg.analysis.workers)
    with ProcessPoolExecutor(n, mp_context=mp.get_context("fork"), initializer=_init, initargs=(cfg,)) as ex:
        for r in tqdm(ex.map(_footprint_job, jobs), total=len(jobs), desc="footprint"):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    summ = (
        df.groupby(["artifact", "band"])["artifact_energy_share_pct"]
        .agg(median="median", q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75))
        .reset_index()
    )
    summ.to_csv(cal / "footprint_summary.csv", index=False)
    log.info(
        "frequency footprint (median share of artifact energy, %%): %s",
        "; ".join(f"{r.artifact}/{r.band} {r.median:.1f}" for r in summ.itertuples()),
    )
