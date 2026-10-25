from __future__ import annotations

import json

import numpy as np
import pandas as pd

from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.utils import atomic_json, log


def load_long(cfg: Config, write: bool = False) -> pd.DataFrame:
    """Per-prediction metrics of every (model, seed) in one table; checks that nothing is missing."""
    frames = []
    for m in cfg.models:
        for s in cfg.seeds:
            path = cfg.paths.result_csv(m, s)
            if not path.exists():
                raise SystemExit(f"missing {path}; run `simulated-mri-artifact-stress-test infer` for {m} seed {s}")
            d = pd.read_csv(path)
            d.insert(0, "seed", s)
            d.insert(0, "model", m)
            frames.append(d)
    df = pd.concat(frames).drop_duplicates(["model", "seed", "case_id", "artifact", "level"])
    df = df.sort_values(["model", "seed", "case_id", "artifact", "level"]).reset_index(drop=True)
    n_cases = df.case_id.nunique()
    expected = len(cfg.models) * len(cfg.seeds) * n_cases * len(cfg.conditions)
    if len(df) != expected:
        raise SystemExit(f"{len(df)} result rows, expected {expected}: inference is incomplete")
    df["vol_err_pct"] = 100.0 * (df.pred_ml / df.gt_ml - 1.0)
    if write:
        df.to_csv(cfg.paths.results_long, index=False)
        log.info(
            "results_test_long.csv: %d rows (%d models x %d seeds x %d cases x %d conditions)",
            len(df),
            len(cfg.models),
            len(cfg.seeds),
            n_cases,
            len(cfg.conditions),
        )
    return df


def load_ssim(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.paths.ssim_test).drop_duplicates(["case_id", "artifact", "level"])


def summarize(cfg: Config) -> pd.DataFrame:
    """Merged result table, per-level SSIM statistics and a per-checkpoint training/inference summary."""
    paths = cfg.paths
    paths.summary.mkdir(parents=True, exist_ok=True)
    df = load_long(cfg, write=True)
    deg = json.load(open(paths.degradation_config))
    ssim = load_ssim(cfg)
    rows = []
    for a in cfg.artifacts:
        for l in cfg.levels:
            sub = ssim[(ssim.artifact == a) & (ssim.level == l)]
            v = sub.ssim.to_numpy()
            rows.append(
                {
                    "artifact": a,
                    "level": l,
                    "primary": l in cfg.primary_levels,
                    "capped": deg["capped"][a][str(l)],
                    "target_ssim": cfg.target_ssim(l),
                    "param": float(sub.param.iloc[0]),
                    "median": float(np.median(v)),
                    "q1": float(np.percentile(v, 25)),
                    "q3": float(np.percentile(v, 75)),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "n": len(v),
                }
            )
    pd.DataFrame(rows).to_csv(paths.summary / "ssim_levels_test.csv", index=False)

    rep_path = paths.sanity
    rep = pd.read_csv(rep_path).set_index(["model", "seed"]) if rep_path.exists() else None
    ck = []
    for m in cfg.models:
        for s in cfg.seeds:
            sm = json.load(open(paths.run_dir(m, s) / "summary.json"))
            sub = df[(df.model == m) & (df.seed == s)]
            ck.append(
                {
                    "model": m,
                    "seed": s,
                    "best_epoch": sm.get("best_epoch"),
                    "best_val_dice": sm.get("best_val_dice"),
                    "inference_val_dice": float(rep.loc[(m, s), "inference_val_dice"]) if rep is not None else None,
                    "train_time_h": sm.get("train_time_h"),
                    "n_empty_pred": int(sub.empty_pred.sum()),
                }
            )
    pd.DataFrame(ck).to_csv(paths.summary / "checkpoint_summary.csv", index=False)
    df.groupby(["model", "seed", "artifact", "level"]).empty_pred.sum().reset_index().to_csv(
        paths.summary / "empty_predictions.csv", index=False
    )
    atomic_json(
        {
            "models": cfg.models,
            "seeds": cfg.seeds,
            "test_cases": int(df.case_id.nunique()),
            "conditions": len(cfg.conditions),
            "rows": len(df),
            "primary_levels": cfg.primary_levels,
            "capped_levels": {a: [l for l in cfg.levels if deg["capped"][a][str(l)]] for a in cfg.artifacts},
        },
        paths.summary / "run_summary.json",
    )
    return df
