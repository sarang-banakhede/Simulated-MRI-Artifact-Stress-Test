from __future__ import annotations

import numpy as np
import pandas as pd

from simulated_mri_artifact_stress_test.config import Config

_trapz = getattr(np, "trapezoid", None) or np.trapz


class Arrays:
    """arr[model, metric, artifact] -> (seed, level 0..maxL, case); level 0 is the clean prediction."""

    def __init__(
        self, cfg: Config, df: pd.DataFrame, metrics=("dsc", "nsd", "hd95", "precision", "recall", "pred_ml", "gt_ml")
    ):
        self.cfg = cfg
        self.models, self.seeds, self.arts = cfg.models, cfg.seeds, cfg.artifacts
        self.cases = sorted(df.case_id.unique())
        ci = {c: i for i, c in enumerate(self.cases)}
        self.v = {}
        self.maxL = {m: int(df[df.model == m].level.max()) for m in self.models}
        clean = df[df.level == 0].drop_duplicates(["model", "seed", "case_id"])
        for m in self.models:
            dm, cm = df[df.model == m], clean[clean.model == m]
            for met in metrics:
                for a in self.arts:
                    arr = np.full((len(self.seeds), self.maxL[m] + 1, len(self.cases)), np.nan)
                    for si, s in enumerate(self.seeds):
                        cs = cm[cm.seed == s]
                        arr[si, 0, [ci[c] for c in cs.case_id]] = cs[met].to_numpy()
                        da = dm[(dm.artifact == a) & (dm.seed == s) & (dm.level > 0)]
                        arr[si, da.level.to_numpy(), [ci[c] for c in da.case_id]] = da[met].to_numpy()
                    self.v[(m, met, a)] = arr

    def get(self, m: str, met: str, a: str) -> np.ndarray:
        return self.v[(m, met, a)]


def rauc_arr(arr: np.ndarray, exclude_clean: float, lmax: int = 5) -> np.ndarray:
    """(seed, level, case) -> area under the retention curve over L0..Lmax; NaN if the clean value is < exclude_clean."""
    R = arr[:, : lmax + 1, :] / arr[:, :1, :]
    R[np.broadcast_to(arr[:, :1, :] < exclude_clean, R.shape)] = np.nan
    return _trapz(R, dx=1.0, axis=1) / lmax


def hboot(S_by_model: dict, B: int, seed: int, n_seeds: int) -> dict:
    """Hierarchical bootstrap of per-model means: cases jointly, seeds with replacement within each model."""
    rng = np.random.default_rng(seed)
    n = next(iter(S_by_model.values())).shape[1]
    idx = rng.integers(0, n, (B, n))
    out = {}
    for m, S in S_by_model.items():
        W = rng.multinomial(n_seeds, [1 / n_seeds] * n_seeds, size=B) / float(n_seeds)
        vals = np.empty(B)
        for b in range(B):
            vals[b] = np.nanmean((W[b] @ S)[idx[b]])
        out[m] = vals
    return out


def ci(x, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def boot_p(d: np.ndarray) -> float:
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return max(p, 1.0 / (len(d) + 1))


def holm(ps) -> np.ndarray:
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run, m = 0.0, len(ps)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def first_cross(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """SSIM at which curve y (starting at 1) first drops below thr, by linear interpolation."""
    for l in range(1, len(y)):
        if y[l] < thr <= y[l - 1]:
            f = (y[l - 1] - thr) / (y[l - 1] - y[l])
            return x[l - 1] + f * (x[l] - x[l - 1])
        if y[l] < thr and y[l - 1] < thr:
            return x[l - 1]
    return np.nan


def median_ssim(cfg: Config, ssim_test: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        a: np.array(
            [1.0]
            + [float(ssim_test[(ssim_test.artifact == a) & (ssim_test.level == l)].ssim.median()) for l in cfg.levels]
        )
        for a in cfg.artifacts
    }
