from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from simulated_mri_artifact_stress_test.analysis.common import Arrays, boot_p, ci, hboot, holm, rauc_arr
from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.utils import log


def run(cfg: Config, df: pd.DataFrame) -> None:
    """Model summaries, H1 (clean DSC vs robustness), H2 (pairwise reversals), curves, tail risk, size and ranks."""
    tab = cfg.paths.tables
    tab.mkdir(parents=True, exist_ok=True)
    models, seeds, arts = cfg.models, cfg.seeds, cfg.artifacts
    an, bt = cfg.analysis, cfg.analysis.bootstrap
    excl = an.exclude_clean_dsc
    rng = np.random.default_rng(an.seeds.stats)
    A = Arrays(cfg, df)
    n = len(A.cases)

    S = {}
    for m in models:
        for a in arts:
            for met in ("dsc", "nsd"):
                arr = A.get(m, met, a)
                S[(m, f"rauc_{met}_{a}")] = rauc_arr(arr, excl, 5)
                S[(m, f"d_{met}_{a}")] = np.nanmean(arr[:, 1:6, :], axis=1)
            arr = A.get(m, "hd95", a)
            S[(m, f"hdinfl_{a}")] = np.nanmean(arr[:, 1:6, :] / np.maximum(arr[:, :1, :], 1e-6), axis=1)
            S[(m, f"drop5_{a}")] = A.get(m, "dsc", a)[:, :1, :].squeeze(1) - A.get(m, "dsc", a)[:, 5, :]
        S[(m, "clean_dsc")] = A.get(m, "dsc", arts[0])[:, 0, :]
        S[(m, "clean_nsd")] = A.get(m, "nsd", arts[0])[:, 0, :]
        S[(m, "clean_hd95")] = A.get(m, "hd95", arts[0])[:, 0, :]
        S[(m, "gt_ml")] = A.get(m, "gt_ml", arts[0])[:, 0, :]
        for sc in ("rauc_dsc", "rauc_nsd", "d_dsc", "d_nsd"):
            S[(m, f"{sc}_overall")] = np.mean([S[(m, f"{sc}_{a}")] for a in arts], axis=0)

    rows = []
    for m in models:
        r = {"model": m}
        for name in ["clean_dsc", "clean_nsd", "clean_hd95"] + [
            f"{p}_{a}" for p in ("d_dsc", "rauc_dsc", "d_nsd", "rauc_nsd") for a in arts + ["overall"]
        ]:
            per_seed = np.nanmean(S[(m, name)], axis=1)
            r[name] = per_seed.mean()
            r[name + "_sd"] = per_seed.std(ddof=1)
        rows.append(r)
    pd.DataFrame(rows).to_csv(tab / "model_summary.csv", index=False)

    ck = []
    for m in models:
        for si, s in enumerate(seeds):
            r = {
                "model": m,
                "seed": s,
                "clean_dsc": np.nanmean(S[(m, "clean_dsc")][si]),
                "clean_nsd": np.nanmean(S[(m, "clean_nsd")][si]),
            }
            for a in arts + ["overall"]:
                for sc in ("rauc_dsc", "d_dsc", "rauc_nsd", "d_nsd"):
                    r[f"{sc}_{a}"] = np.nanmean(S[(m, f"{sc}_{a}")][si])
            ck.append(r)
    CK = pd.DataFrame(ck)
    CK.to_csv(tab / "checkpoint_scores.csv", index=False)

    h1 = []
    for a in arts + ["overall"]:
        for sc in ("rauc_dsc", "d_dsc", "rauc_nsd", "d_nsd"):
            base = "clean_nsd" if sc.endswith("nsd") else "clean_dsc"
            t, p = kendalltau(CK[base], CK[f"{sc}_{a}"])
            cs_x = np.nanmean(np.stack([S[(m, base)] for m in models]), axis=(0, 1))
            cs_y = np.nanmean(np.stack([S[(m, f"{sc}_{a}")] for m in models]), axis=(0, 1))
            ok = ~np.isnan(cs_x) & ~np.isnan(cs_y)
            x, y = cs_x[ok], cs_y[ok]
            rho = spearmanr(x, y)[0]
            bs = []
            for _ in range(bt.h1_rho):
                i = rng.integers(0, len(x), len(x))
                bs.append(spearmanr(x[i], y[i])[0])
            h1.append(
                dict(
                    artifact=a,
                    score=sc,
                    tau=t,
                    tau_p=p,
                    rho=rho,
                    rho_lo=np.percentile(bs, 2.5),
                    rho_hi=np.percentile(bs, 97.5),
                    n_cases=len(x),
                )
            )
    pd.DataFrame(h1).to_csv(tab / "h1.csv", index=False)

    h2 = []
    for a in arts + ["overall"]:
        boots = {
            key: hboot({m: S[(m, key)] for m in models}, bt.h2, an.seeds.h2_bootstrap, len(seeds))
            for key in ("clean_dsc", f"rauc_dsc_{a}", f"d_dsc_{a}", f"rauc_nsd_{a}", f"d_nsd_{a}")
        }
        pt = {key: {m: np.nanmean(np.nanmean(S[(m, key)], axis=0)) for m in models} for key in boots}
        for m1, m2 in combinations(models, 2):
            r = dict(pair=f"{m1} - {m2}", artifact=a)
            for key, nm in (
                ("clean_dsc", "clean"),
                (f"rauc_dsc_{a}", "rauc"),
                (f"d_dsc_{a}", "ddsc"),
                (f"rauc_nsd_{a}", "rauc_nsd"),
                (f"d_nsd_{a}", "dnsd"),
            ):
                dd = boots[key][m1] - boots[key][m2]
                lo, hi = ci(dd)
                r.update(
                    {f"d_{nm}": pt[key][m1] - pt[key][m2], f"d_{nm}_lo": lo, f"d_{nm}_hi": hi, f"p_{nm}": boot_p(dd)}
                )
            w = np.nanmean(S[(m1, f"rauc_dsc_{a}")], axis=0) > np.nanmean(S[(m2, f"rauc_dsc_{a}")], axis=0)
            r["winrate_rauc"] = float(np.mean(w))
            cnt = 0
            for s1 in range(len(seeds)):
                for s2 in range(len(seeds)):
                    dc = np.nanmean(S[(m1, "clean_dsc")][s1]) - np.nanmean(S[(m2, "clean_dsc")][s2])
                    dr = np.nanmean(S[(m1, f"rauc_dsc_{a}")][s1]) - np.nanmean(S[(m2, f"rauc_dsc_{a}")][s2])
                    cnt += int(np.sign(dc) != np.sign(dr))
            r["seed_combos_opposite"] = cnt
            h2.append(r)
    H2 = pd.DataFrame(h2)
    mask = H2.artifact.isin(arts)
    H2.loc[mask, "p_rauc_holm"] = holm(H2.loc[mask, "p_rauc"].to_numpy())
    H2.loc[mask, "p_rauc_nsd_holm"] = holm(H2.loc[mask, "p_rauc_nsd"].to_numpy())
    H2["reversal"] = (
        (np.sign(H2.d_clean) != np.sign(H2.d_rauc))
        & ((H2.d_rauc_lo > 0) | (H2.d_rauc_hi < 0))
        & (H2.d_clean_lo * H2.d_clean_hi > 0)
        & (H2.seed_combos_opposite == len(seeds) ** 2)
    )
    H2.to_csv(tab / "h2.csv", index=False)

    idx = rng.integers(0, n, (bt.curves, n))
    curves = []
    for m in models:
        for a in arts:
            for met in ("dsc", "nsd", "hd95"):
                arr = np.nanmean(A.get(m, met, a), axis=0)
                clean = arr[0]
                for l in range(0, A.maxL[m] + 1):
                    x = arr[l]
                    bs = x[idx].mean(1) / clean[idx].mean(1)
                    curves.append(
                        dict(
                            model=m,
                            artifact=a,
                            metric=met,
                            level=l,
                            mean=x.mean(),
                            ratio=x.mean() / clean.mean(),
                            lo=np.percentile(bs, 2.5),
                            hi=np.percentile(bs, 97.5),
                        )
                    )
    pd.DataFrame(curves).to_csv(tab / "curves.csv", index=False)

    gt_case = np.nanmean(np.stack([S[(m, "gt_ml")] for m in models]), axis=(0, 1))
    q = pd.qcut(gt_case, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    tail, size = [], []
    for m in models:
        for a in arts:
            dr = S[(m, f"drop5_{a}")].ravel()
            w = np.sort(dr)[::-1][: max(1, len(dr) // 10)]
            rho_v = spearmanr(np.log(np.tile(gt_case, len(seeds))), dr)[0]
            rho_c = spearmanr(np.tile(np.nanmean(S[(m, "clean_dsc")], axis=0), len(seeds)), dr)[0]
            tail.append(
                dict(
                    model=m,
                    artifact=a,
                    mean_drop=dr.mean(),
                    median_drop=np.median(dr),
                    p90=np.percentile(dr, 90),
                    cvar10=w.mean(),
                    max_drop=dr.max(),
                    frac005=(dr > 0.05).mean(),
                    frac010=(dr > 0.10).mean(),
                    rho_volume=rho_v,
                    rho_cleandsc=rho_c,
                )
            )
            for qq in ["Q1", "Q2", "Q3", "Q4"]:
                sel = q == qq
                size.append(
                    dict(
                        model=m,
                        artifact=a,
                        quartile=qq,
                        rauc=100 * np.nanmean(S[(m, f"rauc_dsc_{a}")][:, sel]),
                        drop5=np.mean(S[(m, f"drop5_{a}")][:, sel]),
                        gt_ml=gt_case[sel].mean(),
                        n=int(sel.sum()),
                    )
                )
    pd.DataFrame(tail).to_csv(tab / "tail.csv", index=False)
    pd.DataFrame(size).to_csv(tab / "size.csv", index=False)
    np.savez(tab / "drop5.npz", **{f"{m}_{a}": S[(m, f"drop5_{a}")] for m in models for a in arts}, gt=gt_case)

    rk = []
    for a in arts:
        crit = (
            ("clean DSC", lambda m: np.mean(S[(m, "clean_dsc")])),
            ("clean NSD", lambda m: np.mean(S[(m, "clean_nsd")])),
            ("dDSC", lambda m: np.mean(S[(m, f"d_dsc_{a}")])),
            ("rAUC (DSC)", lambda m: np.nanmean(S[(m, f"rauc_dsc_{a}")])),
            ("rAUC (NSD)", lambda m: np.nanmean(S[(m, f"rauc_nsd_{a}")])),
            ("HD95 inflation (inv.)", lambda m: -np.mean(S[(m, f"hdinfl_{a}")])),
            ("tail share (inv.)", lambda m: -np.mean(S[(m, f"drop5_{a}")] > 0.05)),
        )
        for name, fn in crit:
            vals = {m: fn(m) for m in models}
            for r_, m in enumerate(sorted(models, key=lambda m: -vals[m]), 1):
                rk.append(dict(artifact=a, criterion=name, model=m, rank=r_, value=vals[m]))
    pd.DataFrame(rk).to_csv(tab / "ranks.csv", index=False)

    pool = []
    idx2 = np.random.default_rng(an.seeds.pooled_retention).integers(0, n, (bt.pooled, n))
    for a in arts:
        for met in ("dsc", "nsd", "hd95"):
            per = {m: np.nanmean(A.get(m, met, a), axis=0) for m in models}
            for l in range(0, len(cfg.levels) + 1):
                xs = [per[m][l] for m in models if l < per[m].shape[0]]
                cs_ = [per[m][0] for m in models if l < per[m].shape[0]]
                x, c0 = np.mean(xs, axis=0), np.mean(cs_, axis=0)
                bs = x[idx2].mean(1) / c0[idx2].mean(1)
                pool.append(
                    dict(
                        artifact=a,
                        metric=met,
                        level=l,
                        ratio=x.mean() / c0.mean(),
                        lo=np.percentile(bs, 2.5),
                        hi=np.percentile(bs, 97.5),
                    )
                )
    pd.DataFrame(pool).to_csv(tab / "pooled_retention.csv", index=False)
    log.info("statistics written to %s", tab)
