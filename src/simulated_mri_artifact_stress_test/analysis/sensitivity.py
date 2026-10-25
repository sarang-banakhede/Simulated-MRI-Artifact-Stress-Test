from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from simulated_mri_artifact_stress_test.analysis.common import Arrays, first_cross, median_ssim
from simulated_mri_artifact_stress_test.analysis.operating import effects_ss
from simulated_mri_artifact_stress_test.analysis.results import load_ssim
from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.utils import log


def _family(T, names) -> str:
    s = {names[i] for i in T}
    c, m_, s_ = "case" in s, "arch" in s, "seed" in s
    if not (c or m_ or s_):
        return "severity and artifact"
    if c and s_:
        return "case × seed"
    if c and m_:
        return "case × architecture"
    if c:
        return "case"
    if s_:
        return "training seed"
    return "architecture"


def run(cfg: Config, df: pd.DataFrame) -> None:
    """Sensitivity checks: SSIM spread, SSIM-restricted and adjusted effects, pass-share sensitivity, relative-loss
    variance decomposition, H1 with the absolute loss, Bland-Altman volumes and seed agreement."""
    tab = cfg.paths.tables
    models, seeds, arts = cfg.models, cfg.seeds, cfg.artifacts
    bt = cfg.analysis.bootstrap
    A = Arrays(cfg, df)
    n, cases = len(A.cases), A.cases
    ss = load_ssim(cfg)
    rng = np.random.default_rng(cfg.analysis.seeds.sensitivity)
    LMAX = {a: int(v) for a, v in cfg.analysis.max_level.items()}
    rng.integers(0, n, (bt.sensitivity, n))  # draw kept so later bootstraps follow the published random stream

    rows = []
    for a in arts:
        for l in cfg.levels:
            v = ss[(ss.artifact == a) & (ss.level == l)].ssim
            rows.append(
                dict(
                    artifact=a,
                    level=l,
                    median=v.median(),
                    p05=v.quantile(0.05),
                    p25=v.quantile(0.25),
                    p75=v.quantile(0.75),
                    p95=v.quantile(0.95),
                )
            )
    pd.DataFrame(rows).to_csv(tab / "ssim_spread.csv", index=False)

    ssim_case = {
        (a, l): ss[(ss.artifact == a) & (ss.level == l)].set_index("case_id").ssim.reindex(cases).to_numpy()
        for a in arts
        for l in cfg.levels
    }
    srows, arows = [], []
    for a in arts:
        for l in (3, 5):
            tgt = cfg.target_ssim(l)
            sel = np.abs(ssim_case[(a, l)] - tgt) <= 0.03
            for m in models:
                for met in ("dsc", "nsd"):
                    arr = np.nanmean(A.get(m, met, a), axis=0)
                    srows.append(
                        dict(
                            model=m,
                            artifact=a,
                            level=l,
                            metric=met,
                            n_sub=int(sel.sum()),
                            retention_full=100 * arr[l].mean() / arr[0].mean(),
                            retention_sub=100 * arr[l][sel].mean() / arr[0][sel].mean(),
                        )
                    )
            recs = []
            for m in models:
                dsc = A.get(m, "dsc", a)
                drop = dsc[:, 0, :] - dsc[:, l, :]
                for si in range(len(seeds)):
                    recs.append(pd.DataFrame(dict(model=m, case=cases, drop=drop[si], dssim=ssim_case[(a, l)] - tgt)))
            D = pd.concat(recs, ignore_index=True)
            fit = smf.ols("drop ~ C(model) + dssim", data=D).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(D.case)[0]}
            )
            raw = D.groupby("model")["drop"].mean()
            for m in models:
                adj = fit.params["Intercept"] + fit.params.get(f"C(model)[T.{m}]", 0.0)
                arows.append(
                    dict(
                        artifact=a,
                        level=l,
                        model=m,
                        drop_raw=raw[m],
                        drop_adj=adj,
                        slope=fit.params["dssim"],
                        slope_p=fit.pvalues["dssim"],
                    )
                )
    pd.DataFrame(srows).to_csv(tab / "ssim_restricted.csv", index=False)
    pd.DataFrame(arows).to_csv(tab / "ssim_adjusted.csv", index=False)

    SS_MED = median_ssim(cfg, ss)
    rows = []
    for m in models:
        for a in arts:
            x = SS_MED[a][: LMAX[a] + 1]
            dsc = A.get(m, "dsc", a)[:, : LMAX[a] + 1, :]
            drop = dsc[:, :1, :] - dsc
            for eps in (0.02, 0.05):
                for share in (0.80, 0.90, 0.95):
                    rows.append(
                        dict(
                            model=m,
                            artifact=a,
                            eps=eps,
                            share=share,
                            ssim=first_cross(x, (drop <= eps).mean(axis=(0, 2)), share),
                        )
                    )
    pd.DataFrame(rows).to_csv(tab / "envelope_sens.csv", index=False)

    levels5 = cfg.primary_levels

    def build(metric, relative, sel):
        Y = np.zeros((n, len(models), len(seeds), len(sel), len(levels5)))
        for im, m in enumerate(models):
            for ia, a in enumerate(sel):
                arr = A.get(m, metric, a)
                dr = arr[:, :1, :] - arr[:, levels5, :]
                if relative:
                    dr = dr / np.maximum(arr[:, :1, :], 1e-6)
                Y[:, im, :, ia, :] = np.transpose(dr, (2, 0, 1))
        return Y

    vrows = []
    for metric, relative, lab in (
        ("dsc", False, "DSC loss"),
        ("nsd", False, "NSD loss"),
        ("dsc", True, "DSC relative loss"),
    ):
        for scope, sel in [("all", arts)] + [(a, [a]) for a in arts]:
            Y = build(metric, relative, sel)
            names = ["case", "arch", "seed", "artifact", "level"]
            if len(sel) == 1:
                Y, names = Y[:, :, :, 0, :], ["case", "arch", "seed", "level"]
            SS = effects_ss(Y)
            tot = sum(SS.values())
            agg = {}
            for T, v in SS.items():
                agg[_family(T, names)] = agg.get(_family(T, names), 0.0) + v
            for f_, v in agg.items():
                vrows.append(dict(response=lab, scope=scope, family=f_, share=100 * v / tot))
    pd.DataFrame(vrows).to_csv(tab / "variance_all.csv", index=False)

    rows = []
    for a in arts + ["overall"]:
        cs_x = np.mean([np.nanmean(A.get(m, "dsc", arts[0])[:, 0, :], axis=0) for m in models], axis=0)
        ddsc = []
        for m in models:
            if a == "overall":
                v = np.mean([np.nanmean(A.get(m, "dsc", aa)[:, 1:6, :], axis=(0, 1)) for aa in arts], axis=0)
            else:
                v = np.nanmean(A.get(m, "dsc", a)[:, 1:6, :], axis=(0, 1))
            ddsc.append(v)
        loss = cs_x - np.mean(ddsc, axis=0)
        bs = [spearmanr(cs_x[i], loss[i])[0] for i in rng.integers(0, n, (bt.sensitivity_rho, n))]
        rows.append(
            dict(
                artifact=a, rho_abs_loss=spearmanr(cs_x, loss)[0], lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)
            )
        )
    pd.DataFrame(rows).to_csv(tab / "h1_abs_loss.csv", index=False)

    vrows = []
    for m in models:
        for a in arts:
            P = A.get(m, "pred_ml", a)
            G = A.get(m, "gt_ml", a)[:, 0, :]
            for l in (0, 5):
                diff = P[:, l, :] - G
                vrows.append(
                    dict(
                        model=m,
                        artifact=a,
                        level=l,
                        bias_ml=diff.mean(),
                        loa_lo_ml=diff.mean() - 1.96 * diff.std(ddof=1),
                        loa_hi_ml=diff.mean() + 1.96 * diff.std(ddof=1),
                        mae_ml=np.abs(diff).mean(),
                        med_abs_ml=np.median(np.abs(diff)),
                    )
                )
    pd.DataFrame(vrows).to_csv(tab / "volumetry_ba.csv", index=False)
    np.savez(
        tab / "ba_arrays.npz",
        gt=A.get(models[0], "gt_ml", arts[0])[0, 0, :],
        **{f"{m}_{a}_L{l}": A.get(m, "pred_ml", a)[:, l, :].mean(axis=0) for m in models for a in arts for l in (0, 5)},
    )

    ag = pd.read_csv(cfg.paths.post / "agreement.csv")
    sa = (
        ag[ag.kind == "seed"]
        .groupby(["a", "artifact", "level"])
        .dice.mean()
        .reset_index()
        .rename(columns={"a": "model"})
    )
    sa.to_csv(tab / "seed_agreement_levels.csv", index=False)
    log.info("sensitivity analyses written to %s", tab)
