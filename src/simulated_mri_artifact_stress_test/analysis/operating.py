from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from simulated_mri_artifact_stress_test.analysis.common import Arrays, first_cross, median_ssim
from simulated_mri_artifact_stress_test.analysis.results import load_ssim
from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.utils import log

NO_LIMIT_SSIM = 0.45


def icc21(v1: np.ndarray, v2: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-rater ICC."""
    X = np.stack([v1, v2], axis=1)
    nn, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (nn - 1)
    msc = nn * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((nn - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / nn)


def effects_ss(Y: np.ndarray) -> dict:
    """Balanced full-factorial sum of squares of every effect (subset of axes) of Y."""
    nd = Y.ndim
    mean_T = {}
    for r in range(nd + 1):
        for T in itertools.combinations(range(nd), r):
            other = tuple(i for i in range(nd) if i not in T)
            mean_T[T] = Y.mean(axis=other, keepdims=True) if other else Y
    SS = {}
    for r in range(1, nd + 1):
        for T in itertools.combinations(range(nd), r):
            e = np.zeros(Y.shape)
            for rr in range(len(T) + 1):
                for S in itertools.combinations(T, rr):
                    e = e + (-1) ** (len(T) - len(S)) * mean_T[S]
            SS[T] = float((e**2).sum())
    return SS


def family(T, names) -> str:
    s = {names[i] for i in T}
    has_c, has_m, has_s = "case" in s, "arch" in s, "seed" in s
    if not (has_c or has_m or has_s):
        return "severity and artifact"
    if has_c and has_s:
        return "case × seed (idiosyncratic)"
    if has_c and has_m:
        return "case × architecture"
    if has_c:
        return "case vulnerability"
    if has_s:
        return "training seed"
    return "architecture response"


def run(cfg: Config, df: pd.DataFrame) -> None:
    """Operating envelopes, clinical volumetry and the variance decomposition of the artifact-induced loss."""
    tab = cfg.paths.tables
    models, seeds, arts = cfg.models, cfg.seeds, cfg.artifacts
    bt = cfg.analysis.bootstrap
    A = Arrays(cfg, df)
    n = len(A.cases)
    SSIM = median_ssim(cfg, load_ssim(cfg))
    LMAX = {a: int(v) for a, v in cfg.analysis.max_level.items()}
    levels5 = cfg.primary_levels
    rng = np.random.default_rng(cfg.analysis.seeds.envelope)
    idx = rng.integers(0, n, (bt.envelope, n))

    rows, prow = [], []
    crit = [
        ("DSC loss 2%", "dsc", 0.98),
        ("DSC loss 5%", "dsc", 0.95),
        ("NSD loss 5%", "nsd", 0.95),
        ("NSD loss 10%", "nsd", 0.90),
    ]
    for m in models:
        for a in arts:
            Lm = min(LMAX[a], A.maxL[m])
            x = SSIM[a][: Lm + 1]
            for name, met, thr in crit:
                arr = np.nanmean(A.get(m, met, a), axis=0)[: Lm + 1]
                ratio = arr.mean(1) / arr[0].mean()
                pt = first_cross(x, ratio, thr)
                bs = []
                for b in range(bt.envelope):
                    v = first_cross(x, arr[:, idx[b]].mean(1) / arr[0, idx[b]].mean(), thr)
                    bs.append(NO_LIMIT_SSIM if np.isnan(v) else v)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(model=m, artifact=a, criterion=name, ssim=pt, lo=lo, hi=hi, reached=not np.isnan(pt)))
            dsc = A.get(m, "dsc", a)[:, : Lm + 1, :]
            drop = dsc[:, :1, :] - dsc
            for tol in (0.02, 0.05):
                pr = (drop <= tol).mean(axis=(0, 2))
                pr_b = np.stack([(drop[:, :, idx[b]] <= tol).mean(axis=(0, 2)) for b in range(bt.envelope_passrate)])
                for l in range(Lm + 1):
                    prow.append(
                        dict(
                            model=m,
                            artifact=a,
                            tol=tol,
                            level=l,
                            ssim=x[l],
                            passrate=pr[l],
                            lo=np.percentile(pr_b[:, l], 2.5),
                            hi=np.percentile(pr_b[:, l], 97.5),
                        )
                    )
                pt = first_cross(x, pr, 0.90)
                bs = [
                    NO_LIMIT_SSIM if np.isnan(v) else v
                    for v in (first_cross(x, pr_b[b], 0.90) for b in range(bt.envelope_passrate))
                ]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(
                    dict(
                        model=m,
                        artifact=a,
                        criterion=f"90% of cases within {tol:.2f} DSC",
                        ssim=pt,
                        lo=lo,
                        hi=hi,
                        reached=not np.isnan(pt),
                    )
                )
    pd.DataFrame(rows).to_csv(tab / "envelope.csv", index=False)
    pd.DataFrame(prow).to_csv(tab / "passrate.csv", index=False)

    vrows, crow = [], []
    for m in models:
        for a in arts:
            P = A.get(m, "pred_ml", a)
            G = A.get(m, "gt_ml", a)[:, 0, :]
            for l in (0, 3, 5, 7):
                vp, vg = P[:, l, :].mean(axis=0), G.mean(axis=0)
                rel = 100 * (P[:, l, :] - G) / G
                icc = icc21(vg, vp)
                ib = [icc21(vg[i], vp[i]) for i in rng.integers(0, n, (bt.volumetry_icc, n))]
                bias, sd = rel.mean(), rel.std(ddof=1)
                vrows.append(
                    dict(
                        model=m,
                        artifact=a,
                        level=l,
                        bias_pct=bias,
                        loa_lo=bias - 1.96 * sd,
                        loa_hi=bias + 1.96 * sd,
                        icc=icc,
                        icc_lo=np.percentile(ib, 2.5),
                        icc_hi=np.percentile(ib, 97.5),
                        frac_err_gt10=(np.abs(rel) > 10).mean(),
                        frac_err_gt20=(np.abs(rel) > 20).mean(),
                        mae_ml=np.abs(P[:, l, :] - G).mean(),
                    )
                )
            for l in cfg.levels:
                if l > min(LMAX[a], A.maxL[m]):
                    continue
                ch = 100 * (P[:, l, :] - P[:, 0, :]) / np.maximum(P[:, 0, :], 1e-3)
                ach = np.abs(ch)
                boot10 = [(ach[:, i] > 10).mean() for i in idx[: bt.volumetry_icc]]
                crow.append(
                    dict(
                        model=m,
                        artifact=a,
                        level=l,
                        ssim=SSIM[a][l],
                        median_abs=np.median(ach),
                        p90_abs=np.percentile(ach, 90),
                        frac_gt10=(ach > 10).mean(),
                        frac_gt20=(ach > 20).mean(),
                        frac_gt10_lo=np.percentile(boot10, 2.5),
                        frac_gt10_hi=np.percentile(boot10, 97.5),
                        median_signed=np.median(ch),
                    )
                )
    pd.DataFrame(vrows).to_csv(tab / "volumetry.csv", index=False)
    pd.DataFrame(crow).to_csv(tab / "volchange.csv", index=False)
    np.savez(
        tab / "volchange_L5.npz",
        **{
            f"{m}_{a}": 100
            * (A.get(m, "pred_ml", a)[:, 5, :] - A.get(m, "pred_ml", a)[:, 0, :])
            / np.maximum(A.get(m, "pred_ml", a)[:, 0, :], 1e-3)
            for m in models
            for a in arts
        },
    )

    def build_Y(metric, arts_sel):
        Y = np.zeros((n, len(models), len(seeds), len(arts_sel), len(levels5)))
        for im, m in enumerate(models):
            for ia, a in enumerate(arts_sel):
                arr = A.get(m, metric, a)
                Y[:, im, :, ia, :] = np.transpose(arr[:, :1, :] - arr[:, levels5, :], (2, 0, 1))
        return Y

    vr = []
    for metric in ("dsc", "nsd"):
        for label, sel in [("all", arts)] + [(a, [a]) for a in arts]:
            Y = build_Y(metric, sel)
            names = ["case", "arch", "seed", "artifact", "level"]
            if len(sel) == 1:
                Y, names = Y[:, :, :, 0, :], ["case", "arch", "seed", "level"]
            SS = effects_ss(Y)
            tot = sum(SS.values())
            fam = {}
            for T, v in SS.items():
                nm = [names[i] for i in T]
                f_ = "severity and artifact" if (len(sel) == 1 and set(nm) <= {"level"}) else family(T, names)
                fam[f_] = fam.get(f_, 0.0) + v
            sev = fam.get("severity and artifact", 0.0)
            for f_, v in fam.items():
                vr.append(
                    dict(
                        metric=metric,
                        scope=label,
                        family=f_,
                        share_total=100 * v / tot,
                        share_beyond_severity=(100 * v / (tot - sev) if f_ != "severity and artifact" else np.nan),
                        ss=v,
                        total_ss=tot,
                    )
                )
    pd.DataFrame(vr).to_csv(tab / "variance.csv", index=False)

    tr = []
    for metric in ("dsc", "nsd"):
        SS = effects_ss(build_Y(metric, arts))
        tot = sum(SS.values())
        names = ["case", "arch", "seed", "artifact", "level"]
        for T, v in SS.items():
            tr.append(dict(metric=metric, term=" × ".join(names[i] for i in T), share_total=100 * v / tot))
    pd.DataFrame(tr).sort_values(["metric", "share_total"], ascending=[True, False]).to_csv(
        tab / "variance_terms.csv", index=False
    )
    V = pd.DataFrame(vr)
    out = {
        "variance_all_dsc": V[(V.metric == "dsc") & (V.scope == "all")]
        .set_index("family")[["share_total", "share_beyond_severity"]]
        .round(2)
        .to_dict("index")
    }
    json.dump(out, open(tab / "extra.json", "w"), indent=1)
    log.info("operating envelopes, volumetry and variance decomposition written to %s", tab)
