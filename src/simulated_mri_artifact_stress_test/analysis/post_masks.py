from __future__ import annotations

import json
import multiprocessing as mp
from itertools import combinations

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import kendalltau, spearmanr
from tqdm.auto import tqdm

from simulated_mri_artifact_stress_test.analysis.common import _trapz
from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.data.nifti import nii_path
from simulated_mri_artifact_stress_test.utils import log

_CTX: dict = {}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = int(a.sum()), int(b.sum())
    if sa + sb == 0:
        return 1.0
    return 2.0 * int((a & b).sum()) / (sa + sb)


def _crop(p):
    ax = [np.where(p.any(axis=tuple(j for j in range(3) if j != i)))[0] for i in range(3)]
    if any(len(x) == 0 for x in ax):
        return None
    return tuple(slice(x[0], x[-1] + 1) for x in ax)


def comp_stats(p: np.ndarray, g: np.ndarray) -> tuple[int, int, float]:
    """Connected components of a prediction, fragments (< 1% of the largest) and volume that never touches the reference."""
    sl = _crop(p)
    if sl is None:
        return 0, 0, 0.0
    lab, n = ndi.label(p[sl])
    if n == 0:
        return 0, 0, 0.0
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    hit = np.unique(lab[g[sl] & p[sl]])
    hit = hit[hit > 0]
    spur = cnt[1:].sum() - cnt[hit].sum()
    frag = int((cnt[1:] < 0.01 * cnt[1:].max()).sum())
    return n, frag, float(spur)


def _init(cfg: Config) -> None:
    _CTX["cfg"] = cfg


def _case_job(case: str):
    cfg: Config = _CTX["cfg"]
    models, seeds = cfg.models, cfg.seeds
    conds = [("clean", 0)] + [(a, l) for a in cfg.artifacts for l in cfg.levels]
    img = nib.load(str(nii_path(cfg.paths.data, "test", "labels", case)))
    g = np.asarray(img.dataobj) > 0
    vox = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    ck_rows, agree_rows, ens_rows = [], [], []
    clean_masks = {}
    gz = np.argwhere(g)
    cen = gz.mean(0) if len(gz) else np.array([np.nan] * 3)
    for a, l in conds:
        M = {
            (m, s): np.asarray(nib.load(str(cfg.paths.mask(m, a, l, s, case))).dataobj) > 0
            for m in models
            for s in seeds
        }
        if l == 0:
            clean_masks = M
        for (m, s), p in M.items():
            n, frag, spur = comp_stats(p, g)
            ck_rows.append(
                dict(
                    model=m,
                    seed=s,
                    case_id=case,
                    artifact=a,
                    level=l,
                    dsc_recomputed=dice(p, g),
                    fp_ml=float((p & ~g).sum()) * vox,
                    fn_ml=float((~p & g).sum()) * vox,
                    n_comp=n,
                    n_frag=frag,
                    spurious_ml=spur * vox,
                    dsc_vs_clean=1.0 if l == 0 else dice(p, clean_masks[(m, s)]),
                    cen_x=cen[0],
                    cen_y=cen[1],
                    cen_z=cen[2],
                )
            )
        for m in models:
            for s1, s2 in combinations(seeds, 2):
                agree_rows.append(
                    dict(kind="seed", a=m, b=m, case_id=case, artifact=a, level=l, dice=dice(M[(m, s1)], M[(m, s2)]))
                )
        for m1, m2 in combinations(models, 2):
            v = [dice(M[(m1, s1)], M[(m2, s2)]) for s1 in seeds for s2 in seeds]
            agree_rows.append(dict(kind="model", a=m1, b=m2, case_id=case, artifact=a, level=l, dice=float(np.mean(v))))
        for m in models:
            ens = sum(M[(m, s)].astype(np.uint8) for s in seeds) >= 2
            ens_rows.append(dict(ens=m, case_id=case, artifact=a, level=l, dsc=dice(ens, g)))
        ens_all = sum(p.astype(np.uint8) for p in M.values()) >= (len(M) // 2 + 1)
        ens_rows.append(dict(ens="all9", case_id=case, artifact=a, level=l, dsc=dice(ens_all, g)))
    return ck_rows, agree_rows, ens_rows


def mask_tables(cfg: Config, cases: list[str], force: bool = False):
    """Read every saved mask once: error type, fragmentation, agreement between seeds/models and majority-vote ensembles."""
    out = cfg.paths.post
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"{k}.csv" for k in ("mask_metrics", "agreement", "ensemble")}
    if not force and all(p.exists() for p in paths.values()):
        log.info("mask tables cached")
        return [pd.read_csv(paths[k]) for k in paths]
    n = cfg.workers(cfg.analysis.workers)
    log.info("reading masks of %d cases with %d workers", len(cases), n)
    ck, ag, en = [], [], []
    with mp.get_context("fork").Pool(n, initializer=_init, initargs=(cfg,)) as pool:
        for a, b, c in tqdm(pool.imap_unordered(_case_job, cases), total=len(cases), desc="masks"):
            ck.append(a)
            ag.append(b)
            en.append(c)
    res = [pd.DataFrame(sum(x, [])) for x in (ck, ag, en)]
    for k, d in zip(paths, res):
        d.to_csv(paths[k], index=False)
    return res


def _boot_ci(x, rng, boot, stat=np.mean):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), (boot, len(x)))
    b = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


class PostAnalysis:
    def __init__(self, cfg: Config, df: pd.DataFrame, mm: pd.DataFrame, ag: pd.DataFrame, en: pd.DataFrame):
        self.cfg, self.mm, self.ag, self.en = cfg, mm, ag, en
        self.models, self.seeds, self.arts = cfg.models, cfg.seeds, cfg.artifacts
        self.levels = cfg.primary_levels
        self.boot = cfg.analysis.bootstrap.post_masks
        self.rng = np.random.default_rng(cfg.analysis.seeds.post_masks)
        self.excl = cfg.analysis.exclude_clean_dsc
        self.all = df
        d = df[df.level <= self.levels[-1]]
        self.clean = d[d.level == 0].drop_duplicates(["model", "seed", "case_id"])
        self.deg = d[d.level > 0]

    def case_table(self) -> pd.DataFrame:
        base = self.clean.set_index(["model", "seed", "case_id"])[["dsc", "hd95", "nsd", "gt_ml"]]
        base.columns = ["clean_dsc", "clean_hd95", "clean_nsd", "gt_ml"]
        d = self.deg.set_index(["model", "seed", "case_id", "artifact", "level"])
        rows = []
        n = len(self.levels)
        for (m, s, c), b in base.iterrows():
            r = dict(model=m, seed=s, case_id=c, **b.to_dict())
            for a in self.arts:
                v = np.array([d.loc[(m, s, c, a, l), "dsc"] for l in self.levels])
                r[f"ddsc_{a}"] = v.mean()
                ret = np.concatenate([[1.0], v / b.clean_dsc]) if b.clean_dsc >= self.excl else np.full(n + 1, np.nan)
                r[f"rauc_{a}"] = _trapz(ret, dx=1.0) / n
                r[f"drop5_{a}"] = b.clean_dsc - v[-1]
            rows.append(r)
        T = pd.DataFrame(rows)
        T["ddsc_overall"] = T[[f"ddsc_{a}" for a in self.arts]].mean(axis=1)
        T["rauc_overall"] = T[[f"rauc_{a}" for a in self.arts]].mean(axis=1)
        return T

    def h1(self, T):
        ck = T.groupby(["model", "seed"]).mean(numeric_only=True).reset_index()
        rows = []
        for a in self.arts + ["overall"]:
            for sc in ("rauc", "ddsc"):
                t, p = kendalltau(ck.clean_dsc, ck[f"{sc}_{a}"])
                rows.append(dict(level="checkpoint(n=9)", artifact=a, score=sc, tau=t, p=p))
        cs = T.groupby("case_id").mean(numeric_only=True)
        for a in self.arts + ["overall"]:
            for sc in ("rauc", "ddsc"):
                ok = cs[[f"{sc}_{a}", "clean_dsc"]].dropna()
                r = spearmanr(ok.clean_dsc, ok[f"{sc}_{a}"])[0]
                b = []
                for _ in range(self.boot):
                    i = self.rng.integers(0, len(ok), len(ok))
                    b.append(spearmanr(ok.clean_dsc.values[i], ok[f"{sc}_{a}"].values[i])[0])
                rows.append(
                    dict(
                        level="case(n=97, 9-ckpt mean)",
                        artifact=a,
                        score=sc,
                        tau=r,
                        ci_lo=np.percentile(b, 2.5),
                        ci_hi=np.percentile(b, 97.5),
                    )
                )
        for m in self.models:
            cm = T[T.model == m].groupby("case_id").mean(numeric_only=True)
            for a in self.arts + ["overall"]:
                ok = cm[[f"rauc_{a}", "clean_dsc"]].dropna()
                rows.append(
                    dict(
                        level=f"case within {m}",
                        artifact=a,
                        score="rauc",
                        tau=spearmanr(ok.clean_dsc, ok[f"rauc_{a}"])[0],
                    )
                )
        return pd.DataFrame(rows), ck

    def h2(self, T, ck):
        rows = []
        for m1, m2 in combinations(self.models, 2):
            for a in self.arts + ["overall"]:
                row = dict(pair=f"{m1} - {m2}", artifact=a)
                for sc, col in (("clean", "clean_dsc"), ("ddsc", f"ddsc_{a}"), ("rauc", f"rauc_{a}")):
                    x = (
                        T[T.model == m1].groupby("case_id")[col].mean()
                        - T[T.model == m2].groupby("case_id")[col].mean()
                    )
                    mean, lo, hi = _boot_ci(x.values, self.rng, self.boot)
                    row.update({f"d_{sc}": mean, f"d_{sc}_lo": lo, f"d_{sc}_hi": hi})
                sg = []
                for s1 in self.seeds:
                    for s2 in self.seeds:
                        A_ = ck[(ck.model == m1) & (ck.seed == s1)].iloc[0]
                        B_ = ck[(ck.model == m2) & (ck.seed == s2)].iloc[0]
                        sg.append(np.sign(A_.clean_dsc - B_.clean_dsc) != np.sign(A_[f"rauc_{a}"] - B_[f"rauc_{a}"]))
                row["reversal_seed_combos_of_9"] = int(sum(sg))
                row["reversal_ci"] = bool(
                    np.sign(row["d_clean"]) != np.sign(row["d_rauc"]) and (row["d_rauc_lo"] > 0 or row["d_rauc_hi"] < 0)
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def curves(self):
        rows = []
        for m in self.models:
            c0 = self.clean[self.clean.model == m]
            for a in self.arts:
                for l in [0] + self.levels:
                    d = (
                        c0
                        if l == 0
                        else self.deg[(self.deg.model == m) & (self.deg.artifact == a) & (self.deg.level == l)]
                    )
                    rows.append(
                        dict(
                            model=m,
                            artifact=a,
                            level=l,
                            ssim=d.ssim.mean(),
                            dsc=d.dsc.mean(),
                            hd95=d.hd95.mean(),
                            hd95_med=d.hd95.median(),
                            nsd=d.nsd.mean(),
                            precision=d.precision.mean(),
                            recall=d.recall.mean(),
                        )
                    )
        C = pd.DataFrame(rows)
        for k in ("dsc", "hd95", "nsd", "precision", "recall"):
            C[f"{k}_rel"] = C[k] / C.groupby(["model", "artifact"])[k].transform("first")
        return C

    def tail(self):
        cl = self.clean.set_index(["model", "seed", "case_id"]).dsc
        rows = []
        for (m, a, l), d in self.deg.groupby(["model", "artifact", "level"]):
            drop = cl.reindex(pd.MultiIndex.from_frame(d[["model", "seed", "case_id"]])).values - d.dsc.values
            worst = np.sort(drop)[::-1][: max(1, len(drop) // 10)]
            rows.append(
                dict(
                    model=m,
                    artifact=a,
                    level=l,
                    mean_drop=drop.mean(),
                    median_drop=np.median(drop),
                    p90_drop=np.percentile(drop, 90),
                    cvar10_drop=worst.mean(),
                    max_drop=drop.max(),
                    frac_drop_gt_005=(drop > 0.05).mean(),
                    frac_drop_gt_010=(drop > 0.10).mean(),
                    p10_dsc=np.percentile(d.dsc, 10),
                )
            )
        return pd.DataFrame(rows)

    def size(self, T):
        cq = T.groupby("case_id").gt_ml.first()
        q = pd.qcut(cq, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
        T = T.assign(size=T.case_id.map(q))
        rows = []
        for (m, s), d in T.groupby(["model", "size"], observed=True):
            r = dict(
                model=m, size=s, n_cases=d.case_id.nunique(), gt_ml_mean=d.gt_ml.mean(), clean_dsc=d.clean_dsc.mean()
            )
            for a in self.arts + ["overall"]:
                r[f"ddsc_{a}"] = d[f"ddsc_{a}"].mean()
                r[f"rauc_{a}"] = d[f"rauc_{a}"].mean()
            rows.append(r)
        return pd.DataFrame(rows)

    def error(self):
        mm = self.mm
        g = (
            mm.groupby(["model", "artifact", "level"])[
                ["fp_ml", "fn_ml", "n_comp", "n_frag", "spurious_ml", "dsc_vs_clean"]
            ]
            .mean()
            .reset_index()
        )
        for k in ("fp_ml", "fn_ml", "n_comp", "n_frag", "spurious_ml"):
            c = mm[mm.level == 0].groupby("model")[k].mean()
            g[f"{k}_delta"] = g[k] - g.model.map(c)
        return g

    def hidden(self):
        mm = self.mm
        d = mm[mm.level > 0].merge(
            self.deg[["model", "seed", "case_id", "artifact", "level", "dsc"]],
            on=["model", "seed", "case_id", "artifact", "level"],
        )
        c = mm[mm.level == 0].set_index(["model", "seed", "case_id"]).dsc_recomputed
        d["clean_dsc"] = c.reindex(pd.MultiIndex.from_frame(d[["model", "seed", "case_id"]])).values
        d["retention"] = d.dsc / d.clean_dsc
        d["hidden"] = (d.retention >= 0.98) & (d.dsc_vs_clean < 0.90)
        d["dsc_up_but_changed"] = (d.dsc >= d.clean_dsc) & (d.dsc_vs_clean < 0.90)
        return (
            d.groupby(["model", "artifact", "level"])
            .agg(
                retention=("retention", "mean"),
                dsc_vs_clean=("dsc_vs_clean", "mean"),
                frac_hidden_change=("hidden", "mean"),
                frac_dsc_up_but_changed=("dsc_up_but_changed", "mean"),
            )
            .reset_index()
        )

    def ensemble(self):
        r = pd.concat([self.clean.assign(artifact="clean"), self.deg])
        mem = r.groupby(["model", "case_id", "artifact", "level"]).dsc.mean().reset_index()
        rows = []
        for (e, a), d in self.en[self.en.artifact != "clean"].groupby(["ens", "artifact"]):
            cl = self.en[(self.en.ens == e) & (self.en.level == 0)].set_index("case_id").dsc
            for l in self.levels:
                x = d[d.level == l].set_index("case_id").dsc
                row = dict(
                    ens=e, artifact=a, level=l, ens_dsc=x.mean(), ens_clean=cl.mean(), ens_retention=(x / cl).mean()
                )
                if e != "all9":
                    mm_ = mem[(mem.model == e) & (mem.artifact == a) & (mem.level == l)].set_index("case_id").dsc
                    row["member_mean_dsc"] = mm_.mean()
                    row["gain_over_members"] = x.mean() - mm_.mean()
                rows.append(row)
        return pd.DataFrame(rows)

    def nonmonotone(self):
        p = self.deg.pivot_table(index=["model", "seed", "case_id", "artifact"], columns="level", values="dsc")
        inc = (p.diff(axis=1).iloc[:, 1:] > 0.01).any(axis=1)
        return inc.groupby(["model", "artifact"]).mean().rename("frac_nonmonotone").reset_index()

    def agreement(self):
        return self.ag.groupby(["kind", "a", "b", "artifact", "level"]).dice.agg(["mean", "std"]).reset_index()

    def recompute_check(self):
        r = pd.concat([self.clean, self.deg]).rename(columns={"dsc": "dsc_saved"})
        r["artifact"] = np.where(r.level == 0, "clean", r.artifact)
        m = self.mm.merge(
            r[["model", "seed", "case_id", "artifact", "level", "dsc_saved"]],
            on=["model", "seed", "case_id", "artifact", "level"],
        )
        m["abs_diff"] = (m.dsc_recomputed - m.dsc_saved).abs()
        return m.groupby("model").abs_diff.agg(["max", "mean", "count"]).reset_index()

    def extended(self):
        allr = self.all
        clean = allr[allr.level == 0].drop_duplicates(["model", "seed", "case_id"])
        deg = allr[allr.level > 0]
        top = max(self.cfg.levels)
        rows = []
        for m in self.models:
            c = clean[clean.model == m]
            for a in self.arts:
                for l in range(0, top + 1):
                    d = c if l == 0 else deg[(deg.model == m) & (deg.artifact == a) & (deg.level == l)]
                    cl = c.set_index(["seed", "case_id"]).dsc
                    drop = (
                        cl.reindex(pd.MultiIndex.from_frame(d[["seed", "case_id"]])).values - d.dsc.values
                        if l > 0
                        else np.zeros(len(d))
                    )
                    rows.append(
                        dict(
                            model=m,
                            artifact=a,
                            level=l,
                            ssim=d.ssim.mean(),
                            dsc=d.dsc.mean(),
                            nsd=d.nsd.mean(),
                            hd95=d.hd95.mean(),
                            hd95_med=d.hd95.median(),
                            vol_err_pct=d.vol_err_pct.mean(),
                            vol_err_med=d.vol_err_pct.median(),
                            precision=d.precision.mean(),
                            recall=d.recall.mean(),
                            n_empty=int(d.empty_pred.sum()),
                            frac_drop_gt_005=(drop > 0.05).mean(),
                            frac_drop_gt_010=(drop > 0.10).mean(),
                            frac_drop_gt_020=(drop > 0.20).mean(),
                            p10_dsc=np.percentile(d.dsc, 10),
                        )
                    )
        E = pd.DataFrame(rows)
        for k in ("dsc", "nsd", "hd95"):
            E[f"{k}_rel"] = E[k] / E.groupby(["model", "artifact"])[k].transform("first")
        return E

    def crossings(self, E):
        out = []
        top = max(self.cfg.levels)
        for a in self.arts:
            for m1, m2 in combinations(self.models, 2):
                x = E[(E.artifact == a) & (E.model == m1)].set_index("level").dsc
                y = E[(E.artifact == a) & (E.model == m2)].set_index("level").dsc
                diff = x - y
                sgn = np.sign(diff.to_numpy())
                flips = [int(l) for l, s0, s1 in zip(diff.index[1:], sgn[:-1], sgn[1:]) if s0 != s1]
                out.append(
                    dict(
                        artifact=a,
                        pair=f"{m1} - {m2}",
                        clean_diff=diff.iloc[0],
                        diff_L5=diff.loc[5],
                        diff_L10=diff.loc[top],
                        order_flips_at_levels=str(flips),
                    )
                )
        return pd.DataFrame(out)


def run(cfg: Config, df: pd.DataFrame, force: bool = False, limit_cases: int | None = None) -> None:
    cases = sorted(json.load(open(cfg.paths.splits))["test"])
    if limit_cases:
        cases, force = cases[:limit_cases], True
    mm, ag, en = mask_tables(cfg, cases, force)
    df = df[df.case_id.isin(cases)]
    P = PostAnalysis(cfg, df, mm, ag, en)
    out = cfg.paths.post
    T = P.case_table()
    h1, ck = P.h1(T)
    R = dict(
        check=P.recompute_check(),
        h1=h1,
        h2=P.h2(T, ck),
        curves=P.curves(),
        tail=P.tail(),
        size=P.size(T),
        error=P.error(),
        hidden=P.hidden(),
        agree=P.agreement(),
        ens=P.ensemble(),
        nonmono=P.nonmonotone(),
    )
    T.to_csv(out / "case_checkpoint_scores.csv", index=False)
    ck.to_csv(out / "checkpoint_scores.csv", index=False)
    for k, d in R.items():
        d.to_csv(out / f"{k}.csv", index=False)
    E = P.extended()
    E.to_csv(out / "extended_levels.csv", index=False)
    P.crossings(E).to_csv(out / "extended_crossings.csv", index=False)
    mm.groupby(["model", "artifact", "level"])[
        ["fp_ml", "fn_ml", "n_comp", "n_frag", "spurious_ml", "dsc_vs_clean"]
    ].mean().reset_index().to_csv(out / "extended_mask_metrics.csv", index=False)
    log.info("post-inference mask analysis written to %s", out)
