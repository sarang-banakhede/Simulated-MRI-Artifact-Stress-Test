from __future__ import annotations

import contextlib
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from simulated_mri_artifact_stress_test.artifacts.calibration import calibrate, frequency_footprint
from simulated_mri_artifact_stress_test.artifacts.simulate import ArtifactSimulator
from simulated_mri_artifact_stress_test.config import Config, Paths, canonical_model
from simulated_mri_artifact_stress_test.data.nifti import bbox_of, load_case, nii_path, preprocess, to_slices
from simulated_mri_artifact_stress_test.data.splits import get_splits
from simulated_mri_artifact_stress_test.evaluation.metrics import METRIC_KEYS, case_metrics
from simulated_mri_artifact_stress_test.models import load_checkpoint
from simulated_mri_artifact_stress_test.utils import (
    Deadline,
    append_rows,
    free_gpu,
    log,
)

ROW_KEYS = ["case_id", "artifact", "level", "param", "capped", "ssim"] + METRIC_KEYS
SSIM_KEYS = ["case_id", "artifact", "level", "param", "capped", "target_ssim", "ssim"]


def _keep(x):
    return x


class CaseSet(Dataset):
    """Loads one case and builds the network input of every requested condition (clean + artifacts x levels)."""

    def __init__(
        self,
        cfg: Config,
        cases,
        split: str,
        deg_cfg: dict | None,
        conditions,
        paths: Paths,
        sim: ArtifactSimulator,
        save_degraded: bool = False,
    ):
        self.cfg, self.cases, self.split, self.deg_cfg = cfg, cases, split, deg_cfg
        self.conditions, self.paths, self.sim, self.save_degraded = conditions, paths, sim, save_degraded

    def __len__(self):
        return len(self.cases)

    def _write_degraded(self, x, mask, affine, artifact, level, case):
        path = self.paths.degraded / f"{artifact}_L{level}" / f"{case}.nii.gz"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.where(mask, x, 0.0).astype(np.float32), affine), str(path))

    def __getitem__(self, i):
        case = self.cases[i]
        data = self.cfg.paths.data
        flair, mask, lab, affine, spacing = load_case(
            nii_path(data, self.split, "images", case),
            nii_path(data, self.split, "labels", case),
            self.cfg.data.flair_channel,
        )
        lo, hi = bbox_of(mask)
        sl = to_slices(lo, hi)
        mc = mask[sl]
        clean = preprocess(flair, mask, sl)
        gt_c = np.ascontiguousarray(lab[sl])
        gt_out = int(lab.sum()) - int(gt_c.sum())
        vols, patterns, built = [], {}, {}
        for a, l in self.conditions:
            if l == 0:
                x, p, s, cap = clean, 0.0, 1.0, False
            else:
                p, cap = self.deg_cfg["params"][a][l], self.deg_cfg["capped"][a][l]
                if (a, p) not in built:
                    if a not in patterns:
                        patterns[a] = self.sim.pattern(case, a, flair.shape)
                    raw = self.sim.degrade(flair, a, p, patterns[a], mask, spacing)
                    xd = preprocess(raw, mask, sl)
                    built[(a, p)] = (xd, self.sim.ssim(clean, xd, mc), raw if self.save_degraded else None)
                x, s, raw = built[(a, p)]
                if raw is not None:
                    self._write_degraded(raw, mask, affine, a, l, case)
            vols.append({"artifact": a, "level": l, "param": float(p), "capped": bool(cap), "ssim": float(s), "x": x})
        return {
            "case": case,
            "vols": vols,
            "gt_c": gt_c,
            "gt_out": gt_out,
            "gt_native": lab if gt_out > 0 else None,
            "lo": lo,
            "hi": hi,
            "shape": list(flair.shape),
            "spacing": spacing,
            "affine": affine,
        }


def _metric_job(job: dict) -> dict:
    sl = to_slices(job["lo"], job["hi"])
    shape, sp = tuple(job["shape"]), tuple(job["spacing"])
    diag = float(np.sqrt(sum((n * s) ** 2 for n, s in zip(shape, sp))))
    native = job["gt_native"] is not None
    gt = (job["gt_native"] if native else job["gt_c"]).astype(bool)
    out = {}
    for k, pred_c in job["preds"].items():
        if native:
            pred = np.zeros(shape, bool)
            pred[sl] = pred_c.astype(bool)
        else:
            pred = pred_c.astype(bool)
        out[k] = case_metrics(pred, gt, sp, diag, job["nsd_tol"])
        path = (job["saves"] or {}).get(k)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            full = np.zeros(shape, np.uint8)
            full[sl] = pred_c
            nib.save(nib.Nifti1Image(full, job["affine"]), path)
    return out


def _load_done(path: Path, n_conditions: int) -> set[str]:
    if not Path(path).exists():
        return set()
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["case_id"], []).append(r)
    good = {
        c
        for c, rs in by.items()
        if len(rs) == n_conditions and len({(r["artifact"], int(r["level"])) for r in rs}) == n_conditions
    }
    if len(good) != len(by):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_KEYS)
            w.writeheader()
            w.writerows(r for r in rows if r["case_id"] in good)
        log.info("%s: dropped %d incomplete cases", path, len(by) - len(good))
    return good


class InferenceRunner:
    def __init__(self, cfg: Config, models: list[str], seeds: list[int], debug: bool = False):
        self.cfg, self.models, self.seeds = cfg, models, seeds
        self.debug_cases = cfg.inference.debug_max_cases if debug or cfg.inference.debug_max_cases else None
        self.paths = cfg.paths.for_debug() if self.debug_cases else cfg.paths
        self.ckpts = [(m, s) for m in models for s in seeds]
        self.conditions = cfg.conditions
        self.sim = ArtifactSimulator.from_config(cfg)
        self.deadline = Deadline(cfg.inference.time_limit_hours, cfg.inference.safety_margin_s)
        self.device = self._pick_device()
        self.n_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if cfg.training.data_parallel is False or str(cfg.training.data_parallel).lower() in ("off", "false", "no"):
            self.n_gpu = min(self.n_gpu, 1)

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _discover(self) -> dict:
        runs = {}
        for ck in self.ckpts:
            d = self.cfg.paths.run_dir(*ck)
            for f in ("best.pt", "summary.json", "config.json", "val_cases.csv"):
                if not (d / f).exists():
                    raise SystemExit(f"missing {d / f}; train {ck[0]} seed {ck[1]} first (simulated-mri-artifact-stress-test train)")
            cfg_run = json.load(open(d / "config.json"))["config"]
            if list(cfg_run.get("patch", self.cfg.training.patch_size)) != list(self.cfg.training.patch_size):
                raise SystemExit(f"{d}: patch {cfg_run.get('patch')} differs from the configured patch")
            runs[ck] = d
        return runs

    def _amp(self, run_dir: Path):
        name = json.load(open(run_dir / "config.json"))["config"].get("amp_dtype", "torch.float16")
        if self.device.type != "cuda" or "off" in name:
            return None
        dt = torch.bfloat16 if "bfloat16" in name else torch.float16
        if dt == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            log.info("%s: bf16 not supported on this GPU, using fp16", run_dir)
            dt = torch.float16
        return dt

    def _load_models(self, runs: dict) -> dict:
        models = {}
        for ck in self.ckpts:
            core, data = load_checkpoint(self.cfg, ck[0], runs[ck] / "best.pt", self.device)
            net = nn.DataParallel(core) if self.n_gpu > 1 else core
            models[ck] = {"core": core, "net": net, "amp": self._amp(runs[ck]), "epoch": data.get("epoch")}
            log.info("loaded %s/seed%d (epoch %s, amp %s)", ck[0], ck[1], data.get("epoch"), models[ck]["amp"])
            del data
        free_gpu()
        return models

    @torch.no_grad()
    def _predict(self, net, x: np.ndarray, amp) -> np.ndarray:
        inf = self.cfg.inference
        t = torch.from_numpy(x)[None, None].to(self.device)
        ctx = torch.autocast(self.device.type, dtype=amp) if amp is not None else contextlib.nullcontext()
        with ctx:
            logits = sliding_window_inference(
                t, tuple(self.cfg.training.patch_size), inf.sw_batch_size, net, overlap=inf.sw_overlap, mode="gaussian"
            )
        return (torch.sigmoid(logits.float()) > inf.threshold)[0, 0].to(torch.uint8).cpu().numpy()

    def _loader(self, ds) -> DataLoader:
        workers = self.cfg.workers(self.cfg.inference.loader_workers)
        kw = dict(batch_size=None, shuffle=False, num_workers=workers, collate_fn=_keep)
        if workers > 0:
            kw.update(prefetch_factor=1, persistent_workers=False)
        return DataLoader(ds, **kw)

    def _pool(self) -> ProcessPoolExecutor:
        n = max(1, self.cfg.workers(self.cfg.inference.metric_workers))
        return ProcessPoolExecutor(n, mp_context=mp.get_context("fork"))

    def _job(self, item: dict, preds: dict, saves) -> dict:
        return {
            "preds": preds,
            "gt_c": item["gt_c"],
            "gt_native": item["gt_native"],
            "lo": item["lo"],
            "hi": item["hi"],
            "shape": item["shape"],
            "spacing": item["spacing"],
            "affine": item["affine"],
            "saves": saves,
            "nsd_tol": self.cfg.inference.nsd_tolerance_mm,
        }

    def sanity_check(self, splits: dict, runs: dict, models: dict, pool) -> bool:
        """The inference code must reproduce the validation Dice recorded at the end of training."""
        path = self.paths.sanity
        done = pd.read_csv(path) if path.exists() else pd.DataFrame()
        key = lambda d: set(zip(d.get("model", []), d.get("seed", [])))  # noqa: E731
        if len(done) and all(ck in key(done[done["pass"].astype(bool)]) for ck in self.ckpts):
            log.info("validation reproduction already passed for all selected checkpoints")
            return True
        cases = sorted(splits["val"])[: self.debug_cases] if self.debug_cases else sorted(splits["val"])
        rows = []
        ds = CaseSet(self.cfg, cases, "train", None, [("clean", 0)], self.paths, self.sim)
        for item in tqdm(self._loader(ds), total=len(cases), desc="validation reproduction"):
            x = item["vols"][0]["x"]
            preds = {ck: self._predict(models[ck]["net"], x, models[ck]["amp"]) for ck in self.ckpts}
            m = pool.submit(_metric_job, self._job(item, preds, None)).result()
            for ck in self.ckpts:
                rows.append({"model": ck[0], "seed": ck[1], "case_id": item["case"], **m[ck]})
        df = pd.DataFrame(rows)
        rep = []
        tol = self.cfg.inference.sanity_tolerance
        for ck in self.ckpts:
            summ = json.load(open(runs[ck] / "summary.json"))
            tr = pd.read_csv(runs[ck] / "val_cases.csv")
            tr = tr[tr.epoch == summ["best_epoch"]].drop_duplicates("case", keep="last").set_index("case").dice
            got = df[(df.model == ck[0]) & (df.seed == ck[1])].set_index("case_id").dsc
            ref = tr.loc[got.index]
            diff = abs(float(got.mean()) - float(ref.mean()))
            rep.append(
                {
                    "model": ck[0],
                    "seed": ck[1],
                    "best_epoch": summ["best_epoch"],
                    "training_best_val_dice": summ["best_val_dice"],
                    "training_val_dice_same_cases": float(ref.mean()),
                    "inference_val_dice": float(got.mean()),
                    "abs_diff": diff,
                    "max_case_abs_diff": float((got - ref).abs().max()),
                    "n_cases": len(got),
                    "pass": diff <= tol,
                }
            )
            log.info(
                "validation reproduction %s/seed%d: training %.4f inference %.4f diff %.5f %s",
                ck[0],
                ck[1],
                ref.mean(),
                got.mean(),
                diff,
                "PASS" if diff <= tol else "FAIL",
            )
        new = pd.DataFrame(rep)
        if len(done):
            done = done[~done.apply(lambda r: (r["model"], r["seed"]) in key(new), axis=1)]
            new = pd.concat([done, new], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False)
        free_gpu()
        return all(r["pass"] for r in rep)

    def run_test(self, splits: dict, deg_cfg: dict, models: dict, pool) -> bool:
        paths, inf = self.paths, self.cfg.inference
        cases = sorted(splits["test"])[: self.debug_cases] if self.debug_cases else sorted(splits["test"])
        done = {ck: _load_done(paths.result_csv(*ck), len(self.conditions)) for ck in self.ckpts}
        todo = [c for c in cases if any(c not in done[ck] for ck in self.ckpts)]
        log.info(
            "test: %d cases x %d conditions x %d checkpoints; %d cases done, %d to do",
            len(cases),
            len(self.conditions),
            len(self.ckpts),
            len(cases) - len(todo),
            len(todo),
        )
        if not todo:
            return True
        ds = CaseSet(self.cfg, todo, "test", deg_cfg, self.conditions, paths, self.sim, inf.save_degraded_volumes)
        pending, finished, times = None, True, []
        pbar = tqdm(total=len(todo), desc="test")
        for item in self._loader(ds):
            t0 = time.time()
            case = item["case"]
            cks = [ck for ck in self.ckpts if case not in done[ck]]
            futs, cache = [], {}
            for v in item["vols"]:
                key = (v["artifact"], v["param"])
                if key not in cache:
                    cache[key] = {ck: self._predict(models[ck]["net"], v["x"], models[ck]["amp"]) for ck in cks}
                v["x"] = None
                saves = (
                    {ck: str(paths.mask(ck[0], v["artifact"], v["level"], ck[1], case)) for ck in cks}
                    if inf.save_masks
                    else None
                )
                futs.append((v, pool.submit(_metric_job, self._job(item, cache[key], saves))))
            if pending is not None:
                self._write_case(pending)
            pending = (item, cks, futs)
            times.append(time.time() - t0)
            pbar.update(1)
            avg = float(np.mean(times[-5:]))
            pbar.set_postfix(case=case, s_per_case=f"{avg:.0f}", eta_h=f"{avg * (len(todo) - len(times)) / 3600:.1f}")
            if self.deadline.should_stop(2 * avg):
                finished = False
                log.info("time limit reached: results saved, rerun the same command to continue")
                break
        if pending is not None:
            self._write_case(pending)
        pbar.close()
        free_gpu()
        if finished:
            finished = all(
                len(_load_done(paths.result_csv(*ck), len(self.conditions))) == len(cases) for ck in self.ckpts
            )
        return finished

    def _write_case(self, pending) -> None:
        item, cks, futs = pending
        rows = {ck: [] for ck in cks}
        ssim_rows = []
        for v, fut in futs:
            m = fut.result()
            meta = {
                "case_id": item["case"],
                "artifact": v["artifact"],
                "level": v["level"],
                "param": v["param"],
                "capped": int(v["capped"]),
                "ssim": v["ssim"],
            }
            for ck in cks:
                rows[ck].append({**meta, **m[ck]})
            ssim_rows.append({**meta, "target_ssim": self.cfg.target_ssim(v["level"]) if v["level"] else 1.0})
        for ck in cks:
            append_rows(self.paths.result_csv(*ck), rows[ck], ROW_KEYS)
        self._append_ssim(ssim_rows, item["case"])

    def _append_ssim(self, rows: list[dict], case: str) -> None:
        path = self.paths.ssim_test
        if path.exists() and case in set(pd.read_csv(path, usecols=["case_id"]).case_id):
            return
        append_rows(path, rows, SSIM_KEYS)

    def complexity(self, models: dict) -> None:
        path = self.paths.complexity
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model"])
        rows = []
        for m in self.models:
            if m in set(old.model):
                continue
            core = models[(m, self.seeds[0])]["core"]
            rec = {
                "model": m,
                "label": self.cfg.model_label(m),
                "params": int(sum(p.numel() for p in core.parameters())),
                "input": "1x1x" + "x".join(map(str, self.cfg.training.patch_size)),
            }
            try:
                from torch.utils.flop_counter import FlopCounterMode

                x = torch.zeros(1, 1, *self.cfg.training.patch_size, device=self.device)
                with torch.no_grad(), FlopCounterMode(display=False) as fc:
                    core(x)
                rec["gflops"] = fc.get_total_flops() / 1e9
                del x
            except Exception as e:  # noqa: BLE001
                rec["gflops"], rec["error"] = None, repr(e)
            free_gpu()
            rows.append(rec)
            log.info("complexity %s: %.1fM parameters, %s GFLOPs", rec["label"], rec["params"] / 1e6, rec.get("gflops"))
        if rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    def run(self) -> bool:
        cfg, paths = self.cfg, self.paths
        paths.ensure()
        splits = get_splits(cfg)
        runs = self._discover()
        deg_cfg = calibrate(cfg)
        frequency_footprint(cfg, deg_cfg)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info("inference %s seeds %s on %s | output %s", self.models, self.seeds, self.device, paths.out)
        models = self._load_models(runs)
        finished = False
        try:
            self.complexity(models)
            with self._pool() as pool:
                if not self.sanity_check(splits, runs, models, pool):
                    msg = (
                        f"validation reproduction failed (tolerance {cfg.inference.sanity_tolerance}); "
                        f"see {paths.sanity}"
                    )
                    if cfg.inference.strict_sanity and not self.debug_cases:
                        raise SystemExit(msg)
                    log.info("warning: %s", msg)
                finished = self.run_test(splits, deg_cfg, models, pool)
        finally:
            for v in models.values():
                v["core"].to("cpu")
            models.clear()
            free_gpu()
        if not finished:
            log.info("test inference incomplete; rerun the same command to continue")
        return finished


def run_inference(cfg: Config, models: list[str], seeds: list[int], debug: bool = False) -> bool:
    models = [canonical_model(m) for m in models]
    return InferenceRunner(cfg, models, seeds, debug).run()
