from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import monai
import numpy as np
import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    EnsureTyped,
    Lambdad,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandScaleIntensityFixedMeand,
    SpatialPadd,
)
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from torchmetrics.functional.classification import binary_auroc
from tqdm.auto import tqdm

from simulated_mri_artifact_stress_test.config import Config
from simulated_mri_artifact_stress_test.data.splits import get_splits
from simulated_mri_artifact_stress_test.data.training_cache import build_training_cache
from simulated_mri_artifact_stress_test.models import build_model
from simulated_mri_artifact_stress_test.utils import (
    Deadline,
    append_rows,
    atomic_json,
    environment_info,
    free_gpu,
    log,
    truncate_csv,
)

CASE_KEYS = ["dice", "iou", "precision", "recall", "hd95_mm", "nsd", "auroc", "gt_volume_ml", "pred_volume_ml"]


def _atomic_save(obj, path: Path) -> None:
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _items(cases, cache: Path, with_mask: bool) -> list[dict]:
    out = []
    for c in cases:
        d = {"image": str(cache / f"{c}_img.npy"), "label": str(cache / f"{c}_seg.npy"), "case": c}
        if with_mask:
            d["mask"] = str(cache / f"{c}_mask.npy")
        out.append(d)
    return out


def train_transforms(cfg: Config) -> Compose:
    t, aug = cfg.training, cfg.training.augmentation
    patch = tuple(t.patch_size)
    rot = math.radians(aug.rotate_deg)
    return Compose(
        [
            Lambdad(keys=["image", "label"], func=np.load),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
            EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=False),
            SpatialPadd(keys=["image", "label"], spatial_size=patch, mode="constant"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch,
                pos=t.pos_neg[0],
                neg=t.pos_neg[1],
                num_samples=1,
            ),
            EnsureTyped(keys=["label"], dtype=torch.float32, track_meta=False),
            RandAffined(
                keys=["image", "label"],
                prob=aug.affine_prob,
                rotate_range=(rot,) * 3,
                scale_range=(tuple(aug.scale_range),) * 3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=patch,
            ),
            RandFlipd(keys=["image", "label"], prob=aug.flip_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=aug.flip_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=aug.flip_prob, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=aug.intensity_factor, prob=aug.intensity_prob),
            RandScaleIntensityFixedMeand(
                keys=["image"], factors=aug.intensity_factor, preserve_range=True, prob=aug.intensity_prob
            ),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False),
        ]
    )


def val_transforms() -> Compose:
    return Compose(
        [
            Lambdad(keys=["image", "label", "mask"], func=np.load),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
            EnsureTyped(keys=["label", "mask"], dtype=torch.uint8, track_meta=False),
        ]
    )


def _surface(pred, gt, spacing, tol):
    idx = np.argwhere(pred | gt)
    lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, pred.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    p, g = pred[sl], gt[sl]
    st = generate_binary_structure(3, 1)
    sp = p & ~binary_erosion(p, structure=st)
    sg = g & ~binary_erosion(g, structure=st)
    d_pg = distance_transform_edt(~sg, sampling=spacing)[sp]
    d_gp = distance_transform_edt(~sp, sampling=spacing)[sg]
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    allv = np.concatenate([d_pg, d_gp])
    return hd95, float((allv <= tol).sum() / allv.size)


def _case_metrics(prob, pred, gt, mask, spacing, tol):
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    gt_n, pr_n = tp + fn, tp + fp
    vox_ml = float(np.prod(spacing)) / 1000.0
    diag = float(np.sqrt(sum((s * sp) ** 2 for s, sp in zip(gt.shape[-3:], spacing))))
    if gt_n == 0 and pr_n == 0:
        dice = iou = precision = recall = nsd = 1.0
        hd95 = 0.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / pr_n if pr_n > 0 else 0.0
        recall = tp / gt_n if gt_n > 0 else 0.0
        if gt_n > 0 and pr_n > 0:
            hd95, nsd = _surface(pred.cpu().numpy(), gt.cpu().numpy(), spacing, tol)
        else:
            hd95, nsd = diag, 0.0
    t_m = gt[mask].int()
    auroc = (
        binary_auroc(prob[mask], t_m, validate_args=False).item()
        if 0 < t_m.sum().item() < t_m.numel()
        else float("nan")
    )
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95_mm": hd95,
        "nsd": nsd,
        "auroc": auroc,
        "gt_volume_ml": gt_n * vox_ml,
        "pred_volume_ml": pr_n * vox_ml,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "empty_pred": int(pr_n == 0),
    }


def _summarize(rows: list[dict]) -> dict:
    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    def agg(f, v):
        return float(f(v)) if np.isfinite(v).any() else float("nan")

    dice, hd95 = col("dice"), col("hd95_mm")
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {
        "val_loss": agg(np.nanmean, col("val_loss")),
        "dice_mean": agg(np.nanmean, dice),
        "dice_median": agg(np.nanmedian, dice),
        "dice_std": agg(np.nanstd, dice),
        "iou_mean": agg(np.nanmean, col("iou")),
        "precision_mean": agg(np.nanmean, col("precision")),
        "recall_mean": agg(np.nanmean, col("recall")),
        "hd95_mm_mean": agg(np.nanmean, hd95),
        "hd95_mm_median": agg(np.nanmedian, hd95),
        "nsd_mean": agg(np.nanmean, col("nsd")),
        "auroc_mean": agg(np.nanmean, col("auroc")),
        "global_dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan"),
        "n_empty_pred": int(sum(r["empty_pred"] for r in rows)),
    }


def _cycle(loader):
    while True:
        yield from loader


class Trainer:
    def __init__(self, cfg: Config, model: str):
        self.cfg, self.model = cfg, model
        t = cfg.training
        self.patch = tuple(t.patch_size)
        self.batch_size = cfg.model_cfg(model).batch_size
        self.total_iters = t.iters_per_epoch * t.epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.workers = min(8, os.cpu_count() or 1) if t.num_workers == "auto" else int(t.num_workers)
        self.deadline = Deadline(t.time_limit_hours, t.safety_margin_s)
        self.amp = self._amp_dtype()
        self.n_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if cfg.training.data_parallel is False or str(cfg.training.data_parallel).lower() in ("off", "false", "no"):
            self.n_gpu = min(self.n_gpu, 1)

    def _amp_dtype(self):
        mode = self.cfg.training.amp
        if self.device.type != "cuda" or mode == "off":
            return None
        if mode == "bf16":
            return torch.bfloat16
        if mode == "fp16":
            return torch.float16
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    def _lr_lambda(self, it: int) -> float:
        warm = int(self.total_iters * self.cfg.training.warmup_fraction)
        if it < warm:
            return (it + 1) / warm
        prog = (it - warm) / max(1, self.total_iters - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        monai.utils.set_determinism(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    @staticmethod
    def _rng_state() -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    @staticmethod
    def _set_rng_state(s: dict) -> None:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch"].cpu())
        states = [t.cpu() for t in s["cuda"]]
        if states and len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)

    def _config_dict(self, seed: int, n_params: int) -> dict:
        t, inf = self.cfg.training, self.cfg.inference
        return {
            "model": self.model,
            "seed": seed,
            "params": n_params,
            "patch": list(self.patch),
            "batch_size": self.batch_size,
            "data_parallel_gpus": self.n_gpu,
            "total_iters": self.total_iters,
            "iters_per_epoch": t.iters_per_epoch,
            "epochs": t.epochs,
            "val_every_epochs": t.val_every_epochs,
            "optimizer": "AdamW",
            "lr": t.lr,
            "weight_decay": t.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "schedule": "linear warmup + cosine to 0",
            "warmup_fraction": t.warmup_fraction,
            "grad_clip_norm": t.grad_clip,
            "loss": "DiceLoss(sigmoid) + BCEWithLogits, equal weight",
            "amp_dtype": str(self.amp) if self.amp is not None else "off",
            "pos_neg": list(t.pos_neg),
            "sw_batch": t.sw_batch_size,
            "sw_overlap": inf.sw_overlap,
            "sw_mode": "gaussian",
            "threshold": inf.threshold,
            "nsd_tolerance_mm": inf.nsd_tolerance_mm,
            "selection_metric": "val dice_mean",
            "augmentation": json.loads(json.dumps(dict(t.augmentation))),
            "preprocessing": "RAS, FLAIR channel, crop to brain mask, z-score in mask, 0 outside, pad to patch",
        }

    @torch.no_grad()
    def _validate(self, net, loader, dice_loss, bce_loss) -> list[dict]:
        t, inf = self.cfg.training, self.cfg.inference
        net.eval()
        rows = []
        for b in tqdm(loader, desc="val", leave=False):
            x = b["image"].to(self.device, non_blocking=True)
            y = b["label"].to(self.device, non_blocking=True)
            m = b["mask"].to(self.device, non_blocking=True).bool()
            ctx = (
                torch.autocast(self.device.type, dtype=self.amp)
                if self.amp is not None
                else torch.autocast("cpu", enabled=False)
            )
            with ctx:
                logits = sliding_window_inference(
                    x, self.patch, t.sw_batch_size, net, overlap=inf.sw_overlap, mode="gaussian"
                )
            logits = logits.float()
            yf = y.float()
            vloss = (dice_loss(logits, yf) + bce_loss(logits, yf)).item()
            prob = torch.sigmoid(logits)[0, 0]
            pred = prob > inf.threshold
            gt = y[0, 0].bool()
            r = _case_metrics(prob, pred, gt, m[0, 0], (1.0, 1.0, 1.0), inf.nsd_tolerance_mm)
            r["val_loss"], r["case"] = vloss, b["case"][0]
            rows.append(r)
        free_gpu()
        net.train()
        return rows

    def fit(self, seeds: list[int]) -> bool:
        cfg = self.cfg
        splits = get_splits(cfg)
        cache = build_training_cache(cfg)
        workers = self.workers
        train_ds = CacheDataset(
            _items(splits["train"], cache, False), transform=train_transforms(cfg), cache_rate=1.0, num_workers=workers
        )
        val_ds = CacheDataset(
            _items(splits["val"], cache, True), transform=val_transforms(), cache_rate=1.0, num_workers=workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=min(2, workers), pin_memory=self.device.type == "cuda"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        for seed in seeds:
            try:
                finished = self._train_run(seed, train_ds, val_loader)
            finally:
                free_gpu()
            if not finished:
                log.info("stopped at the time limit; rerun the same command to resume")
                return False
        return True

    def _train_run(self, seed: int, train_ds, val_loader) -> bool:
        cfg, t = self.cfg, self.cfg.training
        run_dir = cfg.paths.run_dir(self.model, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "DONE").exists():
            log.info("skip %s seed%d: already done", self.model, seed)
            return True
        self._seed_everything(seed)
        train_ds.transform.set_random_state(seed=seed)
        core = build_model(cfg, self.model).to(self.device)
        n_params = sum(p.numel() for p in core.parameters())
        net = nn.DataParallel(core) if self.n_gpu > 1 else core
        conf = self._config_dict(seed, n_params)
        atomic_json({"config": conf, "env": environment_info()}, run_dir / "config.json")
        opt = torch.optim.AdamW(core.parameters(), lr=t.lr, weight_decay=t.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, self._lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.amp == torch.float16))
        dice_loss = DiceLoss(sigmoid=True, include_background=True, smooth_nr=1e-5, smooth_dr=1e-5)
        bce_loss = nn.BCEWithLogitsLoss()

        start_it, best_dice, best_epoch, train_time = 0, -1.0, -1, 0.0
        best_metrics = last_metrics = None
        resume_p = run_dir / "resume.pt"
        ipe = t.iters_per_epoch
        if resume_p.exists():
            s = torch.load(resume_p, map_location=self.device, weights_only=False)
            core.load_state_dict(s["model"])
            opt.load_state_dict(s["optimizer"])
            sched.load_state_dict(s["scheduler"])
            scaler.load_state_dict(s["scaler"])
            start_it, best_dice, best_epoch = s["iteration"], s["best_dice"], s["best_epoch"]
            best_metrics, last_metrics = s.get("best_metrics"), s.get("last_metrics")
            train_time = s.get("train_time_s", 0.0)
            self._set_rng_state(s["rng"])
            for name in ("train_log.csv", "val_log.csv", "val_cases.csv"):
                truncate_csv(run_dir / name, start_it // ipe)
            log.info(
                "resume %s seed%d from epoch %d (best dice %.4f at %d)",
                self.model,
                seed,
                start_it // ipe,
                best_dice,
                best_epoch,
            )
            del s
            free_gpu()

        g = torch.Generator()
        g.manual_seed(seed * 100003 + start_it)
        loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.workers > 0,
            generator=g,
        )

        stream = _cycle(loader)
        net.train()
        total_epochs = self.total_iters // ipe
        log.info(
            "start %s seed%d: epochs %d-%d, amp=%s, gpus=%d, params=%.1fM",
            self.model,
            seed,
            start_it // ipe + 1,
            total_epochs,
            self.amp,
            self.n_gpu,
            n_params / 1e6,
        )
        it, last_ep_time, last_val_time, finished = start_it, 0.0, 0.0, True
        pbar = tqdm(total=self.total_iters, initial=start_it, desc=f"{self.model}/seed{seed}", smoothing=0.05)
        while it < self.total_iters:
            epoch = it // ipe + 1
            t0 = time.time()
            acc = {"loss_total": 0.0, "loss_dice": 0.0, "loss_bce": 0.0, "patch_dice": 0.0}
            steps = 0
            for _ in range(ipe):
                batch = next(stream)
                x = batch["image"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                ctx = (
                    torch.autocast(self.device.type, dtype=self.amp)
                    if self.amp is not None
                    else torch.autocast("cpu", enabled=False)
                )
                with ctx:
                    logits = net(x)
                logits = logits.float()
                ld, lb = dice_loss(logits, y), bce_loss(logits, y)
                loss = ld + lb
                if not torch.isfinite(loss):
                    sched.step()
                    it += 1
                    pbar.update(1)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), t.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                with torch.no_grad():
                    p, yb = logits > 0, y > 0.5
                    den = p.sum().item() + yb.sum().item()
                    acc["patch_dice"] += 2 * (p & yb).sum().item() / den if den > 0 else 1.0
                acc["loss_total"] += loss.item()
                acc["loss_dice"] += ld.item()
                acc["loss_bce"] += lb.item()
                steps += 1
                it += 1
                pbar.update(1)
                if steps % 10 == 0:
                    pbar.set_postfix(
                        ep=f"{epoch}/{total_epochs}",
                        loss=f"{acc['loss_total'] / steps:.4f}",
                        dice=f"{acc['patch_dice'] / steps:.3f}",
                    )
            last_ep_time = time.time() - t0
            train_time += last_ep_time
            n = max(1, steps)
            row = {
                "epoch": epoch,
                "loss_total": acc["loss_total"] / n,
                "loss_dice": acc["loss_dice"] / n,
                "loss_bce": acc["loss_bce"] / n,
                "train_patch_dice": acc["patch_dice"] / n,
                "lr": opt.param_groups[0]["lr"],
            }
            append_rows(run_dir / "train_log.csv", [row], list(row))
            free_gpu()
            log.info(
                "%s seed%d epoch %d/%d loss %.4f patch_dice %.4f lr %.2e (%ds)",
                self.model,
                seed,
                epoch,
                total_epochs,
                row["loss_total"],
                row["train_patch_dice"],
                row["lr"],
                last_ep_time,
            )

            if epoch % t.val_every_epochs == 0 or it >= self.total_iters:
                tv = time.time()
                rows = self._validate(net, val_loader, dice_loss, bce_loss)
                summ = _summarize(rows)
                is_best = summ["dice_mean"] > best_dice
                case_rows = [{"epoch": epoch, "case": r["case"], **{k: r[k] for k in CASE_KEYS}} for r in rows]
                append_rows(run_dir / "val_cases.csv", case_rows, list(case_rows[0]))
                last_val_time = time.time() - tv
                vrow = {"epoch": epoch, **summ, "is_best": int(is_best)}
                append_rows(run_dir / "val_log.csv", [vrow], list(vrow))
                last_metrics = {**summ, "epoch": epoch, "iteration": it}
                if is_best:
                    best_dice, best_epoch, best_metrics = summ["dice_mean"], epoch, last_metrics
                    _atomic_save(
                        {
                            "model": core.state_dict(),
                            "model_name": self.model,
                            "seed": seed,
                            "epoch": epoch,
                            "iteration": it,
                            "val_metrics": last_metrics,
                            "config": conf,
                        },
                        run_dir / "best.pt",
                    )
                log.info(
                    "%s seed%d VAL epoch %d: dice %.4f nsd %.4f hd95 %.2f best %.4f@%d%s",
                    self.model,
                    seed,
                    epoch,
                    summ["dice_mean"],
                    summ["nsd_mean"],
                    summ["hd95_mm_mean"],
                    best_dice,
                    best_epoch,
                    " *" if is_best else "",
                )
                free_gpu()

            if it < self.total_iters:
                _atomic_save(
                    {
                        "model": core.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(),
                        "iteration": it,
                        "epoch": epoch,
                        "best_dice": best_dice,
                        "best_epoch": best_epoch,
                        "best_metrics": best_metrics,
                        "last_metrics": last_metrics,
                        "train_time_s": train_time,
                        "rng": self._rng_state(),
                        "model_name": self.model,
                        "seed": seed,
                        "config": conf,
                    },
                    resume_p,
                )
                need = last_ep_time + (last_val_time if (epoch + 1) % t.val_every_epochs == 0 else 0)
                if self.deadline.should_stop(need):
                    log.info("time limit reached after epoch %d, state saved to %s", epoch, resume_p)
                    finished = False
                    break
        pbar.close()

        if finished:
            atomic_json(
                {
                    "model": self.model,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_dice": best_dice,
                    "best_val_metrics": best_metrics,
                    "last_val_metrics": last_metrics,
                    "train_time_h": train_time / 3600,
                },
                run_dir / "summary.json",
            )
            (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            if resume_p.exists():
                resume_p.unlink()
            log.info(
                "done %s seed%d: best dice %.4f at epoch %d, %.2f h",
                self.model,
                seed,
                best_dice,
                best_epoch,
                train_time / 3600,
            )
        del stream, loader, net, core, opt, sched, scaler
        gc.collect()
        free_gpu()
        return finished


def train_models(cfg: Config, models: list[str], seeds: list[int]) -> bool:
    cfg.paths.ensure()
    for m in models:
        if not Trainer(cfg, m).fit(seeds):
            return False
    return True
