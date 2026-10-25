from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger("simulated_mri_artifact_stress_test")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    log.handlers.clear()
    log.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)


def atomic_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def append_rows(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def truncate_csv(path: Path, max_epoch: int) -> None:
    if not Path(path).exists():
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    fields = list(rows[0].keys())
    keep = [r for r in rows if int(r["epoch"]) <= max_epoch]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class Deadline:
    """Wall-clock budget so long runs can stop cleanly and be resumed by rerunning the same command."""

    def __init__(self, hours: float | None, margin_s: float = 600.0):
        self.start = time.time()
        self.limit = None if hours is None else float(hours) * 3600.0
        self.margin = margin_s

    def left(self) -> float:
        return float("inf") if self.limit is None else self.limit - (time.time() - self.start)

    def should_stop(self, next_step_s: float) -> bool:
        return self.left() < next_step_s + self.margin


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def environment_info() -> dict:
    import numpy
    import scipy

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import monai
        import torch

        info.update(
            torch=torch.__version__,
            monai=monai.__version__,
            cuda=torch.version.cuda,
            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )
    except ImportError:
        pass
    return info
