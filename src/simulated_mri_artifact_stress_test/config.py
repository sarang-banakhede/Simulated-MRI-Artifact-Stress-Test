from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

LEGACY_MODEL_NAMES = {"unet3d": "dynunet"}


def canonical_model(name: str) -> str:
    return LEGACY_MODEL_NAMES.get(name, name)


class Box(dict):
    """Dictionary with attribute access, used for the nested configuration."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _box(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Box({k: _box(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_box(v) for v in obj]
    return obj


def _coerce(text: str) -> Any:
    return yaml.safe_load(text)


def _set_path(raw: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = raw
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


class Paths:
    """Every file the pipeline reads or writes, derived from paths.output_root."""

    def __init__(self, data_root: Path, out: Path, cache: Path):
        self.data = data_root
        self.out = out
        self.cache = cache
        self.splits = out / "splits.json"
        self.case_split = out / "case_split.csv"
        self.training = out / "training"
        self.calibration = out / "calibration"
        self.degradation_config = self.calibration / "degradation_config.json"
        self.inference = out / "inference"
        self.predictions = self.inference / "predictions"
        self.degraded = self.inference / "degraded_volumes"
        self.results = self.inference / "results"
        self.ssim_test = self.results / "ssim_test.csv"
        self.results_long = self.results / "results_test_long.csv"
        self.sanity = self.results / "val_reproduction.csv"
        self.complexity = self.results / "model_complexity.csv"
        self.analysis = out / "analysis"
        self.summary = self.analysis
        self.tables = self.analysis / "tables"
        self.post = self.analysis / "post_masks"
        self.reader = out / "reader_study"
        self.logs = out / "logs"

    def for_debug(self) -> "Paths":
        """Same inputs (weights, splits, calibration), but every output goes to <output_root>/debug."""
        d = Paths(self.data, self.out / "debug", self.cache)
        d.splits, d.case_split, d.training = self.splits, self.case_split, self.training
        d.calibration = self.calibration
        d.degradation_config = self.degradation_config
        return d

    def run_dir(self, model: str, seed: int) -> Path:
        return self.training / model / f"seed{seed}"

    def mask(self, model: str, artifact: str, level: int, seed: int, case: str) -> Path:
        cond = "clean_L0" if level == 0 else f"{artifact}_L{level}"
        return self.predictions / model / cond / f"seed{seed}" / f"{case}.nii.gz"

    def result_csv(self, model: str, seed: int) -> Path:
        return self.results / model / f"seed{seed}.csv"

    def ensure(self) -> None:
        for p in (
            self.out,
            self.training,
            self.calibration,
            self.results,
            self.tables,
            self.post,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self, raw: dict, base: Path, source: Path | None = None):
        self.raw = raw
        self.base = base
        self.source = source
        self._box = _box(copy.deepcopy(raw))
        p = self._box.paths
        data = self._resolve(p.data_root)
        out = self._resolve(p.output_root)
        cache = self._resolve(p.cache_dir) if p.cache_dir else Path(tempfile.gettempdir()) / "simulated_mri_artifact_stress_test_cache"
        self.paths = Paths(data, out, cache)

    def _resolve(self, value: str | Path) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base / p).resolve()

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_") or key in ("raw", "base", "source", "paths"):
            raise AttributeError(key)
        return getattr(self._box, key)

    def __getitem__(self, key: str) -> Any:
        return self._box[key]

    @property
    def models(self) -> list[str]:
        return list(self._box.experiment.models)

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self._box.experiment.seeds]

    @property
    def art(self) -> Box:
        return self._box.artifacts

    @property
    def artifacts(self) -> list[str]:
        return list(self._box.artifacts.names)

    @property
    def levels(self) -> list[int]:
        return [int(l) for l in self._box.artifacts.levels]

    @property
    def primary_levels(self) -> list[int]:
        return [int(l) for l in self._box.artifacts.primary_levels]

    def target_ssim(self, level: int) -> float:
        return round(1.0 - self._box.artifacts.target_ssim_step * level, 2)

    @property
    def conditions(self) -> list[tuple[str, int]]:
        return [("clean", 0)] + [(a, l) for a in self.artifacts for l in self.levels]

    def model_cfg(self, model: str) -> Box:
        return self._box.models[model]

    def model_label(self, model: str) -> str:
        return self._box.models[model].label

    def select(self, models: list[str] | None, seeds: list[int] | None) -> tuple[list[str], list[int]]:
        ms = list(models) if models else self.models
        ss = [int(s) for s in seeds] if seeds is not None and len(seeds) else self.seeds
        unknown = [m for m in ms if m not in self._box.models]
        if unknown:
            raise SystemExit(f"unknown model(s) {unknown}; configured: {list(self._box.models)}")
        return ms, ss

    def workers(self, value: Any) -> int:
        if value in (None, "auto"):
            return max(1, (os.cpu_count() or 2) // 2)
        return int(value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        _set_path(raw, k.strip(), _coerce(v))
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    return Config(raw, base, path)
