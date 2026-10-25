from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import UNETR, DynUNet, SwinUNETR

from simulated_mri_artifact_stress_test.config import Config, canonical_model

REGISTRY = {"dynunet": DynUNet, "unetr": UNETR, "swinunetr": SwinUNETR}


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def build_model(cfg: Config, name: str) -> torch.nn.Module:
    kwargs = _plain(cfg.model_cfg(name).kwargs)
    if name == "dynunet":
        act = kwargs["act_name"]
        kwargs["act_name"] = (act[0], act[1])
    if name == "unetr":
        kwargs["img_size"] = tuple(cfg.training.patch_size)
    if name == "swinunetr":
        kwargs["depths"] = tuple(kwargs["depths"])
        kwargs["num_heads"] = tuple(kwargs["num_heads"])
    return REGISTRY[name](**kwargs)


def load_checkpoint(cfg: Config, name: str, path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved = canonical_model(data.get("model_name", name))
    if saved != name:
        raise SystemExit(f"{path}: checkpoint holds {saved}, expected {name}")
    net = build_model(cfg, name)
    net.load_state_dict(data["model"], strict=True)
    return net.to(device).eval(), data
