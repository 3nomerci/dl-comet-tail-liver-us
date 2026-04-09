'''
Model registry for building models based on configuration.
'''

from __future__ import annotations

from typing import Any

from torch import nn

from .dinov3 import (
    build_dinov3_vitb16_distilled,
    get_dinov3_vitb16_distilled_normalization,
)
from .resnet import build_resnet18, get_resnet18_normalization


def build_model(model_cfg: dict[str, Any]):
    name = model_cfg["name"].lower()

    if name == "resnet18":
        return build_resnet18(
            num_classes=int(model_cfg.get("num_classes", 2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )

    if name == "dinov3_vitb16_distilled":
        return build_dinov3_vitb16_distilled(
            num_classes=int(model_cfg.get("num_classes", 2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )

    raise ValueError(f"Unsupported model name: {name}")


def get_model_normalization(model_cfg: dict[str, Any]) -> dict[str, list[float]] | None:
    name = model_cfg["name"].lower()

    if name == "resnet18":
        return get_resnet18_normalization(pretrained=bool(model_cfg.get("pretrained", True)))

    if name == "dinov3_vitb16_distilled":
        return get_dinov3_vitb16_distilled_normalization(
            pretrained=bool(model_cfg.get("pretrained", True))
        )

    raise ValueError(f"Unsupported model name: {name}")


def get_model_head_module(model: nn.Module, model_cfg: dict[str, Any]) -> nn.Module:
    name = model_cfg["name"].lower()

    if name == "resnet18":
        return model.fc

    if name == "dinov3_vitb16_distilled":
        return model.head

    raise ValueError(f"Unsupported model name for head warmup: {name}")
