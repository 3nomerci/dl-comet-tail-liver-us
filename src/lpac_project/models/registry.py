'''
Model registry for building models based on configuration.
'''

from __future__ import annotations

from typing import Any

from .resnet import build_resnet18, get_resnet18_normalization


def build_model(model_cfg: dict[str, Any]):
    name = model_cfg["name"].lower()

    if name == "resnet18":
        return build_resnet18(
            num_classes=int(model_cfg.get("num_classes", 2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )

    raise ValueError(f"Unsupported model name: {name}")


def get_model_normalization(model_cfg: dict[str, Any]) -> dict[str, list[float]] | None:
    name = model_cfg["name"].lower()

    if name == "resnet18":
        return get_resnet18_normalization(pretrained=bool(model_cfg.get("pretrained", True)))

    raise ValueError(f"Unsupported model name: {name}")