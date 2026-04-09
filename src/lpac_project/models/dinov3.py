from __future__ import annotations

import torch.nn as nn
import timm


DINOv3_VITB16_DISTILLED_TIMM_ID = "vit_base_patch16_dinov3"


def build_dinov3_vitb16_distilled(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Build DINOv3 ViT-B/16 (distilled, LVD-1689M pretrain in timm)."""

    model = timm.create_model(
        DINOv3_VITB16_DISTILLED_TIMM_ID,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if dropout > 0.0:
        model.head = nn.Sequential(nn.Dropout(p=dropout), model.head)

    return model


def get_dinov3_vitb16_distilled_normalization(pretrained: bool) -> dict[str, list[float]] | None:
    if not pretrained:
        return None

    # Matches timm pretrained_cfg for vit_base_patch16_dinov3
    return {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
