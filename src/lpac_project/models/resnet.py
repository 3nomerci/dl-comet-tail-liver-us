from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = True,dropout: float = 0.0,) -> nn.Module:
    '''Builds a ResNet-18 model, replacing the final classification head with a custom one'''
    
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    head = nn.Linear(in_features, num_classes)

    if dropout > 0.0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), head)
    else:
        model.fc = head

    return model


def get_resnet18_normalization(pretrained: bool) -> dict[str, list[float]] | None:
    if not pretrained:
        return None

    return {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }