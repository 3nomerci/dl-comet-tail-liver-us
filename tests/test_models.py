import torch

from lpac_project.models.registry import build_model


def test_resnet18_forward_shape():
    model = build_model(
        {
            "name": "resnet18",
            "num_classes": 2,
            "pretrained": False,
            "dropout": 0.0,
        }
    )

    x = torch.rand(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 2)