import torch

from lpac_project.models.registry import build_model, get_model_head_module
from lpac_project.train import set_requires_grad


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


def test_resnet18_head_only_warmup_trainability_pattern():
    # Build without downloading external weights, then validate the warmup
    # trainability contract used when pretrained mode is enabled in training.
    model = build_model(
        {
            "name": "resnet18",
            "num_classes": 2,
            "pretrained": False,
            "dropout": 0.0,
        }
    )
    warmup_cfg = {
        "name": "resnet18",
        "num_classes": 2,
        "pretrained": True,
        "dropout": 0.0,
    }

    head_module = get_model_head_module(model, warmup_cfg)

    set_requires_grad(model, enabled=False)
    set_requires_grad(head_module, enabled=True)

    head_param_ids = {id(param) for param in head_module.parameters()}
    trainable_param_ids = {id(param) for param in model.parameters() if param.requires_grad}

    assert head_param_ids
    assert trainable_param_ids == head_param_ids

    set_requires_grad(model, enabled=True)
    assert all(param.requires_grad for param in model.parameters())