import torch
from torch import nn
from torch.utils.data import DataLoader

from lpac_project.data import PackedPatientDataset
from lpac_project.engine import run_train_epoch


def make_pack():
    n = 8
    return {
        "images": torch.rand(n, 3, 64, 64),
        "labels": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
        "patients": torch.arange(n, dtype=torch.long),
        "paths": [f"img_{i}.npy" for i in range(n)],
        "image_size": 64,
    }


def test_single_train_epoch_runs():
    dataset = PackedPatientDataset(make_pack())
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, 2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    metrics = run_train_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        scaler=None,
        use_amp=False,
    )

    assert "loss" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0