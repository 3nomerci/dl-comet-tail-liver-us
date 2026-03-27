from __future__ import annotations

import torch
from tqdm import tqdm

from .metrics import classification_metrics


def run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler=None,
    use_amp: bool = True,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    amp_enabled = use_amp and device.type == "cuda"

    progress = tqdm(loader, desc="train", leave=False)

    for images, labels, _patients in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_samples

    return metrics


@torch.no_grad()
def run_eval_epoch(
    model,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    amp_enabled = use_amp and device.type == "cuda"

    progress = tqdm(loader, desc="eval", leave=False)

    for images, labels, _patients in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_samples

    return metrics