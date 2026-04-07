from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    PackedPatientDataset,
    build_tensor_transform,
    limit_indices,
    load_pack,
    patient_split_indices,
    save_split_artifact,
)
from .engine import run_eval_epoch, run_train_epoch
from .models.registry import build_model, get_model_head_module, get_model_normalization
from .utils import (
    append_metrics_row,
    copy_file,
    load_config,
    make_run_dir,
    save_json,
    seed_everything,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
    epochs: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, str]:
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_name = str(scheduler_cfg.get("name", "none")).lower()

    if scheduler_name == "none":
        return None, "none"

    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 2)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )
        return scheduler, "plateau"

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(scheduler_cfg.get("t_max", epochs))),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
        return scheduler, "cosine"

    raise ValueError(
        f"Unsupported scheduler '{scheduler_name}'. "
        "Supported values are: none, plateau, cosine."
    )


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def log_epoch_and_save_checkpoint(
    run_dir: Path,
    metrics_csv: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
    best_val_bal_acc: float,
    config: dict,
    model: nn.Module,
) -> float:
    current_lr = float(optimizer.param_groups[0]["lr"])

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "lr": current_lr,
    }
    append_metrics_row(metrics_csv, row)

    print(
        "train "
        f"loss={train_metrics['loss']:.4f} "
        f"acc={train_metrics['accuracy']:.4f} "
        f"bal_acc={train_metrics['balanced_accuracy']:.4f}"
    )
    print(
        "val   "
        f"loss={val_metrics['loss']:.4f} "
        f"acc={val_metrics['accuracy']:.4f} "
        f"bal_acc={val_metrics['balanced_accuracy']:.4f}"
    )
    print(f"lr={current_lr:.6g}")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, run_dir / "last_model.pt")

    if val_metrics["balanced_accuracy"] >= best_val_bal_acc:
        best_val_bal_acc = val_metrics["balanced_accuracy"]
        torch.save(checkpoint, run_dir / "best_model.pt")

    return best_val_bal_acc


def main():
    args = parse_args()

    config = load_config(args.config)
    run_cfg = config["run"]
    data_cfg = config["data"]
    split_cfg = config.get("split", {})
    model_cfg = config["model"]
    train_cfg = config["train"]

    seed = int(split_cfg["seed"])
    train_fraction = float(split_cfg["train_fraction"])
    val_fraction = float(split_cfg["val_fraction"])
    test_fraction = float(split_cfg["test_fraction"])
    stratify = bool(split_cfg.get("stratify", True))
    split_method = str(split_cfg.get("method", "heuristic_balanced"))
    save_split_artifact_flag = bool(split_cfg.get("save_artifact", True))

    seed_everything(seed)

    device = select_device(args.device)
    run_dir = make_run_dir(run_cfg["output_root"], run_cfg["name"])
    copy_file(args.config, run_dir / "config.toml")

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    pack = load_pack(data_cfg["dataset_path"])

    train_idx, val_idx, test_idx = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        stratify=stratify,
        method=split_method
    )

    if save_split_artifact_flag:
        # NOTE: even if args.smoke is enabled the split artifact is saved with the full indices, 
        # as the smoke mode only limits the indices used in training/validation/test loaders, not the actual split saved.
        save_split_artifact(
            output_path=run_dir / "split.json",
            summary_output_path=run_dir / "split_summary.json",
            patients=pack["patients"],
            labels=pack["labels"],
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            method=split_method,
            stratify=stratify,
        )

    if args.smoke:
        train_idx = limit_indices(train_idx, max_items=100, seed=seed + 1)
        val_idx = limit_indices(val_idx, max_items=100, seed=seed + 2)
        test_idx = limit_indices(test_idx, max_items=100, seed=seed + 3)

    normalization = get_model_normalization(model_cfg)
    transform_cfg = data_cfg.get("transform", {})

    train_transform = build_tensor_transform(
        mean=normalization["mean"] if normalization else None,
        std=normalization["std"] if normalization else None,
        random_horizontal_flip_p=float(transform_cfg.get("random_horizontal_flip_p", 0.0)),  # try to get the flip probability from config, default to 0.0
    )
    eval_transform = build_tensor_transform(
        mean=normalization["mean"] if normalization else None,
        std=normalization["std"] if normalization else None,
        random_horizontal_flip_p=0.0,  # no augmentation during evaluation
    )

    train_dataset = PackedPatientDataset(pack, indices=train_idx, transform=train_transform)
    val_dataset = PackedPatientDataset(pack, indices=val_idx, transform=eval_transform)
    test_dataset = PackedPatientDataset(pack, indices=test_idx, transform=eval_transform)

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))

    loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model(model_cfg).to(device)

    if bool(train_cfg.get("use_class_weights", False)):
        train_labels = pack["labels"][train_idx]
        class_weights = compute_class_weights(
            labels=train_labels,
            num_classes=int(model_cfg["num_classes"]),
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    epochs = 1 if args.smoke else int(train_cfg["epochs"])
    pretrained = bool(model_cfg.get("pretrained", True))
    default_warmup_epochs = 1 if pretrained and not args.smoke else 0
    head_warmup_epochs = int(train_cfg.get("head_warmup_epochs", default_warmup_epochs))

    if head_warmup_epochs < 0:
        raise ValueError("head_warmup_epochs must be >= 0")

    if not pretrained and head_warmup_epochs > 0:
        print("Head warmup requested but model is not pretrained. Skipping head warmup.")
        head_warmup_epochs = 0

    use_amp = bool(train_cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    metrics_csv = run_dir / "metrics.csv"

    best_val_bal_acc = float("-inf")

    total_epochs = head_warmup_epochs + epochs
    global_epoch = 0

    if head_warmup_epochs > 0:
        print(f"\nStarting head warmup for {head_warmup_epochs} epoch(s) because pretrained=True")

        head_module = get_model_head_module(model, model_cfg)
        set_requires_grad(model, enabled=False)
        set_requires_grad(head_module, enabled=True)

        warmup_optimizer = torch.optim.AdamW(
            head_module.parameters(),
            lr=float(train_cfg.get("head_warmup_lr", train_cfg["lr"])),
            weight_decay=float(train_cfg.get("head_warmup_weight_decay", train_cfg.get("weight_decay", 0.0))),
        )

        for warmup_epoch in range(1, head_warmup_epochs + 1):
            global_epoch += 1
            print(f"\nWarmup Epoch {warmup_epoch}/{head_warmup_epochs} (global {global_epoch}/{total_epochs})")

            train_metrics = run_train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=warmup_optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )

            val_metrics = run_eval_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            best_val_bal_acc = log_epoch_and_save_checkpoint(
                run_dir=run_dir,
                metrics_csv=metrics_csv,
                epoch=global_epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                optimizer=warmup_optimizer,
                best_val_bal_acc=best_val_bal_acc,
                config=config,
                model=model,
            )

        set_requires_grad(model, enabled=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    scheduler, scheduler_name = build_scheduler(
        optimizer=optimizer,
        train_cfg=train_cfg,
        epochs=epochs,
    )

    for epoch in range(1, epochs + 1):
        global_epoch += 1
        print(f"\nEpoch {epoch}/{epochs} (global {global_epoch}/{total_epochs})")

        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics = run_eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_metrics["balanced_accuracy"])
            else:
                scheduler.step()

        best_val_bal_acc = log_epoch_and_save_checkpoint(
            run_dir=run_dir,
            metrics_csv=metrics_csv,
            epoch=global_epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            optimizer=optimizer,
            best_val_bal_acc=best_val_bal_acc,
            config=config,
            model=model,
        )

    best_checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Test metrics are obtained from the model with the best validation balanced accuracy
    test_metrics = run_eval_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    save_json(
        run_dir / "test_metrics.json",
        {
            "best_val_balanced_accuracy": best_val_bal_acc,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            "test_tp": test_metrics["tp"],
        },
    )

    print("\nFinal test metrics")
    print(
        f"loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"bal_acc={test_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()