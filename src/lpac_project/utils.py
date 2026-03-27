from __future__ import annotations

import csv
import json
import random
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def make_run_dir(output_root: str | Path, run_name: str) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_metrics_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)