from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


REQUIRED_KEYS = ("images", "labels", "patients")


def load_pack(path: str | Path) -> dict:
    '''Loads a dataset pack from a .pt file and validates its structure.'''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    pack = torch.load(path, map_location="cpu", weights_only=False)
    validate_pack(pack)
    return pack


def validate_pack(pack: dict) -> None:
    if not isinstance(pack, dict):
        raise TypeError(f"Expected dataset pack to be a dict, got {type(pack)}")

    missing = [k for k in REQUIRED_KEYS if k not in pack]
    if missing:
        raise KeyError(f"Missing required keys in dataset pack: {missing}")

    images = pack["images"]
    labels = pack["labels"]
    patients = pack["patients"]

    if not isinstance(images, torch.Tensor):
        raise TypeError("pack['images'] must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("pack['labels'] must be a torch.Tensor")
    if not isinstance(patients, torch.Tensor):
        raise TypeError("pack['patients'] must be a torch.Tensor")

    if images.ndim != 4:
        raise ValueError(f"Expected images shape [N, C, H, W], got {tuple(images.shape)}")
    if images.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {images.shape[1]}")
    if not torch.is_floating_point(images):
        raise TypeError(f"Expected floating point images, got {images.dtype}")

    n_samples = images.shape[0]

    if labels.ndim != 1 or labels.shape[0] != n_samples:
        raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}")
    if patients.ndim != 1 or patients.shape[0] != n_samples:
        raise ValueError(f"patients must have shape [N], got {tuple(patients.shape)}")

    validate_patient_label_consistency(labels=labels, patients=patients)


def validate_patient_label_consistency(labels: torch.Tensor, patients: torch.Tensor) -> None:
    '''Ensures that each patient has a consistent label across all samples.'''
    patient_to_label: dict[int, int] = {}

    for patient, label in zip(patients.tolist(), labels.tolist(), strict=False):
        patient = int(patient)
        label = int(label)

        if patient in patient_to_label and patient_to_label[patient] != label:
            raise ValueError(
                f"Patient {patient} has inconsistent labels: "
                f"{patient_to_label[patient]} and {label}."
            )

        patient_to_label[patient] = label


class PackedPatientDataset(Dataset):
    '''This class is a subclass of torch.utils.data.Dataset that provides access to the images, labels, and patient IDs stored in a dataset pack. 
    It supports optional indexing to create subsets and can apply transformations to the images.'''
    def __init__(
        self,
        pack: dict,
        indices: torch.Tensor | None = None,
        transform=None,
    ) -> None:
        validate_pack(pack)

        self.images = pack["images"]
        self.labels = pack["labels"].long()
        self.patients = pack["patients"].long()
        self.transform = transform

        if indices is None:
            self.indices = torch.arange(self.images.shape[0], dtype=torch.long)
        else:
            self.indices = indices.long()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx].item())

        image = self.images[sample_idx].float()
        label = self.labels[sample_idx]
        patient = self.patients[sample_idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, patient


def patient_split_indices(
    patients: torch.Tensor,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Splits the dataset into train/val/test sets based on patientIDs, ensuring that all samples from the same patient are in the same split.
    TODO: Add stratification to ensure class balance
    TODO: Add euristic to handle cases where patients have very different number of samples (e.g. one patient has 100 samples, others have 1-2)
'''
    
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    unique_patients = torch.unique(patients).cpu().numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)

    n_patients = len(unique_patients)
    if n_patients < 3:
        raise ValueError("Need at least 3 unique patients for train/val/test split.")

    n_train = int(n_patients * train_fraction)
    n_val = int(n_patients * val_fraction)
    n_test = n_patients - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"Invalid split sizes for {n_patients} patients: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    train_patients = torch.as_tensor(unique_patients[:n_train], dtype=patients.dtype)
    val_patients = torch.as_tensor(unique_patients[n_train:n_train + n_val], dtype=patients.dtype)
    test_patients = torch.as_tensor(unique_patients[n_train + n_val:], dtype=patients.dtype)

    train_indices = torch.nonzero(torch.isin(patients, train_patients), as_tuple=False).squeeze(1)
    val_indices = torch.nonzero(torch.isin(patients, val_patients), as_tuple=False).squeeze(1)
    test_indices = torch.nonzero(torch.isin(patients, test_patients), as_tuple=False).squeeze(1)

    return train_indices.long(), val_indices.long(), test_indices.long()


def limit_indices(indices: torch.Tensor, max_items: int | None, seed: int) -> torch.Tensor:
    if max_items is None or indices.numel() <= max_items:
        return indices

    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices.numel())[:max_items]
    perm = torch.as_tensor(perm, dtype=torch.long)

    return indices[perm].long()


def build_tensor_transform(
    mean: list[float] | None = None,
    std: list[float] | None = None,
    random_horizontal_flip_p: float = 0.0, # probability of flipping each image
):
    ops = []

    if random_horizontal_flip_p > 0.0:
        ops.append(T.RandomHorizontalFlip(p=random_horizontal_flip_p)) 

    if mean is not None and std is not None:
        ops.append(T.Normalize(mean=mean, std=std))

    if not ops:
        return None

    return T.Compose(ops)


def save_split_artifact(
    output_path: str | Path,
    patients: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    test_indices: torch.Tensor,
) -> None:
    '''Saves the patient split information to a JSON file for reproducibility and analysis.'''
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_patients": torch.unique(patients[train_indices]).tolist(),
        "val_patients": torch.unique(patients[val_indices]).tolist(),
        "test_patients": torch.unique(patients[test_indices]).tolist(),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)