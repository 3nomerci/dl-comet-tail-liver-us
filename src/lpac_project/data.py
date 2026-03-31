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
    labels: torch.Tensor,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    stratify: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Split samples by patient while targeting sample fractions and optional label stratification.'''

    if patients.ndim != 1 or labels.ndim != 1:
        raise ValueError("patients and labels must be 1D tensors")
    if patients.numel() != labels.numel():
        raise ValueError("patients and labels must have the same number of elements")

    validate_patient_label_consistency(labels=labels, patients=patients)

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    unique_patients = torch.unique(patients).cpu().numpy() # 1D tensor of patient IDs
    rng = np.random.default_rng(seed)

    n_patients = len(unique_patients)
    if n_patients < 3:
        raise ValueError("Need at least 3 unique patients for train/val/test split.")

    # Compute per-patient sample counts and patient label.
    patient_values = patients.cpu().numpy()
    label_values = labels.cpu().numpy()
    patient_stats: list[tuple[int, int, int]] = [] # List of (patient_id, sample_count, patient_label)
    for patient in unique_patients.tolist():
        mask = patient_values == patient
        patient_count = int(mask.sum())
        patient_label = int(label_values[mask][0])
        patient_stats.append((int(patient), patient_count, patient_label))

    # Randomize tie-breakers, then assign patients to splits in order of decreasing sample count
    rng.shuffle(patient_stats)
    patient_stats.sort(key=lambda x: x[1], reverse=True)

    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=np.float64)
    target_samples = fractions * float(len(patient_values)) # target number of samples per split
    target_pos_samples = fractions * float((labels == 1).sum().item()) # target number of positive samples per split (stratification)
    target_patients = fractions * float(n_patients) # target number of patients per split (secondary balancing criterion)

    split_patients: list[list[int]] = [[], [], []] # patient IDs assigned to each split
    split_sample_counts = np.zeros(3, dtype=np.float64)
    split_pos_sample_counts = np.zeros(3, dtype=np.float64)
    split_patient_counts = np.zeros(3, dtype=np.float64)

    for idx, (patient, patient_count, patient_label) in enumerate(patient_stats): # iterate over patients in order of decreasing sample count
        remaining = n_patients - idx
        empty_splits = [i for i in range(3) if len(split_patients[i]) == 0]
        if len(empty_splits) == remaining:
            candidate_splits = empty_splits
        else:
            candidate_splits = [0, 1, 2]

        best_split = None
        best_score = None

        for split_id in candidate_splits: # for each split, compute hypothetical new sample counts and score based on distance from target, then assign patient to best split
            new_sample_counts = split_sample_counts.copy()
            new_sample_counts[split_id] += float(patient_count)
            new_patient_counts = split_patient_counts.copy()
            new_patient_counts[split_id] += 1.0

            sample_score = np.abs(new_sample_counts - target_samples) / np.maximum(target_samples, 1.0)
            score = float(sample_score.sum())

            patient_score = np.abs(new_patient_counts - target_patients) / np.maximum(target_patients, 1.0)
            score += 0.6 * float(patient_score.sum())

            if stratify:
                new_pos_counts = split_pos_sample_counts.copy()
                new_pos_counts[split_id] += float(patient_count if patient_label == 1 else 0)
                pos_score = np.abs(new_pos_counts - target_pos_samples) / np.maximum(target_pos_samples, 1.0)
                score += 1.5 * float(pos_score.sum())

            # Tiny jitter avoids deterministic bias toward lower-index splits on exact ties.
            score += float(rng.uniform(0.0, 1e-12))

            # best split is determined with an heuristic score based on distance from target sample counts, 
            # with optional stratification and secondary balancing based on patient counts to avoid extreme imbalances when patients have many samples.
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_id

        assert best_split is not None
        split_patients[best_split].append(patient)
        split_sample_counts[best_split] += float(patient_count)
        split_patient_counts[best_split] += 1.0
        if patient_label == 1:
            split_pos_sample_counts[best_split] += float(patient_count)

    if any(len(x) == 0 for x in split_patients):
        raise ValueError("Unable to create train/val/test split with at least one patient per split")

    train_patients = torch.as_tensor(split_patients[0], dtype=patients.dtype)
    val_patients = torch.as_tensor(split_patients[1], dtype=patients.dtype)
    test_patients = torch.as_tensor(split_patients[2], dtype=patients.dtype)

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
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    test_indices: torch.Tensor,
    train_fraction: float | None = None,
    val_fraction: float | None = None,
    test_fraction: float | None = None,
    seed: int | None = None,
    method: str | None = None,
    stratify: bool | None = None,
    summary_output_path: str | Path | None = None,
) -> None:
    '''Saves the patient split information to a JSON file for reproducibility and analysis.'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

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

    if summary_output_path is None:
        summary_output_path = output_path.with_name("split_summary.json")

    summary_output_path = Path(summary_output_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = int(labels.numel())
    total_patients = int(torch.unique(patients).numel())
    total_positive = int((labels == 1).sum().item())
    total_negative = int((labels == 0).sum().item())

    target_fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }

    split_summary: dict[str, dict] = {}
    for split_name, indices in split_indices.items():
        split_patients = torch.unique(patients[indices])
        split_labels = labels[indices]
        sample_count = int(indices.numel())
        patient_count = int(split_patients.numel())
        positive_count = int((split_labels == 1).sum().item())
        negative_count = int((split_labels == 0).sum().item())

        target_fraction = target_fractions[split_name]
        actual_fraction = float(sample_count / total_samples) if total_samples > 0 else None

        split_summary[split_name] = {
            "num_samples": sample_count,
            "num_patients": patient_count,
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "patient_ids": split_patients.tolist(),
            "target_fraction": target_fraction,
            "actual_fraction": actual_fraction,
            "fraction_error": (
                float(abs(actual_fraction - target_fraction))
                if actual_fraction is not None and target_fraction is not None
                else None
            ),
        }

    summary_payload = {
        "split_method": method,
        "seed": seed,
        "stratify": stratify,
        "totals": {
            "num_samples": total_samples,
            "num_patients": total_patients,
            "positive_samples": total_positive,
            "negative_samples": total_negative,
        },
        "splits": split_summary,
    }

    with summary_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)