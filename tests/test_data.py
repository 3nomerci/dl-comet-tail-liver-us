import json

import torch

from lpac_project.data import PackedPatientDataset, patient_split_indices, save_split_artifact


def make_pack():
    return {
        "images": torch.rand(6, 3, 64, 64),
        "labels": torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.long),
        "patients": torch.tensor([10, 10, 20, 20, 30, 40], dtype=torch.long),
        "paths": [f"img_{i}.npy" for i in range(6)],
        "image_size": 64,
    }


def test_dataset_subset_returns_expected_triplet():
    pack = make_pack()
    dataset = PackedPatientDataset(pack, indices=torch.tensor([0, 2, 4]))

    image, label, patient = dataset[1]

    assert image.shape == (3, 64, 64)
    assert label.item() == 1
    assert patient.item() == 20


def test_patient_split_has_no_leakage():
    pack = make_pack()
    train_idx, val_idx, test_idx = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )

    patients = pack["patients"]
    train_patients = set(patients[train_idx].tolist())
    val_patients = set(patients[val_idx].tolist())
    test_patients = set(patients[test_idx].tolist())

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def make_unbalanced_pack():
    # Patient IDs with variable number of samples and binary labels per patient.
    patient_specs = [
        (100, 8, 0),
        (101, 7, 0),
        (102, 6, 0),
        (103, 5, 0),
        (200, 9, 1),
        (201, 8, 1),
        (202, 7, 1),
        (203, 6, 1),
        (204, 4, 1),
        (205, 3, 0),
    ]

    labels = []
    patients = []
    for patient_id, count, label in patient_specs:
        patients.extend([patient_id] * count)
        labels.extend([label] * count)

    n = len(labels)
    return {
        "images": torch.rand(n, 3, 64, 64),
        "labels": torch.tensor(labels, dtype=torch.long),
        "patients": torch.tensor(patients, dtype=torch.long),
        "paths": [f"img_{i}.npy" for i in range(n)],
        "image_size": 64,
    }


def test_patient_split_is_deterministic_for_fixed_seed():
    pack = make_unbalanced_pack()

    split_a = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=123,
    )
    split_b = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=123,
    )

    assert torch.equal(split_a[0], split_b[0])
    assert torch.equal(split_a[1], split_b[1])
    assert torch.equal(split_a[2], split_b[2])


def test_patient_split_approximately_matches_sample_fractions():
    pack = make_unbalanced_pack()
    train_idx, val_idx, test_idx = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )

    total_samples = float(pack["labels"].numel())
    got = [
        float(train_idx.numel()) / total_samples,
        float(val_idx.numel()) / total_samples,
        float(test_idx.numel()) / total_samples,
    ]
    target = [0.7, 0.15, 0.15]

    # Small datasets with grouped patients cannot exactly hit all targets.
    assert abs(got[0] - target[0]) <= 0.2
    assert abs(got[1] - target[1]) <= 0.2
    assert abs(got[2] - target[2]) <= 0.2


def test_patient_split_preserves_approximate_label_balance():
    pack = make_unbalanced_pack()
    train_idx, val_idx, test_idx = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        stratify=True,
    )

    labels = pack["labels"]
    patients = pack["patients"]
    global_pos_rate = float((labels == 1).float().mean().item())

    for indices in (train_idx, val_idx, test_idx):
        split_labels = labels[indices]
        split_pos_rate = float((split_labels == 1).float().mean().item())
        n_split_patients = int(torch.unique(patients[indices]).numel())

        if n_split_patients >= 2:
            assert abs(split_pos_rate - global_pos_rate) <= 0.35
        else:
            # A single-patient split can be extreme despite stratification intent.
            assert abs(split_pos_rate - global_pos_rate) <= 0.5


def test_save_split_artifacts_contains_summary_fields(tmp_path):
    pack = make_unbalanced_pack()
    train_idx, val_idx, test_idx = patient_split_indices(
        patients=pack["patients"],
        labels=pack["labels"],
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )

    split_path = tmp_path / "split.json"
    summary_path = tmp_path / "split_summary.json"

    save_split_artifact(
        output_path=split_path,
        summary_output_path=summary_path,
        patients=pack["patients"],
        labels=pack["labels"],
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        method="heuristic_group_holdout",
        stratify=True,
    )

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "train_indices" in split_payload
    assert "val_indices" in split_payload
    assert "test_indices" in split_payload

    assert summary_payload["split_method"] == "heuristic_group_holdout"
    assert summary_payload["seed"] == 42
    assert summary_payload["stratify"] is True
    assert "splits" in summary_payload

    for split_name in ("train", "val", "test"):
        assert "num_samples" in summary_payload["splits"][split_name]
        assert "num_patients" in summary_payload["splits"][split_name]
        assert "positive_samples" in summary_payload["splits"][split_name]
        assert "negative_samples" in summary_payload["splits"][split_name]
        assert "actual_fraction" in summary_payload["splits"][split_name]