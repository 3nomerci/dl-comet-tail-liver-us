import torch

from lpac_project.data import PackedPatientDataset, patient_split_indices


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