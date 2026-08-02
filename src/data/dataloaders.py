'''
This file contains the function to create DataLoaders for training, validation, and testing.
It uses the OvarianUltrasoundDataset and the transforms defined in transforms.py to create the DataLoaders

python -c "from src.data.dataloaders import create_dataloaders; train_loader, val_loader, test_loader = create_dataloaders(); batch = next(iter(train_loader)); print(batch['image'].shape); print(batch['label'].shape); print(batch['label_name']); print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))"
'''

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import OvarianUltrasoundDataset
from src.data.transforms import (
    create_evaluation_transforms,
    create_train_transforms,
)

import torch

def create_dataloaders(
    dataset_root: Path | str = "dataset/raw/ovarian_ultrasound",
    splits_root: Path | str = "dataset/splits",
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test DataLoaders.

    num_workers=0 is a safe default on Windows.
    """

    dataset_root = Path(dataset_root)
    splits_root = Path(splits_root)

    train_dataset = OvarianUltrasoundDataset(
        csv_path=splits_root / "train.csv",
        image_root=dataset_root,
        transform=create_train_transforms(image_size),
    )

    validation_dataset = OvarianUltrasoundDataset(
        csv_path=splits_root / "validation.csv",
        image_root=dataset_root,
        transform=create_evaluation_transforms(image_size),
    )

    test_dataset = OvarianUltrasoundDataset(
        csv_path=splits_root / "test.csv",
        image_root=dataset_root,
        transform=create_evaluation_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader, test_loader