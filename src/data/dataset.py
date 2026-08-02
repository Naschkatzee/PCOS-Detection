"""
Contains only dataset loading.

PyTorch cannot train directly from a CSV. It expects a Dataset object.
So we need to implement a Dataset that reads the CSV and loads images from disk.
And this file is a transformer that converts a CSV into a Dataset.

train.csv
        │
        ▼
load image
        │
convert to RGB
        │
apply transforms
        │
return (image, label)

python -c "from src.data.dataset import OvarianUltrasoundDataset; from src.data.transforms import create_train_transforms; dataset = OvarianUltrasoundDataset('dataset/splits/train.csv', 'dataset/raw/ovarian_ultrasound', create_train_transforms()); sample = dataset[0]; print(len(dataset)); print(sample['image'].shape); print(sample['label']); print(sample['label_name']); print(sample['path'])"
"""


from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_INDEX = {
    "Dominant_Follicle": 0,
    "Normal": 1,
    "PCO": 2,
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


class OvarianUltrasoundDataset(Dataset):
    """Dataset for ovarian ultrasound image classification."""

    def __init__(
        self,
        csv_path: Path | str,
        image_root: Path | str,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image root directory not found: {self.image_root}"
            )

        self.dataframe = pd.read_csv(self.csv_path)

        required_columns = {"path", "label"}
        missing_columns = required_columns - set(self.dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_columns)}"
            )

        unknown_labels = (
            set(self.dataframe["label"].unique()) - set(LABEL_TO_INDEX)
        )

        if unknown_labels:
            raise ValueError(
                f"Unknown labels found in CSV: {sorted(unknown_labels)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        relative_path = Path(row["path"])
        image_path = self.image_root / relative_path
        label_name = str(row["label"])
        label_index = LABEL_TO_INDEX[label_name]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label_index, dtype=torch.long),
            "label_name": label_name,
            "path": str(relative_path),
        }