from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import functional as F
from tqdm import tqdm

from src.data.dataloaders import create_dataloaders
from src.data.dataset import INDEX_TO_LABEL, LABEL_TO_INDEX
from src.models.resnet import create_resnet50_classifier


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained ResNet-50 checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_arguments = checkpoint.get("arguments", {})
    dropout = float(checkpoint_arguments.get("dropout", 0.2))

    model = create_resnet50_classifier(
        number_of_classes=len(LABEL_TO_INDEX),
        freeze_backbone=True,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference and return one row per test image."""

    rows: list[dict[str, Any]] = []

    for batch in tqdm(data_loader, desc="Test inference"):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = batch["label"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        targets_cpu = targets.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        probabilities_cpu = probabilities.cpu().tolist()

        for index in range(len(targets_cpu)):
            true_index = int(targets_cpu[index])
            predicted_index = int(predictions_cpu[index])

            row = {
                "path": batch["path"][index],
                "true_index": true_index,
                "true_label": INDEX_TO_LABEL[true_index],
                "predicted_index": predicted_index,
                "predicted_label": INDEX_TO_LABEL[predicted_index],
                "correct": true_index == predicted_index,
                "confidence": float(
                    probabilities_cpu[index][predicted_index]
                ),
            }

            for class_index, class_name in INDEX_TO_LABEL.items():
                row[f"probability_{class_name}"] = float(
                    probabilities_cpu[index][class_index]
                )

            rows.append(row)

    if not rows:
        raise RuntimeError("The test DataLoader produced no samples.")

    return pd.DataFrame(rows)


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Calculate aggregate and per-class classification metrics."""

    true_labels = predictions["true_index"].to_numpy()
    predicted_labels = predictions["predicted_index"].to_numpy()

    class_indices = sorted(INDEX_TO_LABEL)
    class_names = [
        INDEX_TO_LABEL[index]
        for index in class_indices
    ]

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_indices,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "number_of_test_images": int(len(predictions)),
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "macro_precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
    }

    return metrics, matrix.tolist()


def plot_confusion_matrix(
    matrix: list[list[int]],
    output_path: Path,
) -> None:
    """Save a labelled confusion-matrix figure."""

    class_names = [
        INDEX_TO_LABEL[index].replace("_", " ")
        for index in sorted(INDEX_TO_LABEL)
    ]

    figure, axis = plt.subplots(figsize=(7, 6))

    image = axis.imshow(matrix)

    axis.set_title("ResNet-50 test confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=30, ha="right")
    axis.set_yticklabels(class_names)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def evaluate(arguments: argparse.Namespace) -> None:
    """Evaluate the best frozen ResNet-50 checkpoint."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model, checkpoint = load_model(
        checkpoint_path=arguments.checkpoint,
        device=device,
    )

    _, _, test_loader = create_dataloaders(
        dataset_root=arguments.dataset_root,
        splits_root=arguments.splits_root,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )

    predictions = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    metrics, matrix = calculate_metrics(predictions)

    arguments.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        arguments.output / "predictions.csv",
        index=False,
    )

    result = {
        "checkpoint_epoch": int(
            checkpoint.get("epoch", -1)
        ),
        "checkpoint_validation_metrics": checkpoint.get(
            "validation_metrics",
            {},
        ),
        "test_metrics": metrics,
        "confusion_matrix": matrix,
    }

    with (
        arguments.output / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    plot_confusion_matrix(
        matrix=matrix,
        output_path=(
            arguments.output / "confusion_matrix.png"
        ),
    )

    print("\nTest results")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro precision:   {metrics['macro_precision']:.4f}")
    print(f"Macro recall:      {metrics['macro_recall']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")

    print(
        f"\nSaved evaluation results to: "
        f"{arguments.output.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen ResNet-50 baseline."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/resnet50_frozen_best.pt"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "dataset/raw/ovarian_ultrasound"
        ),
    )

    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("dataset/splits"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/evaluation/resnet50_frozen"
        ),
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()