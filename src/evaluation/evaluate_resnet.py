"""
Evaluate a completed ResNet-50 experiment.

Example:
    python -m src.evaluation.evaluate_resnet \
        --experiment-dir experiments/2026-08-02_16-15-23_experiment_class_test
"""

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
from src.experiments import Experiment
from src.models.resnet import create_resnet50_classifier


def load_config(path: Path) -> dict[str, Any]:
    """Load the configuration of an existing experiment."""
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")

    return config


def load_model(
    experiment: Experiment,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the best checkpoint of a ResNet-50 experiment."""
    if not experiment.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {experiment.checkpoint_path}"
        )

    checkpoint = torch.load(
        experiment.checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_resnet50_classifier(
        number_of_classes=len(LABEL_TO_INDEX),
        freeze_backbone=bool(
            config.get("freeze_backbone", True)
        ),
        dropout=float(config.get("dropout", 0.2)),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Run test inference and return one row per image."""
    rows: list[dict[str, Any]] = []

    for batch in tqdm(
        data_loader,
        desc="Test inference",
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        targets = batch["label"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)

        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        targets_cpu = targets.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        probabilities_cpu = probabilities.cpu().tolist()

        for index in range(len(targets_cpu)):
            true_index = int(targets_cpu[index])
            predicted_index = int(
                predictions_cpu[index]
            )

            row = {
                "path": batch["path"][index],
                "true_index": true_index,
                "true_label": INDEX_TO_LABEL[true_index],
                "predicted_index": predicted_index,
                "predicted_label": (
                    INDEX_TO_LABEL[predicted_index]
                ),
                "correct": true_index == predicted_index,
                "confidence": float(
                    probabilities_cpu[index][predicted_index]
                ),
            }

            for class_index, class_name in (
                INDEX_TO_LABEL.items()
            ):
                row[
                    f"probability_{class_name}"
                ] = float(
                    probabilities_cpu[index][class_index]
                )

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "The test DataLoader produced no samples."
        )

    return pd.DataFrame(rows)


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Calculate aggregate and per-class test metrics."""
    true_labels = predictions[
        "true_index"
    ].to_numpy()

    predicted_labels = predictions[
        "predicted_index"
    ].to_numpy()

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
        "number_of_test_images": int(
            len(predictions)
        ),
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
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
    """Save a labelled confusion matrix."""
    class_names = [
        INDEX_TO_LABEL[index].replace("_", " ")
        for index in sorted(INDEX_TO_LABEL)
    ]

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(matrix)

    axis.set_title(
        "ResNet-50 test confusion matrix"
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    axis.set_xticks(
        range(len(class_names))
    )
    axis.set_yticks(
        range(len(class_names))
    )

    axis.set_xticklabels(
        class_names,
        rotation=30,
        ha="right",
    )
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


def evaluate_experiment(
    experiment: Experiment,
) -> None:
    """Evaluate one completed ResNet-50 experiment."""
    config = load_config(
        experiment.config_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model, checkpoint = load_model(
        experiment=experiment,
        config=config,
        device=device,
    )

    dataset_root = Path(
        config["dataset_root"]
    )
    splits_root = Path(
        config["splits_root"]
    )

    _, _, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        splits_root=splits_root,
        image_size=int(
            config.get("image_size", 224)
        ),
        batch_size=int(
            config.get("batch_size", 8)
        ),
        num_workers=int(
            config.get("num_workers", 0)
        ),
    )

    predictions = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    metrics, matrix = calculate_metrics(
        predictions
    )

    experiment.save_predictions(
        predictions
    )

    result = {
        "checkpoint_epoch": int(
            checkpoint.get("epoch", -1)
        ),
        "checkpoint_validation_metrics": (
            checkpoint.get(
                "validation_metrics",
                {},
            )
        ),
        "test_metrics": metrics,
        "confusion_matrix": matrix,
    }

    experiment.save_metrics(result)

    plot_confusion_matrix(
        matrix=matrix,
        output_path=(
            experiment.confusion_matrix_path
        ),
    )

    experiment.log(
        (
            "Test evaluation completed. "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"macro F1={metrics['macro_f1']:.4f}"
        )
    )

    print("\nTest results")
    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro precision:   "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro recall:      "
        f"{metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1:       "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        "\nPredictions saved to: "
        f"{experiment.predictions_path}"
    )
    print(
        "Metrics saved to: "
        f"{experiment.metrics_path}"
    )
    print(
        "Confusion matrix saved to: "
        f"{experiment.confusion_matrix_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing ResNet-50 experiment."
        )
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    experiment = Experiment.load(
        arguments.experiment_dir
    )

    evaluate_experiment(experiment)


if __name__ == "__main__":
    main()