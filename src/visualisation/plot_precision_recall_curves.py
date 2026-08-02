"""
Create one-vs-rest precision-recall curves for an evaluated experiment.

Example:
    python -m src.visualisation.plot_precision_recall_curves \
        --experiment-dir "experiments/2026-08-02_17-45-58_resnet50_frozen_baseline"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from src.data.dataset import INDEX_TO_LABEL
from src.experiments import Experiment


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate per-image predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}"
        )

    predictions = pd.read_csv(path)

    required_columns = {
        "true_index",
        "true_label",
    }

    probability_columns = {
        f"probability_{class_name}"
        for class_name in INDEX_TO_LABEL.values()
    }

    missing_columns = (
        required_columns | probability_columns
    ) - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "predictions.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return predictions


def calculate_precision_recall_metrics(
    predictions: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, float],
]:
    """Calculate one-vs-rest PR curves and average precision."""
    class_indices = sorted(INDEX_TO_LABEL)

    class_names = [
        INDEX_TO_LABEL[index]
        for index in class_indices
    ]

    true_indices = predictions[
        "true_index"
    ].to_numpy()

    binary_targets = label_binarize(
        true_indices,
        classes=class_indices,
    )

    probability_matrix = np.column_stack(
        [
            predictions[
                f"probability_{class_name}"
            ].to_numpy()
            for class_name in class_names
        ]
    )

    curves: dict[str, dict[str, Any]] = {}
    average_precision_values: dict[str, float] = {}

    for class_position, class_name in enumerate(
        class_names
    ):
        class_targets = binary_targets[
            :,
            class_position,
        ]

        if np.unique(class_targets).size < 2:
            raise ValueError(
                f"Precision-recall curve cannot be calculated "
                f"for {class_name}: the test set does not "
                "contain both positive and negative examples."
            )

        precision, recall, thresholds = (
            precision_recall_curve(
                class_targets,
                probability_matrix[:, class_position],
            )
        )

        average_precision = average_precision_score(
            class_targets,
            probability_matrix[:, class_position],
        )

        curves[class_name] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "positive_prevalence": float(
                class_targets.mean()
            ),
        }

        average_precision_values[class_name] = float(
            average_precision
        )

    macro_average_precision = float(
        np.mean(
            list(
                average_precision_values.values()
            )
        )
    )

    average_precision_values[
        "macro_average_precision"
    ] = macro_average_precision

    return curves, average_precision_values


def plot_precision_recall_curves(
    curves: dict[str, dict[str, Any]],
    average_precision_values: dict[str, float],
    output_path: Path,
) -> None:
    """Plot all one-vs-rest precision-recall curves."""
    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    for class_name in INDEX_TO_LABEL.values():
        curve = curves[class_name]

        display_name = class_name.replace(
            "_",
            " ",
        )

        axis.plot(
            curve["recall"],
            curve["precision"],
            linewidth=2,
            label=(
                f"{display_name} "
                f"(AP = "
                f"{average_precision_values[class_name]:.3f})"
            ),
        )

        axis.axhline(
            y=curve["positive_prevalence"],
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        "One-vs-rest precision-recall curves"
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left")

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


def create_precision_recall_analysis(
    experiment: Experiment,
) -> None:
    """Generate PR curves and save average-precision values."""
    predictions = load_predictions(
        experiment.predictions_path
    )

    curves, average_precision_values = (
        calculate_precision_recall_metrics(
            predictions
        )
    )

    output_path = (
        experiment.figures_dir
        / "precision_recall_curves.png"
    )

    plot_precision_recall_curves(
        curves=curves,
        average_precision_values=average_precision_values,
        output_path=output_path,
    )

    metrics_path = (
        experiment.root
        / "precision_recall_metrics.json"
    )

    experiment.save_json(
        metrics_path,
        {
            "average_precision": (
                average_precision_values
            ),
            "curves": curves,
        },
    )

    experiment.log(
        "Generated one-vs-rest precision-recall curves."
    )

    print("Average precision results")

    for class_name in INDEX_TO_LABEL.values():
        print(
            f"{class_name}: "
            f"{average_precision_values[class_name]:.4f}"
        )

    print(
        "Macro average precision: "
        f"{average_precision_values['macro_average_precision']:.4f}"
    )

    print(f"\nSaved: {output_path}")
    print(f"Saved: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one-vs-rest precision-recall curves "
            "for an evaluated experiment."
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

    create_precision_recall_analysis(
        experiment
    )


if __name__ == "__main__":
    main()