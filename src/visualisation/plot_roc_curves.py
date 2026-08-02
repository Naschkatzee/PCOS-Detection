"""
Create one-vs-rest ROC curves for an evaluated experiment.

Example:
    python -m src.visualisation.plot_roc_curves --experiment-dir "experiments/2026-08-02_17-45-58_resnet50_frozen_baseline"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
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
        required_columns
        | probability_columns
    ) - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "predictions.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return predictions


def calculate_roc_metrics(
    predictions: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, float],
]:
    """Calculate one-vs-rest ROC curves and AUC values."""
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
    auc_values: dict[str, float] = {}

    for class_position, class_name in enumerate(
        class_names
    ):
        class_targets = binary_targets[
            :,
            class_position,
        ]

        if np.unique(class_targets).size < 2:
            raise ValueError(
                f"ROC curve cannot be calculated for "
                f"{class_name}: the test set does not contain "
                "both positive and negative examples."
            )

        false_positive_rate, true_positive_rate, thresholds = (
            roc_curve(
                class_targets,
                probability_matrix[:, class_position],
            )
        )

        class_auc = auc(
            false_positive_rate,
            true_positive_rate,
        )

        curves[class_name] = {
            "false_positive_rate": (
                false_positive_rate.tolist()
            ),
            "true_positive_rate": (
                true_positive_rate.tolist()
            ),
            "thresholds": thresholds.tolist(),
        }

        auc_values[class_name] = float(class_auc)

    macro_auc = float(
        np.mean(list(auc_values.values()))
    )

    auc_values["macro_auc"] = macro_auc

    return curves, auc_values


def plot_roc_curves(
    curves: dict[str, dict[str, Any]],
    auc_values: dict[str, float],
    output_path: Path,
) -> None:
    """Plot all one-vs-rest ROC curves in one figure."""
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
            curve["false_positive_rate"],
            curve["true_positive_rate"],
            linewidth=2,
            label=(
                f"{display_name} "
                f"(AUC = {auc_values[class_name]:.3f})"
            ),
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        label="No-skill classifier",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(
        "One-vs-rest ROC curves"
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")

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


def create_roc_analysis(
    experiment: Experiment,
) -> None:
    """Generate ROC curves and save AUC values."""
    predictions = load_predictions(
        experiment.predictions_path
    )

    curves, auc_values = calculate_roc_metrics(
        predictions
    )

    output_path = (
        experiment.figures_dir
        / "roc_curves.png"
    )

    plot_roc_curves(
        curves=curves,
        auc_values=auc_values,
        output_path=output_path,
    )

    experiment.save_json(
        experiment.root / "roc_metrics.json",
        {
            "auc": auc_values,
            "curves": curves,
        },
    )

    experiment.log(
        "Generated one-vs-rest ROC curves."
    )

    print("ROC AUC results")

    for class_name in INDEX_TO_LABEL.values():
        print(
            f"{class_name}: "
            f"{auc_values[class_name]:.4f}"
        )

    print(
        f"Macro AUC: "
        f"{auc_values['macro_auc']:.4f}"
    )

    print(f"\nSaved: {output_path}")
    print(
        "Saved: "
        f"{experiment.root / 'roc_metrics.json'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one-vs-rest ROC curves for "
            "an evaluated experiment."
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

    create_roc_analysis(experiment)


if __name__ == "__main__":
    main()