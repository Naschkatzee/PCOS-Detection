'''
visualises results of training by plotting the training history for loss, accuracy and macro_f1 score

Plot the training history of an existing experiment.

Example:
    python -m src.visualisation.plot_training_history \
        --experiment-dir experiments/2026-08-02_16-15-23_experiment_class_test
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.experiments import Experiment


METRIC_LABELS = {
    "loss": "Loss",
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
}


def load_history(history_path: Path) -> list[dict[str, Any]]:
    """Load and validate an experiment training history."""
    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list) or not history:
        raise ValueError(
            "Training history must be a non-empty JSON list."
        )

    return history


def plot_metric(
    history: list[dict[str, Any]],
    metric: str,
    output_path: Path,
) -> None:
    """Plot one training and validation metric."""
    epochs = [entry["epoch"] for entry in history]
    training_values = [
        entry["train"][metric]
        for entry in history
    ]
    validation_values = [
        entry["validation"][metric]
        for entry in history
    ]

    label = METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.plot(
        epochs,
        training_values,
        marker="o",
        label="Training",
    )
    axis.plot(
        epochs,
        validation_values,
        marker="s",
        label="Validation",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel(label)
    axis.set_title(f"Training and validation {label.lower()}")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend()

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


def plot_training_history(
    experiment: Experiment,
) -> None:
    """Create all learning-curve figures for one experiment."""
    history = load_history(experiment.history_path)

    for metric in (
        "loss",
        "accuracy",
        "macro_f1",
    ):
        output_path = (
            experiment.figures_dir
            / f"{metric}.png"
        )

        plot_metric(
            history=history,
            metric=metric,
            output_path=output_path,
        )

        print(f"Saved: {output_path}")

    experiment.log(
        "Generated loss, accuracy, and macro-F1 curves."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training curves for an existing experiment."
        )
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing history.json and "
            "the experiment figures folder."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    experiment = Experiment.load(
        arguments.experiment_dir
    )

    plot_training_history(experiment)


if __name__ == "__main__":
    main()