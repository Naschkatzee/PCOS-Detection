"""
Create confidence-analysis figures for an evaluated experiment.

Example:
    python -m src.visualisation.plot_confidence_analysis ^
        --experiment-dir "experiments/2026-08-02_17-45-58_resnet50_frozen_baseline"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.experiments import Experiment


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate predictions produced during evaluation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}"
        )

    predictions = pd.read_csv(path)

    required_columns = {
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "predictions.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    predictions["confidence"] = pd.to_numeric(
        predictions["confidence"],
        errors="raise",
    )

    predictions["correct"] = (
        predictions["correct"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if predictions["correct"].isna().any():
        raise ValueError(
            "The 'correct' column contains unsupported values."
        )

    return predictions


def calculate_summary(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate confidence statistics for correct and incorrect predictions."""
    correct_predictions = predictions[
        predictions["correct"]
    ]

    incorrect_predictions = predictions[
        ~predictions["correct"]
    ]

    return {
        "number_of_predictions": int(len(predictions)),
        "number_correct": int(len(correct_predictions)),
        "number_incorrect": int(len(incorrect_predictions)),
        "mean_confidence_all": float(
            predictions["confidence"].mean()
        ),
        "mean_confidence_correct": float(
            correct_predictions["confidence"].mean()
        )
        if not correct_predictions.empty
        else float("nan"),
        "mean_confidence_incorrect": float(
            incorrect_predictions["confidence"].mean()
        )
        if not incorrect_predictions.empty
        else float("nan"),
        "high_confidence_errors_80_percent": int(
            (
                (~predictions["correct"])
                & (predictions["confidence"] >= 0.80)
            ).sum()
        ),
        "high_confidence_errors_90_percent": int(
            (
                (~predictions["correct"])
                & (predictions["confidence"] >= 0.90)
            ).sum()
        ),
    }


def plot_confidence_histogram(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare confidence distributions for correct and incorrect predictions."""
    correct_confidence = predictions.loc[
        predictions["correct"],
        "confidence",
    ]

    incorrect_confidence = predictions.loc[
        ~predictions["correct"],
        "confidence",
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    axis.hist(
        correct_confidence,
        bins=bins,
        alpha=0.65,
        label="Correct predictions",
    )

    if not incorrect_confidence.empty:
        axis.hist(
            incorrect_confidence,
            bins=bins,
            alpha=0.65,
            label="Incorrect predictions",
        )

    axis.set_xlabel("Predicted-class confidence")
    axis.set_ylabel("Number of test images")
    axis.set_title(
        "Confidence distribution for correct and incorrect predictions"
    )
    axis.set_xlim(0.0, 1.0)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

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


def plot_confidence_by_true_class(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean confidence by true class and prediction correctness."""
    grouped = (
        predictions.groupby(
            ["true_label", "correct"],
            as_index=False,
        )["confidence"]
        .mean()
    )

    table = grouped.pivot(
        index="true_label",
        columns="correct",
        values="confidence",
    )

    table = table.rename(
        columns={
            False: "Incorrect",
            True: "Correct",
        }
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    table.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_xlabel("True class")
    axis.set_ylabel("Mean confidence")
    axis.set_title(
        "Mean prediction confidence by true class"
    )
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.legend(title="Prediction")
    axis.grid(axis="y", alpha=0.3)

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


def create_confidence_analysis(
    experiment: Experiment,
) -> None:
    """Generate confidence figures and save a numerical summary."""
    predictions = load_predictions(
        experiment.predictions_path
    )

    summary = calculate_summary(predictions)

    histogram_path = (
        experiment.figures_dir
        / "confidence_distribution.png"
    )

    class_path = (
        experiment.figures_dir
        / "confidence_by_true_class.png"
    )

    plot_confidence_histogram(
        predictions=predictions,
        output_path=histogram_path,
    )

    plot_confidence_by_true_class(
        predictions=predictions,
        output_path=class_path,
    )

    experiment.save_json(
        experiment.root
        / "confidence_summary.json",
        summary,
    )

    experiment.log(
        "Generated confidence analysis figures."
    )

    print("Confidence analysis")
    print(
        f"Mean confidence, all predictions: "
        f"{summary['mean_confidence_all']:.4f}"
    )
    print(
        f"Mean confidence, correct predictions: "
        f"{summary['mean_confidence_correct']:.4f}"
    )
    print(
        f"Mean confidence, incorrect predictions: "
        f"{summary['mean_confidence_incorrect']:.4f}"
    )
    print(
        "Incorrect predictions with confidence >= 0.80: "
        f"{summary['high_confidence_errors_80_percent']}"
    )
    print(
        "Incorrect predictions with confidence >= 0.90: "
        f"{summary['high_confidence_errors_90_percent']}"
    )

    print(f"\nSaved: {histogram_path}")
    print(f"Saved: {class_path}")
    print(
        "Saved: "
        f"{experiment.root / 'confidence_summary.json'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create confidence-analysis figures for "
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

    create_confidence_analysis(experiment)


if __name__ == "__main__":
    main()