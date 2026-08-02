"""
Create galleries of correct and incorrect predictions.

Example:
    python -m src.visualisation.plot_prediction_gallery --experiment-dir "experiments/2026-08-02_16-15-23_resnet50_frozen_baseline"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.experiments import Experiment


def load_config(path: Path) -> dict:
    """Load an experiment configuration."""
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")

    return config


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate per-image predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}"
        )

    predictions = pd.read_csv(path)

    required_columns = {
        "path",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"predictions.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions["correct"] = (
        predictions["correct"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(predictions["correct"])
        .astype(bool)
    )

    return predictions


def select_examples(
    predictions: pd.DataFrame,
    correct: bool,
    number_of_examples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select examples while trying to include several true classes.

    Incorrect predictions are sorted by confidence so that the most
    confidently wrong cases are prioritized.
    """
    subset = predictions[
        predictions["correct"] == correct
    ].copy()

    if subset.empty:
        return subset

    if correct:
        subset = subset.sample(
            frac=1.0,
            random_state=seed,
        )
    else:
        subset = subset.sort_values(
            "confidence",
            ascending=False,
        )

    selected_parts: list[pd.DataFrame] = []

    for _, class_group in subset.groupby(
        "true_label",
        sort=True,
    ):
        selected_parts.append(class_group.head(1))

    selected = pd.concat(
        selected_parts,
        ignore_index=False,
    )

    remaining_count = number_of_examples - len(selected)

    if remaining_count > 0:
        remaining = subset.drop(
            index=selected.index,
            errors="ignore",
        ).head(remaining_count)

        selected = pd.concat(
            [selected, remaining],
            ignore_index=False,
        )

    return selected.head(number_of_examples)


def plot_gallery(
    examples: pd.DataFrame,
    image_root: Path,
    output_path: Path,
    title: str,
    columns: int = 3,
) -> None:
    """Plot one gallery of prediction examples."""
    if examples.empty:
        print(f"No examples available for: {title}")
        return

    rows = int(np.ceil(len(examples) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
    )

    axes_array = np.asarray(axes).reshape(-1)

    for axis in axes_array:
        axis.axis("off")

    for axis, (_, row) in zip(
        axes_array,
        examples.iterrows(),
    ):
        image_path = image_root / Path(row["path"])

        if not image_path.exists():
            axis.text(
                0.5,
                0.5,
                f"Missing image:\n{row['path']}",
                ha="center",
                va="center",
            )
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            axis.imshow(image)

        true_label = str(
            row["true_label"]
        ).replace("_", " ")

        predicted_label = str(
            row["predicted_label"]
        ).replace("_", " ")

        confidence = float(row["confidence"])

        axis.set_title(
            (
                f"True: {true_label}\n"
                f"Predicted: {predicted_label}\n"
                f"Confidence: {confidence:.1%}"
            ),
            fontsize=10,
        )

        axis.set_xlabel(
            Path(row["path"]).name,
            fontsize=8,
        )

    figure.suptitle(
        title,
        fontsize=15,
    )

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

    print(f"Saved: {output_path}")


def create_prediction_galleries(
    experiment: Experiment,
    examples_per_gallery: int,
    seed: int,
) -> None:
    """Create correct- and incorrect-prediction galleries."""
    config = load_config(
        experiment.config_path
    )

    predictions = load_predictions(
        experiment.predictions_path
    )

    image_root = Path(
        config["dataset_root"]
    )

    correct_examples = select_examples(
        predictions=predictions,
        correct=True,
        number_of_examples=examples_per_gallery,
        seed=seed,
    )

    incorrect_examples = select_examples(
        predictions=predictions,
        correct=False,
        number_of_examples=examples_per_gallery,
        seed=seed,
    )

    plot_gallery(
        examples=correct_examples,
        image_root=image_root,
        output_path=(
            experiment.figures_dir
            / "correct_predictions.png"
        ),
        title="Correct test predictions",
    )

    plot_gallery(
        examples=incorrect_examples,
        image_root=image_root,
        output_path=(
            experiment.figures_dir
            / "incorrect_predictions.png"
        ),
        title="Incorrect test predictions",
    )

    correct_examples.to_csv(
        experiment.root
        / "correct_prediction_examples.csv",
        index=False,
    )

    incorrect_examples.to_csv(
        experiment.root
        / "incorrect_prediction_examples.csv",
        index=False,
    )

    experiment.log(
        "Generated correct and incorrect prediction galleries."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create prediction galleries for an evaluated experiment."
        )
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=9,
        help="Maximum number of images in each gallery.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    experiment = Experiment.load(
        arguments.experiment_dir
    )

    create_prediction_galleries(
        experiment=experiment,
        examples_per_gallery=arguments.examples,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()