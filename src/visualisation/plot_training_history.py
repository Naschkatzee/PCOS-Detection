'''
visualises results of training by plotting the training history for loss, accuracy and macro_f1 score
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric(
    history: list[dict],
    metric: str,
    output_dir: Path,
) -> None:
    epochs = [entry["epoch"] for entry in history]

    train_values = [
        entry["train"][metric]
        for entry in history
    ]

    validation_values = [
        entry["validation"][metric]
        for entry in history
    ]

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        train_values,
        marker="o",
        label="Training",
    )

    plt.plot(
        epochs,
        validation_values,
        marker="s",
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(metric.replace("_", " ").title())
    plt.grid(True)
    plt.legend()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_dir / f"{metric}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            "reports/training/resnet50_frozen_history.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/figures/training"
        ),
    )

    args = parser.parse_args()

    history = json.loads(
        args.history.read_text(
            encoding="utf-8"
        )
    )

    for metric in [
        "loss",
        "accuracy",
        "macro_f1",
    ]:
        plot_metric(
            history,
            metric,
            args.output,
        )

    print(
        f"Saved figures to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()