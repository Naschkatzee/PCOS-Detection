"""
Run a complete ResNet-50 experiment.

Training and learning curves:

    python -m src.training.run_resnet_experiment \
        --epochs 15 \
        --experiment-name resnet50_frozen_baseline

Final run with test-set evaluation:

    python -m src.training.run_resnet_experiment \
        --epochs 15 \
        --experiment-name resnet50_frozen_baseline \
        --evaluate-test


It should automatically:
→ create a new experiment directory
→ save history and checkpoint
→ mark the experiment completed
→ generate loss.png
→ generate accuracy.png
→ generate macro_f1.png
→ rebuild experiment_index.csv        
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.evaluate_resnet import evaluate_experiment
from src.experiments.build_index import build_experiment_index
from src.training.train_resnet import train
from src.visualisation.plot_training_history import (
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, visualize, and optionally evaluate "
            "a ResNet-50 experiment."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/raw/ovarian_ultrasound"),
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("dataset/splits"),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="resnet50_frozen_baseline",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Evaluate the best checkpoint on the test set. "
            "Use this only for finalized experiments."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    experiment = train(arguments)

    print("\nGenerating training visualizations...")

    plot_training_history(experiment)

    if arguments.evaluate_test:
        print("\nEvaluating the best checkpoint on the test set...")

        evaluate_experiment(experiment)
    else:
        print(
            "\nTest evaluation skipped. "
            "Use --evaluate-test only for a finalized run."
        )

    index_path = Path(
        "reports/comparisons/experiment_index.csv"
    )

    build_experiment_index(
        experiments_root=arguments.experiments_root,
        output_path=index_path,
    )

    print("\nExperiment workflow completed.")
    print(f"Experiment directory: {experiment.root.resolve()}")


if __name__ == "__main__":
    main()