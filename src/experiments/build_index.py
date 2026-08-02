"""
Build a CSV summary of all experiment directories.

Example:
    python -m src.experiments.build_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty dictionary if absent."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return payload


def read_experiment(
    experiment_dir: Path,
) -> dict[str, Any]:
    """Extract summary information from one experiment directory."""
    config = load_json(
        experiment_dir / "config.json"
    )
    status = load_json(
        experiment_dir / "status.json"
    )
    metrics = load_json(
        experiment_dir / "metrics.json"
    )

    test_metrics = metrics.get(
        "test_metrics",
        {},
    )

    validation_metrics = metrics.get(
        "checkpoint_validation_metrics",
        {},
    )

    return {
        "experiment": experiment_dir.name,
        "path": str(experiment_dir),
        "status": status.get(
            "status",
            "unknown",
        ),
        "model": config.get(
            "model",
            "unknown",
        ),
        "freeze_backbone": config.get(
            "freeze_backbone",
        ),
        "epochs": config.get(
            "epochs",
        ),
        "batch_size": config.get(
            "batch_size",
        ),
        "image_size": config.get(
            "image_size",
        ),
        "learning_rate": config.get(
            "learning_rate",
        ),
        "weight_decay": config.get(
            "weight_decay",
        ),
        "dropout": config.get(
            "dropout",
        ),
        "seed": config.get(
            "seed",
        ),
        "device": config.get(
            "device",
        ),
        "checkpoint_epoch": metrics.get(
            "checkpoint_epoch",
        ),
        "validation_accuracy": validation_metrics.get(
            "accuracy",
        ),
        "validation_macro_f1": validation_metrics.get(
            "macro_f1",
        ),
        "test_accuracy": test_metrics.get(
            "accuracy",
        ),
        "test_balanced_accuracy": test_metrics.get(
            "balanced_accuracy",
        ),
        "test_macro_precision": test_metrics.get(
            "macro_precision",
        ),
        "test_macro_recall": test_metrics.get(
            "macro_recall",
        ),
        "test_macro_f1": test_metrics.get(
            "macro_f1",
        ),
        "test_weighted_f1": test_metrics.get(
            "weighted_f1",
        ),
        "started_at": status.get(
            "started_at",
        ),
        "finished_at": status.get(
            "finished_at",
        ),
        "duration_seconds": status.get(
            "duration_seconds",
        ),
    }


def build_experiment_index(
    experiments_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Scan all experiment directories and save one summary CSV."""
    if not experiments_root.exists():
        raise FileNotFoundError(
            f"Experiments directory not found: {experiments_root}"
        )

    experiment_dirs = sorted(
        path
        for path in experiments_root.iterdir()
        if path.is_dir()
    )

    rows = [
        read_experiment(experiment_dir)
        for experiment_dir in experiment_dirs
    ]

    index = pd.DataFrame(rows)

    if not index.empty:
        index = index.sort_values(
            by="experiment"
        ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.to_csv(
        output_path,
        index=False,
    )

    return index


def create_readable_summary(index: pd.DataFrame) -> pd.DataFrame:
    """Create a compact experiment summary for human inspection."""
    if index.empty:
        return index.copy()

    summary_columns = [
        "experiment",
        "status",
        "model",
        "freeze_backbone",
        "epochs",
        "checkpoint_epoch",
        "validation_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "duration_seconds",
    ]

    available_columns = [
        column
        for column in summary_columns
        if column in index.columns
    ]

    summary = index[available_columns].copy()

    rename_columns = {
        "experiment": "Experiment",
        "status": "Status",
        "model": "Model",
        "freeze_backbone": "Frozen",
        "epochs": "Epochs",
        "checkpoint_epoch": "Best epoch",
        "validation_macro_f1": "Validation macro F1",
        "test_accuracy": "Test accuracy",
        "test_balanced_accuracy": "Test balanced accuracy",
        "test_macro_f1": "Test macro F1",
        "duration_seconds": "Duration (s)",
    }

    summary = summary.rename(columns=rename_columns)

    metric_columns = [
        "Validation macro F1",
        "Test accuracy",
        "Test balanced accuracy",
        "Test macro F1",
    ]

    for column in metric_columns:
        if column in summary.columns:
            summary[column] = summary[column].round(4)

    integer_columns = [
        "Epochs",
        "Best epoch",
    ]

    for column in integer_columns:
        if column in summary.columns:
            summary[column] = (
                summary[column]
                .astype("Int64")
            )

    if "Duration (s)" in summary.columns:
        summary["Duration (s)"] = (
            summary["Duration (s)"].round(1)
        )

    return summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV index of all experiments."
    )

    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/comparisons/experiment_index.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    index = build_experiment_index(
        experiments_root=arguments.experiments_root,
        output_path=arguments.output,
    )

    readable_summary = create_readable_summary(index)

    readable_output = (
        arguments.output.parent
        / "experiment_summary.csv"
    )

    readable_summary.to_csv(
        readable_output,
        index=False,
    )

    markdown_output = (
        arguments.output.parent
        / "experiment_summary.md"
    )

    markdown_output.write_text(
        readable_summary.to_markdown(
            index=False,
            missingval="-",
        ),
        encoding="utf-8",
    )

    print(
        "Saved Markdown summary to: "
        f"{markdown_output.resolve()}"
    )

    print(
        f"Saved readable summary to: "
        f"{readable_output.resolve()}"
    )

    if not readable_summary.empty:
        print()
        print(
            readable_summary.to_string(
                index=False,
                na_rep="-",
            )
        )

    print(
        f"Indexed {len(index)} experiments."
    )
    print(
        f"Saved experiment index to: "
        f"{arguments.output.resolve()}"
    )


if __name__ == "__main__":
    main()