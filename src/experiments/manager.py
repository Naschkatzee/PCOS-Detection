'''
python -c "from src.experiments.manager import create_experiment; experiment = create_experiment('ResNet50 Frozen Test'); print(experiment.root); print(experiment.checkpoint); print(experiment.figures)"
'''

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem paths belonging to one experiment."""

    root: Path
    config: Path
    history: Path
    checkpoint: Path
    metrics: Path
    predictions: Path
    figures: Path
    confusion_matrix: Path
    logs: Path
    xai: Path


def sanitize_name(name: str) -> str:
    """
    Convert an experiment name into a filesystem-safe identifier.

    Example:
        "ResNet50 Frozen Baseline" -> "resnet50_frozen_baseline"
    """
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")

    if not normalized:
        raise ValueError("Experiment name must contain letters or numbers.")

    return normalized


def create_experiment(
    name: str,
    base_dir: Path | str = "experiments",
) -> ExperimentPaths:
    """
    Create a new timestamped experiment directory.

    Existing experiments are never overwritten.
    """
    base_dir = Path(base_dir)
    safe_name = sanitize_name(name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_root = base_dir / f"{timestamp}_{safe_name}"

    counter = 1
    while experiment_root.exists():
        experiment_root = base_dir / (
            f"{timestamp}_{safe_name}_{counter:02d}"
        )
        counter += 1

    figures_dir = experiment_root / "figures"
    xai_dir = experiment_root / "xai"

    figures_dir.mkdir(parents=True, exist_ok=False)
    xai_dir.mkdir(parents=True, exist_ok=False)

    return ExperimentPaths(
        root=experiment_root,
        config=experiment_root / "config.json",
        history=experiment_root / "history.json",
        checkpoint=experiment_root / "best_checkpoint.pt",
        metrics=experiment_root / "metrics.json",
        predictions=experiment_root / "predictions.csv",
        figures=figures_dir,
        confusion_matrix=figures_dir / "confusion_matrix.png",
        logs=experiment_root / "logs.txt",
        xai=xai_dir,
    )


def load_experiment(
    experiment_dir: Path | str,
) -> ExperimentPaths:
    """
    Return the standard paths for an existing experiment.
    """
    root = Path(experiment_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: {root}"
        )

    return ExperimentPaths(
        root=root,
        config=root / "config.json",
        history=root / "history.json",
        checkpoint=root / "best_checkpoint.pt",
        metrics=root / "metrics.json",
        predictions=root / "predictions.csv",
        figures=root / "figures",
        confusion_matrix=root / "figures" / "confusion_matrix.png",
        logs=root / "logs.txt",
        xai=root / "xai",
    )


def save_json(
    path: Path,
    payload: Any,
) -> None:
    """Save JSON data, creating parent directories when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )


def append_log(
    path: Path,
    message: str,
) -> None:
    """Append a timestamped line to an experiment log file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message}\n")