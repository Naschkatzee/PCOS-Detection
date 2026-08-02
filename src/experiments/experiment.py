from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn


def sanitize_name(name: str) -> str:
    """Convert an experiment name into a filesystem-safe identifier."""
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")

    if not normalized:
        raise ValueError(
            "Experiment name must contain at least one letter or number."
        )

    return normalized


@dataclass
class Experiment:
    """Manage all files belonging to one machine-learning experiment."""

    root: Path
    started_at: datetime

    @classmethod
    def create(
        cls,
        name: str,
        base_dir: Path | str = "experiments",
    ) -> Experiment:
        """Create a new experiment directory without overwriting older runs."""
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

        experiment = cls(
            root=experiment_root,
            started_at=datetime.now(),
        )

        experiment.figures_dir.mkdir(parents=True, exist_ok=False)
        experiment.xai_dir.mkdir(parents=True, exist_ok=False)

        experiment.save_status(
            status="running",
            finished_at=None,
            duration_seconds=None,
            error=None,
        )

        experiment.log("Experiment created.")

        return experiment

    @classmethod
    def load(
        cls,
        experiment_dir: Path | str,
    ) -> Experiment:
        """Load an existing experiment directory."""
        root = Path(experiment_dir)

        if not root.exists():
            raise FileNotFoundError(
                f"Experiment directory does not exist: {root}"
            )

        status_path = root / "status.json"
        started_at = datetime.now()

        if status_path.exists():
            status = json.loads(
                status_path.read_text(encoding="utf-8")
            )

            started_value = status.get("started_at")

            if started_value:
                started_at = datetime.fromisoformat(started_value)

        return cls(
            root=root,
            started_at=started_at,
        )

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def history_path(self) -> Path:
        return self.root / "history.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "best_checkpoint.pt"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.csv"

    @property
    def log_path(self) -> Path:
        return self.root / "logs.txt"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def xai_dir(self) -> Path:
        return self.root / "xai"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.figures_dir / "confusion_matrix.png"

    def save_json(
        self,
        path: Path,
        payload: Any,
    ) -> None:
        """Save a JSON-serializable value."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                default=str,
            )

    def save_config(self, config: dict[str, Any]) -> None:
        """Save the experiment configuration."""
        self.save_json(self.config_path, config)

    def save_history(
        self,
        history: list[dict[str, Any]],
    ) -> None:
        """Save training history."""
        self.save_json(self.history_path, history)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Save evaluation metrics."""
        self.save_json(self.metrics_path, metrics)

    def save_predictions(
        self,
        predictions: pd.DataFrame,
    ) -> None:
        """Save per-image model predictions."""
        predictions.to_csv(
            self.predictions_path,
            index=False,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_metrics: dict[str, float],
        arguments: dict[str, Any],
    ) -> None:
        """Save the best model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": validation_metrics,
                "arguments": arguments,
            },
            self.checkpoint_path,
        )

    def log(self, message: str) -> None:
        """Append a timestamped message to the experiment log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

    def save_status(
        self,
        status: str,
        finished_at: datetime | None,
        duration_seconds: float | None,
        error: str | None,
    ) -> None:
        """Save the current experiment status."""
        payload = {
            "status": status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": (
                finished_at.isoformat(timespec="seconds")
                if finished_at is not None
                else None
            ),
            "duration_seconds": duration_seconds,
            "error": error,
        }

        self.save_json(self.status_path, payload)

    def complete(self) -> None:
        """Mark the experiment as successfully completed."""
        finished_at = datetime.now()
        duration = (finished_at - self.started_at).total_seconds()

        self.save_status(
            status="completed",
            finished_at=finished_at,
            duration_seconds=duration,
            error=None,
        )

        self.log(
            f"Experiment completed in {duration:.1f} seconds."
        )

    def fail(self, error: Exception) -> None:
        """Mark the experiment as failed."""
        finished_at = datetime.now()
        duration = (finished_at - self.started_at).total_seconds()

        self.save_status(
            status="failed",
            finished_at=finished_at,
            duration_seconds=duration,
            error=f"{type(error).__name__}: {error}",
        )

        self.log(
            f"Experiment failed: {type(error).__name__}: {error}"
        )