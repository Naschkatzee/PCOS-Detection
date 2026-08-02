"""
Run with:

python -m src.training.train_resnet --epochs 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataloaders import create_dataloaders
from src.experiments.manager import (
    append_log,
    create_experiment,
    save_json,
)
from src.models.resnet import create_resnet50_classifier


def set_seed(seed: int) -> None:
    """Configure reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(
        data_loader,
        desc="Train" if is_training else "Validation",
        leave=False,
    )

    for batch in progress:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        targets = batch["label"].to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = logits.argmax(dim=1)

        current_batch_size = targets.size(0)
        total_loss += loss.item() * current_batch_size
        total_correct += (predictions == targets).sum().item()
        total_samples += current_batch_size

        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise RuntimeError("The DataLoader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    arguments: argparse.Namespace,
) -> None:
    """Save a reusable training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
            "arguments": vars(arguments),
        },
        path,
    )


def train(arguments: argparse.Namespace) -> None:
    """Train the frozen ResNet-50 baseline."""
    set_seed(arguments.seed)

    experiment = create_experiment(
        name=arguments.experiment_name,
        base_dir=arguments.experiments_root,
    )

    print(f"Experiment directory: {experiment.root.resolve()}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = {
        "experiment_name": arguments.experiment_name,
        "model": "resnet50",
        "pretrained": True,
        "freeze_backbone": True,
        "dataset_root": str(arguments.dataset_root),
        "splits_root": str(arguments.splits_root),
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "image_size": arguments.image_size,
        "num_workers": arguments.num_workers,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "dropout": arguments.dropout,
        "seed": arguments.seed,
        "device": str(device),
    }

    save_json(experiment.config, config)
    append_log(experiment.logs, "Experiment created.")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, validation_loader, _ = create_dataloaders(
        dataset_root=arguments.dataset_root,
        splits_root=arguments.splits_root,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )

    model = create_resnet50_classifier(
        number_of_classes=3,
        freeze_backbone=True,
        dropout=arguments.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    print(
        "Trainable parameter tensors: "
        f"{len(trainable_parameters)}"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    history: list[dict[str, Any]] = []
    best_validation_f1 = float("-inf")

    for epoch in range(1, arguments.epochs + 1):
        print(f"\nEpoch {epoch}/{arguments.epochs}")

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_result)

        save_json(experiment.history, history)

        print(
            "Train — "
            f"loss: {train_metrics['loss']:.4f}, "
            f"accuracy: {train_metrics['accuracy']:.4f}, "
            f"macro F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            "Validation — "
            f"loss: {validation_metrics['loss']:.4f}, "
            f"accuracy: {validation_metrics['accuracy']:.4f}, "
            f"macro F1: {validation_metrics['macro_f1']:.4f}"
        )

        append_log(
            experiment.logs,
            (
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_macro_f1={train_metrics['macro_f1']:.4f}, "
                f"validation_loss={validation_metrics['loss']:.4f}, "
                f"validation_macro_f1="
                f"{validation_metrics['macro_f1']:.4f}"
            ),
        )

        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]

            save_checkpoint(
                path=experiment.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                arguments=arguments,
            )

            print(
                "Saved new best checkpoint with "
                f"validation macro F1 = "
                f"{best_validation_f1:.4f}"
            )

            append_log(
                experiment.logs,
                (
                    f"Saved best checkpoint at epoch {epoch}; "
                    f"validation macro F1="
                    f"{best_validation_f1:.4f}"
                ),
            )

    append_log(
        experiment.logs,
        (
            "Training completed. "
            f"Best validation macro F1: "
            f"{best_validation_f1:.4f}"
        ),
    )

    print(f"\nTraining history saved to: {experiment.history}")
    print(f"Best checkpoint saved to: {experiment.checkpoint}")
    print(f"Experiment saved to: {experiment.root.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen ResNet-50 baseline."
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

    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()