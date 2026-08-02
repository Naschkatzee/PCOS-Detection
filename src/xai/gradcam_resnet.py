"""
Generate Grad-CAM explanations for a completed ResNet-50 experiment.

Example:
    python -m src.xai.gradcam_resnet ^
        --experiment-dir "experiments/2026-08-02_17-45-58_resnet50_frozen_baseline"

    python -m src.xai.gradcam_resnet ^
    --experiment-dir "experiments/2026-08-02_17-45-58_resnet50_frozen_baseline" ^
    --include-errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.nn import functional as F

from src.data.dataset import INDEX_TO_LABEL, LABEL_TO_INDEX
from src.data.transforms import create_evaluation_transforms
from src.experiments import Experiment
from src.models.resnet import create_resnet50_classifier


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return payload


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate saved test predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}"
        )

    predictions = pd.read_csv(path)

    required_columns = {
        "path",
        "true_index",
        "true_label",
        "predicted_index",
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


def load_model(
    experiment: Experiment,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load the best ResNet-50 checkpoint."""
    if not experiment.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {experiment.checkpoint_path}"
        )

    checkpoint = torch.load(
        experiment.checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_resnet50_classifier(
        number_of_classes=len(LABEL_TO_INDEX),
        freeze_backbone=bool(
            config.get("freeze_backbone", True)
        ),
        dropout=float(config.get("dropout", 0.2)),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def select_examples(
    predictions: pd.DataFrame,
    examples_per_class: int,
    include_errors: bool,
) -> pd.DataFrame:
    """
    Select representative correct examples and optional errors.

    Correct examples:
        highest-confidence correct prediction for each class.

    Incorrect examples:
        highest-confidence error for each true class, where available.
    """
    selected_parts: list[pd.DataFrame] = []

    correct_predictions = predictions[
        predictions["correct"]
    ]

    for class_name in LABEL_TO_INDEX:
        class_rows = correct_predictions[
            correct_predictions["true_label"] == class_name
        ].sort_values(
            "confidence",
            ascending=False,
        )

        selected_parts.append(
            class_rows.head(examples_per_class)
        )

    if include_errors:
        incorrect_predictions = predictions[
            ~predictions["correct"]
        ]

        for class_name in LABEL_TO_INDEX:
            class_rows = incorrect_predictions[
                incorrect_predictions["true_label"] == class_name
            ].sort_values(
                "confidence",
                ascending=False,
            )

            selected_parts.append(
                class_rows.head(1)
            )

    non_empty_parts = [
        part
        for part in selected_parts
        if not part.empty
    ]

    if not non_empty_parts:
        raise RuntimeError(
            "No prediction examples could be selected."
        )

    return pd.concat(
        non_empty_parts,
        ignore_index=True,
    )


def prepare_image(
    image_path: Path,
    image_size: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Load one image for display and model inference.

    Returns:
        display_image:
            Float RGB image in [0, 1].

        input_tensor:
            Normalized tensor with shape [1, 3, H, W].
    """
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        display_image = np.asarray(
            rgb_image.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0

        transform = create_evaluation_transforms(
            image_size=image_size
        )

        input_tensor = transform(
            rgb_image
        ).unsqueeze(0)

    return display_image, input_tensor


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    display_image: np.ndarray,
    target_class: int,
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    """Generate one Grad-CAM overlay."""
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        predicted_index = int(
            probabilities.argmax(dim=1).item()
        )

        predicted_confidence = float(
            probabilities[0, predicted_index].item()
        )

    target_layers = [
        model.layer4[-1]
    ]

    targets = [
        ClassifierOutputTarget(target_class)
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

    overlay = show_cam_on_image(
        display_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.70, 
    )

    return (
        overlay,
        predicted_confidence,
        predicted_index,
    )


def save_explanation(
    original_image: np.ndarray,
    overlay: np.ndarray,
    row: pd.Series,
    target_label: str,
    experiment_name: str,
    output_path: Path,
) -> None:
    """Save the original image beside its Grad-CAM overlay."""
    true_label = str(
        row["true_label"]
    ).replace("_", " ")

    predicted_label = str(
        row["predicted_label"]
    ).replace("_", " ")

    confidence = float(row["confidence"])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original ultrasound")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM for: "
        f"{target_label.replace('_', ' ')}"
    )
    axes[1].axis("off")

    figure.suptitle(
        (
            f"True: {true_label} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.1%}"
        ),
        fontsize=12,
    )

    figure.text(
        0.5,
        0.02,
        f"Experiment: {experiment_name}",
        ha="center",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.93)
    )

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


def create_gradcam_explanations(
    experiment: Experiment,
    examples_per_class: int,
    include_errors: bool,
    target_mode: str,
) -> None:
    """Generate Grad-CAM figures for selected test images."""
    config = load_json(
        experiment.config_path
    )

    predictions = load_predictions(
        experiment.predictions_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = load_model(
        experiment=experiment,
        config=config,
        device=device,
    )

    selected = select_examples(
        predictions=predictions,
        examples_per_class=examples_per_class,
        include_errors=include_errors,
    )

    image_root = Path(
        config["dataset_root"]
    )

    image_size = int(
        config.get("image_size", 224)
    )

    output_dir = (
        experiment.xai_dir
        / "gradcam"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows: list[dict[str, Any]] = []

    for example_number, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        relative_path = Path(
            str(row["path"])
        )

        image_path = (
            image_root
            / relative_path
        )

        original_image, input_tensor = prepare_image(
            image_path=image_path,
            image_size=image_size,
        )

        if target_mode == "predicted":
            target_index = int(
                row["predicted_index"]
            )
        else:
            target_index = int(
                row["true_index"]
            )

        target_label = INDEX_TO_LABEL[
            target_index
        ]

        overlay, confidence, predicted_index = (
            generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                display_image=original_image,
                target_class=target_index,
                device=device,
            )
        )

        correctness = (
            "correct"
            if bool(row["correct"])
            else "incorrect"
        )

        output_name = (
            f"{example_number:02d}_"
            f"{correctness}_"
            f"true_{row['true_label']}_"
            f"pred_{row['predicted_label']}.png"
        )

        output_path = (
            output_dir
            / output_name
        )

        save_explanation(
            original_image=original_image,
            overlay=overlay,
            row=row,
            target_label=target_label,
            experiment_name=experiment.root.name,
            output_path=output_path,
        )

        manifest_rows.append(
            {
                "source_path": str(relative_path),
                "true_label": row["true_label"],
                "predicted_label": (
                    row["predicted_label"]
                ),
                "correct": bool(row["correct"]),
                "saved_confidence": float(
                    row["confidence"]
                ),
                "recomputed_confidence": confidence,
                "target_mode": target_mode,
                "gradcam_target_label": target_label,
                "output_path": str(output_path),
            }
        )

        print(f"Saved: {output_path}")

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        output_dir / "manifest.csv",
        index=False,
    )

    experiment.log(
        (
            "Generated Grad-CAM explanations "
            f"for {len(manifest)} test images."
        )
    )

    print(
        f"\nGrad-CAM explanations saved to: "
        f"{output_dir.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM explanations for "
            "a completed ResNet-50 experiment."
        )
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=1,
        help=(
            "Number of correctly classified examples "
            "selected per class."
        ),
    )

    parser.add_argument(
        "--include-errors",
        action="store_true",
        help=(
            "Also include the highest-confidence "
            "incorrect prediction for each true class."
        ),
    )

    parser.add_argument(
        "--target-mode",
        choices=[
            "predicted",
            "true",
        ],
        default="predicted",
        help=(
            "Generate Grad-CAM for the predicted class "
            "or the true class."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    experiment = Experiment.load(
        arguments.experiment_dir
    )

    create_gradcam_explanations(
        experiment=experiment,
        examples_per_class=arguments.examples_per_class,
        include_errors=arguments.include_errors,
        target_mode=arguments.target_mode,
    )


if __name__ == "__main__":
    main()