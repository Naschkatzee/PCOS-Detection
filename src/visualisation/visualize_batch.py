'''
Quality assurance and debugging tool.

Before training any model, we want to be sure that the data entering the network is correct. 
visualize_batch.py lets to  inspect a random training batch after all preprocessing and augmentations 
have been applied. This helps answer questions such as: Are the images loaded correctly? 
Are the labels correct? Does the resizing distort the anatomy? Are the rotations too strong? 
Are the horizontal flips acceptable? 
If something is wrong here, the model will learn from incorrect data, 
and debugging later becomes much harder.


python -m src.visualisation.visualize_batch

'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataloaders import create_dataloaders
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization for visualization.

    Args:
        images: Tensor with shape [batch, channels, height, width].

    Returns:
        Tensor with pixel values clipped to the range [0, 1].
    """
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)

    images = images * std + mean
    return images.clamp(0, 1)


def visualize_batch(
    output_path: Path,
    batch_size: int = 8,
    image_size: int = 224,
    columns: int = 4,
    show: bool = True,
) -> None:
    """Load and visualize one augmented training batch."""
    train_loader, _, _ = create_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    images = denormalize(batch["image"])
    labels = batch["label_name"]
    paths = batch["path"]

    rows = int(np.ceil(len(images) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
    )

    axes_array = np.asarray(axes).reshape(-1)

    for index, axis in enumerate(axes_array):
        axis.axis("off")

        if index >= len(images):
            continue

        image = images[index].permute(1, 2, 0).cpu().numpy()

        axis.imshow(image)
        axis.set_title(
            f"{labels[index]}\n{Path(paths[index]).name}",
            fontsize=9,
        )

    figure.suptitle(
        "Augmented ovarian ultrasound training batch",
        fontsize=14,
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved batch visualization to: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one ovarian ultrasound training batch."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/training_batch.png"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a display window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    visualize_batch(
        output_path=args.output,
        batch_size=args.batch_size,
        image_size=args.image_size,
        columns=args.columns,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()