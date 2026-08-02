"""
Resize(224, 224) makes every image the same size, which is required for batching.

RandomHorizontalFlip is commonly reasonable for ovarian ultrasound because left-right orientation 
is usually not the class-defining feature. We will still mention it explicitly in the methodology.

RandomRotation(5°) adds only a small variation. I would not use large rotations because ultrasound 
anatomy and acquisition orientation matter.

ToTensor() converts a PIL image from shape:

height × width × channels

into a PyTorch tensor with shape:

channels × height × width

and scales pixel values from 0–255 to 0–1.

Normalize uses ImageNet statistics because several of our pretrained models expect ImageNet-like 
normalized input.

Important limitation

This file defines a shared initial preprocessing pipeline. Some foundation models later require 
their own official processor:

DINOv2 may use a Hugging Face image processor.
BiomedCLIP has model-specific normalization.
USFM and OpenUS may have their own input preprocessing.

We should not force every model to use ImageNet normalization if its official checkpoint expects 
something different. For the first data pipeline and ResNet baseline, this file is correct. 
Later, each foundation-model adapter can override the normalization while keeping the dataset split 
identical.

To run:
python -c "from src.data.transforms import create_train_transforms; print(create_train_transforms())"
"""

from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Create training transforms for ovarian ultrasound images.

    The augmentation is deliberately conservative because strong geometric
    transformations could change clinically relevant anatomical structures.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size=(image_size, image_size),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def create_evaluation_transforms(
    image_size: int = 224,
) -> transforms.Compose:
    """
    Create deterministic transforms for validation and test images.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size=(image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )