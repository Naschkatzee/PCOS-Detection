'''
defines the non-foundation baseline model
It loads ImageNet-pretrained ResNet-50 weights and replaces the original 1000-class classification 
layer with a 3-class head. 
TorchVision’s current API uses ResNet50_Weights.DEFAULT for the recommended pretrained weights (?)
The frozen-backbone mode corresponds to using the pretrained network as a fixed feature extractor


python -c "import torch; from src.models.resnet import create_resnet50_classifier, count_parameters; model = create_resnet50_classifier(); x = torch.randn(2, 3, 224, 224); output = model(x); print(output.shape); print(count_parameters(model))"
'''


from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def create_resnet50_classifier(
    number_of_classes: int = 3,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Create an ImageNet-pretrained ResNet-50 classification baseline.

    Args:
        number_of_classes:
            Number of output classes.

        freeze_backbone:
            If True, freeze the pretrained ResNet feature extractor and train
            only the new classification head.

        dropout:
            Dropout probability before the final linear layer.

    Returns:
        Configured ResNet-50 model.
    """
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1).")

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_features, number_of_classes),
    )

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters."""
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total_parameters,
        "trainable": trainable_parameters,
        "frozen": total_parameters - trainable_parameters,
    }