"""
MobileNetV2 Vision Diagnostic Classifier.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 architecture with modular classification head and backbone freezing.

    Suitable for high-efficiency image classification and future edge field diagnostics.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)

        # Backbone: feature extraction convolutions
        self.features = base_model.features

        if freeze_backbone:
            self.freeze_feature_extractor()

        # In_features for MobileNetV2 classifier head is 1280
        in_features = base_model.classifier[1].in_features

        # Replace classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def freeze_feature_extractor(self) -> None:
        """Freeze feature extractor backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze all feature extractor backbone parameters for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: Tensor of shape (B, 3, H, W)
        Returns logits: Tensor of shape (B, num_classes)
        """
        feat = self.features(x)
        # Global Average Pooling (pooling spatial dimensions to 1x1)
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


def build_vision_model(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> MobileNetV2Classifier:
    """Factory helper to construct vision classifier."""
    return MobileNetV2Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
