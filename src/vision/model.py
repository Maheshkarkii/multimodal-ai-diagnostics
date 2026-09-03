"""
MobileNetV2 Vision Diagnostic Classifier & Feature Embedding Extractor.
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 with configurable backbone freezing, fine-tuning, and feature embedding extraction.

    Exposes intermediate representations (1280-dim embedding) for downstream multimodal fusion.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layers: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers
        self.embedding_dim = 1280

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)

        # Feature extractor layers (inverted residual conv blocks 0 to 18)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Apply freezing / fine-tuning policy
        self._configure_parameter_freezing(freeze_backbone, unfreeze_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )

    def _configure_parameter_freezing(
        self, freeze_backbone: bool, unfreeze_layers: Optional[int]
    ) -> None:
        """Configure layer freezing for transfer learning vs fine-tuning."""
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

            if unfreeze_layers and unfreeze_layers > 0:
                # Unfreeze the last N layers/blocks of the feature extractor
                total_blocks = len(self.features)
                unfreeze_start = max(0, total_blocks - unfreeze_layers)
                for idx in range(unfreeze_start, total_blocks):
                    for param in self.features[idx].parameters():
                        param.requires_grad = True
        else:
            for param in self.features.parameters():
                param.requires_grad = True

    def freeze_feature_extractor(self) -> None:
        """Freeze all backbone feature layers."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze all backbone feature layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature embedding before classification head.

        Args:
            x: Tensor of shape (B, 3, H, W)

        Returns:
            embedding: Tensor of shape (B, 1280)
        """
        feat = self.features(x)
        feat = self.pool(feat)
        embedding = torch.flatten(feat, 1)
        return embedding

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)
            return_features: If True, returns (logits, feature_embeddings) tuple.

        Returns:
            logits or (logits, embeddings)
        """
        embeddings = self.extract_features(x)
        logits = self.classifier(embeddings)

        if return_features:
            return logits, embeddings
        return logits


def build_vision_model(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    unfreeze_layers: Optional[int] = None,
    dropout: float = 0.2,
) -> MobileNetV2Classifier:
    """Factory constructor for industrial vision classifier."""
    return MobileNetV2Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_layers=unfreeze_layers,
        dropout=dropout,
    )
