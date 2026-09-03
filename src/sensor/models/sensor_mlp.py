"""
Sensor State Classification MLP with 256-dim Intermediate Embedding Extraction.
"""

from typing import Tuple, List, Optional, Union
import torch
import torch.nn as nn


class SensorMLP(nn.Module):
    """
    Multilayer Perceptron for structured multivariate industrial telemetry.

    Provides:
    - Diagnostic state classification logits (B, num_classes)
    - Intermediate 256-dimensional sensor feature embedding extraction for cross-modal fusion.
    """

    def __init__(
        self,
        in_features: int = 6,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        hidden_dims = hidden_dims or [128, 256, 128]

        # 1. Feature Representation Trunk
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        # Final projection to fixed embedding_dim (256)
        layers.extend([
            nn.Linear(prev_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        ])
        self.encoder = nn.Sequential(*layers)

        # 2. Diagnostic Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract structured sensor embedding vector.

        Args:
            x: Tensor of standardized telemetry (B, in_features)

        Returns:
            embedding: Tensor of shape (B, embedding_dim)
        """
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Returns:
            logits or (logits, embeddings)
        """
        embeddings = self.extract_features(x)
        logits = self.classifier(embeddings)

        if return_features:
            return logits, embeddings
        return logits


def build_sensor_model(
    in_features: int = 6,
    num_classes: int = 5,
    hidden_dims: Optional[List[int]] = None,
    embedding_dim: int = 256,
    dropout: float = 0.2,
) -> SensorMLP:
    """Factory builder for SensorMLP."""
    return SensorMLP(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
