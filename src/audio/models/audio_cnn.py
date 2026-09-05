"""
Acoustic 2D CNN with intermediate feature embedding extraction.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AcousticCNN(nn.Module):
    """
    Acoustic Spectrogram Classification CNN.

    Processes Log-Mel Spectrogram representations (B, 1, n_mels, time_frames) and provides:
    - Diagnostic anomaly classification logits
    - 512-dimensional acoustic feature embedding extraction for cross-modal fusion.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        embedding_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # 4-stage convolutional hierarchical feature extractor
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        # Adaptive global average pooling converts arbitrary time-frequency dimensions to (B, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Acoustic Embedding Projection Layer (256 -> embedding_dim)
        self.embedding_layer = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract acoustic feature embedding vector.

        Args:
            x: Spectrogram tensor (B, 1, n_mels, time_frames)

        Returns:
            embedding: Tensor of shape (B, embedding_dim)
        """
        feat = self.features(x)
        feat = self.global_pool(feat)
        flat_feat = torch.flatten(feat, 1)
        embedding = self.embedding_layer(flat_feat)
        return embedding

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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


def build_audio_model(
    num_classes: int = 5,
    in_channels: int = 1,
    embedding_dim: int = 512,
    dropout: float = 0.3,
) -> AcousticCNN:
    """Factory helper for acoustic classifier."""
    return AcousticCNN(
        num_classes=num_classes,
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
