"""
Multimodal Fusion Network: Modality Projections, Missing-Modality Masking, and Unified Machine Representation.
"""

import torch
import torch.nn as nn


class ModalityProjection(nn.Module):
    """Projects arbitrary raw modality embedding into unified shared dimension."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Machine State Diagnostic Network.

    Architecture:
    1. Modality-specific Projections (Vision: 1280->256, Audio: 512->256, Sensor: 256->256, Text: 256->256)
    2. Modality Masking & Presence Weighting (Handles missing inputs gracefully)
    3. Concatenation of available representations (B, 4 * 256)
    4. Deep Fusion MLP (1024 -> 512 -> 256)
    5. Exposes 256-dim Unified Machine Embedding
    6. Classification Head (256 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 5,
        vision_dim: int = 1280,
        audio_dim: int = 512,
        sensor_dim: int = 256,
        text_dim: int = 256,
        shared_dim: int = 256,
        fusion_hidden_dims: list[int] | None = None,
        unified_embedding_dim: int = 256,
        dropout: float = 0.25,
        modality_dropout_prob: float = 0.20,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.shared_dim = shared_dim
        self.modality_dropout_prob = modality_dropout_prob
        self.modalities = ["vision", "audio", "sensor", "text"]

        # 1. Independent Modality Projections
        self.proj_vision = ModalityProjection(vision_dim, shared_dim, dropout)
        self.proj_audio = ModalityProjection(audio_dim, shared_dim, dropout)
        self.proj_sensor = ModalityProjection(sensor_dim, shared_dim, dropout)
        self.proj_text = ModalityProjection(text_dim, shared_dim, dropout)

        # 2. Learnable Missing-Modality Default Tokens
        self.missing_tokens = nn.ParameterDict({m: nn.Parameter(torch.zeros(1, shared_dim)) for m in self.modalities})

        # 3. Fusion Backbone
        fusion_hidden_dims = fusion_hidden_dims or [512, 256]
        in_dim = len(self.modalities) * shared_dim  # 4 * 256 = 1024

        fusion_layers = []
        prev_dim = in_dim
        for h_dim in fusion_hidden_dims:
            fusion_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = h_dim

        # Unified embedding projection (e.g. 256)
        fusion_layers.extend(
            [
                nn.Linear(prev_dim, unified_embedding_dim),
                nn.BatchNorm1d(unified_embedding_dim),
                nn.ReLU(inplace=True),
            ]
        )
        self.fusion_trunk = nn.Sequential(*fusion_layers)

        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(unified_embedding_dim, num_classes),
        )

    def _apply_modality_dropout(self, masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply random modality dropout during training to prevent single-modality reliance."""
        if not self.training or self.modality_dropout_prob <= 0.0:
            return masks

        updated_masks = {}
        for m, mask in masks.items():
            # Random drop tensor (B, 1)
            drop = (torch.rand_like(mask.float()) >= self.modality_dropout_prob).long()
            updated_masks[m] = mask * drop

        # Ensure at least one modality remains active per sample
        stacked = torch.stack(list(updated_masks.values()), dim=1)  # (B, num_modalities, 1)
        zero_rows = (stacked.sum(dim=1) == 0).squeeze(-1)
        if zero_rows.any():
            for idx in torch.where(zero_rows)[0]:
                active_mod = self.modalities[torch.randint(0, len(self.modalities), (1,)).item()]
                updated_masks[active_mod][idx] = 1

        return updated_masks

    def extract_unified_embedding(
        self,
        embeddings: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Fuse available modalities and extract unified 256-dim machine representation.
        """
        batch_size = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device

        # Default masks: all present (1)
        if masks is None:
            masks = {m: torch.ones(batch_size, 1, dtype=torch.long, device=device) for m in self.modalities}

        masks = self._apply_modality_dropout(masks)

        # Project each modality
        p_vision = self.proj_vision(embeddings["vision"])
        p_audio = self.proj_audio(embeddings["audio"])
        p_sensor = self.proj_sensor(embeddings["sensor"])
        p_text = self.proj_text(embeddings["text"])

        proj_map = {
            "vision": p_vision,
            "audio": p_audio,
            "sensor": p_sensor,
            "text": p_text,
        }

        # Apply presence masks and replace missing with learnable tokens
        fused_components = []
        for m in self.modalities:
            proj = proj_map[m]
            m_mask = masks[m].float()
            token = self.missing_tokens[m].expand(batch_size, -1)
            # Masked blending: if mask==1 -> proj, if mask==0 -> token
            blended = proj * m_mask + token * (1.0 - m_mask)
            fused_components.append(blended)

        # Concatenation of representations: (B, 1024)
        cat_features = torch.cat(fused_components, dim=1)
        unified_emb = self.fusion_trunk(cat_features)
        return unified_emb

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits or (logits, unified_machine_embedding)
        """
        unified_embedding = self.extract_unified_embedding(embeddings, masks=masks)
        logits = self.classifier(unified_embedding)

        if return_features:
            return logits, unified_embedding
        return logits


def build_multimodal_model(
    num_classes: int = 5,
    vision_dim: int = 1280,
    audio_dim: int = 512,
    sensor_dim: int = 256,
    text_dim: int = 256,
    shared_dim: int = 256,
    fusion_hidden_dims: list[int] | None = None,
    unified_embedding_dim: int = 256,
    dropout: float = 0.25,
    modality_dropout_prob: float = 0.20,
) -> MultimodalFusionModel:
    """Factory builder for multimodal fusion model."""
    return MultimodalFusionModel(
        num_classes=num_classes,
        vision_dim=vision_dim,
        audio_dim=audio_dim,
        sensor_dim=sensor_dim,
        text_dim=text_dim,
        shared_dim=shared_dim,
        fusion_hidden_dims=fusion_hidden_dims,
        unified_embedding_dim=unified_embedding_dim,
        dropout=dropout,
        modality_dropout_prob=modality_dropout_prob,
    )
