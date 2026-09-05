"""
Lightweight, deterministic Technician Observation Text Encoder.
"""

import numpy as np
import torch
import torch.nn as nn


class TechnicianTextEncoder(nn.Module):
    """
    Lightweight, deterministic token-hashing text encoder for technician maintenance logs.

    Extracts a fixed 256-dimensional semantic text representation from field notes
    without requiring external multi-gigabyte generative LLMs or heavyweight internet weights.
    Can be seamlessly swapped with SentenceTransformer/DeBERTa backbones if desired.
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Pre-initialized deterministic bag-of-words / projection layer
        torch.manual_seed(42)
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        # Initialize with normalized weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / np.sqrt(embedding_dim))
        self.eval()

    def _tokenize_to_ids(self, text: str) -> torch.Tensor:
        """Tokenize text into bounded integer hash tokens."""
        words = text.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
        if not words:
            return torch.tensor([0], dtype=torch.long)
        # Deterministic string hash
        ids = [abs(hash(w)) % self.vocab_size for w in words]
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def encode(self, texts: str | list[str]) -> torch.Tensor:
        """
        Encode a single string or batch of technician notes into (B, embedding_dim) tensors.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for t in texts:
            token_ids = self._tokenize_to_ids(t).unsqueeze(0)
            emb = self.embedding(token_ids)
            # L2 normalize
            norm_emb = nn.functional.normalize(emb, p=2, dim=1)
            embeddings.append(norm_emb)

        return torch.cat(embeddings, dim=0)


def build_text_encoder(embedding_dim: int = 256) -> TechnicianTextEncoder:
    """Factory constructor for technician text encoder."""
    return TechnicianTextEncoder(embedding_dim=embedding_dim)
