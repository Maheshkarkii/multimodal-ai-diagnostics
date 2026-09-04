"""
Embedding Models and Representation Layer for Technical RAG.
Provides unified BaseEmbeddingModel interface, SentenceTransformers integration,
and robust deterministic dense hashing embeddings for zero-network/standalone environments.
"""

from abc import ABC, abstractmethod
import hashlib
import logging
from typing import List, Optional, Union
import numpy as np

from src.rag.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for all RAG embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate normalized 1D embedding vector for a single string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate normalized 2D embedding matrix (N, D) for a batch of strings."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass


class DeterministicDenseEmbeddingModel(BaseEmbeddingModel):
    """
    Fast, reproducible, multi-ngram deterministic hashing embedding model.
    Encodes subword tokens, words, and character n-grams into a fixed D-dimensional vector space.
    Guarantees zero network dependencies, perfect determinism, and fast cosine similarity.
    """

    def __init__(self, embedding_dim: int = 384, normalize: bool = True):
        self._dim = embedding_dim
        self.normalize = normalize

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)

        for i, text in enumerate(texts):
            words = text.lower().strip().split()
            if not words:
                continue

            for w in words:
                # Word-level hash features
                h1 = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self._dim
                h2 = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16) % self._dim
                vectors[i, h1] += 1.0
                vectors[i, h2] += 0.5

                # Character 3-gram features
                if len(w) >= 3:
                    for j in range(len(w) - 2):
                        tri = w[j:j+3]
                        h_tri = int(hashlib.sha1(tri.encode("utf-8")).hexdigest(), 16) % self._dim
                        vectors[i, h_tri] += 0.3

            # Normalization
            if self.normalize:
                norm = np.linalg.norm(vectors[i])
                if norm > 1e-8:
                    vectors[i] = vectors[i] / norm

        return vectors


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    HuggingFace / sentence-transformers dense neural embedding wrapper.
    Falls back gracefully to DeterministicDenseEmbeddingModel if model fails or network is offline.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self._dim = self.config.embedding_dim
        self._fallback_model = DeterministicDenseEmbeddingModel(
            embedding_dim=self.config.embedding_dim,
            normalize=self.config.normalize_embeddings,
        )

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.config.model_name_or_path,
                device=self.config.device,
            )
            # Query dimension
            test_emb = self.model.encode("test", show_progress_bar=False)
            self._dim = len(test_emb)
            logger.info(f"Loaded SentenceTransformer '{self.config.model_name_or_path}' with dim {self._dim}")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer '{self.config.model_name_or_path}' ({e}). "
                f"Operating using DeterministicDenseEmbeddingModel (dim={self._dim})."
            )
            self.model = None

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                )
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"SentenceTransformer encoding failed: {e}. Falling back to deterministic model.")

        return self._fallback_model.embed_documents(texts)


def create_embedding_model(config: EmbeddingConfig) -> BaseEmbeddingModel:
    """Factory function for instantiating the configured embedding engine."""
    model_type = config.model_type.lower()
    if model_type == "sentence_transformer":
        return SentenceTransformerEmbeddingModel(config)
    elif model_type in ["deterministic", "dense_hash"]:
        return DeterministicDenseEmbeddingModel(
            embedding_dim=config.embedding_dim,
            normalize=config.normalize_embeddings,
        )
    else:
        logger.warning(f"Unknown embedding model type: '{config.model_type}'. Using deterministic model.")
        return DeterministicDenseEmbeddingModel(
            embedding_dim=config.embedding_dim,
            normalize=config.normalize_embeddings,
        )
