"""
Persistent Vector Database Storage.
Provides abstract BaseVectorStore and high-performance persistent NumpyFlatVectorStore
supporting cosine, inner-product, and Euclidean search with metadata filtering.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.rag.config import VectorStoreConfig
from src.rag.schema import DocumentChunk

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract interface for RAG vector stores."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks and corresponding embeddings into vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search vector database by embedding similarity."""
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all indexed chunks associated with a specific document ID."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist vector index and metadata to disk."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load vector index and metadata from disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""
        pass


class NumpyFlatVectorStore(BaseVectorStore):
    """
    Lightweight, deterministic, persistent vector store using numpy arrays and JSON metadata.
    Zero external service dependency, ideal for embedded industrial edge devices.
    """

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self.persist_dir = Path(self.config.persist_directory)
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None  # Shape: (N, D)
        self.load()

    @property
    def vectors_path(self) -> Path:
        return self.persist_dir / f"{self.config.collection_name}_vectors.npy"

    @property
    def chunks_path(self) -> Path:
        return self.persist_dir / f"{self.config.collection_name}_chunks.json"

    def count(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add new chunks and embeddings. Handles duplicate chunk replacement seamlessly."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) != number of embeddings ({len(embeddings)})")

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Remove existing instances of the same chunk IDs if reindexing
        new_chunk_ids = {c.chunk_id for c in chunks}
        if self.chunks:
            keep_indices = [i for i, c in enumerate(self.chunks) if c.chunk_id not in new_chunk_ids]
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.embeddings is not None and len(keep_indices) > 0:
                self.embeddings = self.embeddings[keep_indices]
            else:
                self.embeddings = None

        # Append new entries
        self.chunks.extend(chunks)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        logger.info(f"Added {len(chunks)} chunks to vector store. Total active chunks: {len(self.chunks)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Perform vector similarity search with metadata filtering and minimum score thresholding.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        # Normalize query vector if cosine metric
        if self.config.distance_metric == "cosine":
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 1e-8:
                query_vec = query_vec / q_norm

            # Matrix cosine similarity
            scores = np.dot(self.embeddings, query_vec.T).squeeze(1)  # Shape (N,)
        elif self.config.distance_metric == "l2":
            dists = np.linalg.norm(self.embeddings - query_vec, axis=1)
            scores = 1.0 / (1.0 + dists)  # Convert distance to similarity score
        else:  # inner_product
            scores = np.dot(self.embeddings, query_vec.T).squeeze(1)

        # Apply metadata filters
        candidate_indices = list(range(len(self.chunks)))
        if filters:
            filtered_indices = []
            for idx in candidate_indices:
                chunk = self.chunks[idx]
                match = True
                for key, expected_val in filters.items():
                    actual_val = getattr(chunk, key, None)
                    if actual_val is None:
                        actual_val = chunk.metadata.get(key)

                    if actual_val is None or str(actual_val).lower() != str(expected_val).lower():
                        match = False
                        break
                if match:
                    filtered_indices.append(idx)
            candidate_indices = filtered_indices

        if not candidate_indices:
            return []

        # Sort candidate scores in descending order
        candidate_scores = [(idx, float(scores[idx])) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply similarity threshold
        results: list[tuple[DocumentChunk, float]] = []
        for idx, score in candidate_scores:
            if min_score is not None and score < min_score:
                continue
            results.append((self.chunks[idx], score))
            if len(results) >= top_k:
                break

        return results

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a specific document ID."""
        if not self.chunks:
            return 0

        keep_indices = [i for i, c in enumerate(self.chunks) if c.document_id != document_id]
        deleted_count = len(self.chunks) - len(keep_indices)

        if deleted_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.embeddings is not None and len(keep_indices) > 0:
                self.embeddings = self.embeddings[keep_indices]
            else:
                self.embeddings = None
            logger.info(f"Deleted {deleted_count} chunks for document '{document_id}'")

        return deleted_count

    def save(self) -> None:
        """Persist vector index to disk."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(self.vectors_path, self.embeddings)

        chunks_data = [c.to_dict() for c in self.chunks]
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        logger.info(f"Saved {len(self.chunks)} vectors and metadata chunks to '{self.persist_dir}'")

    def load(self) -> bool:
        """Load vector index from disk if present."""
        if not self.vectors_path.exists() or not self.chunks_path.exists():
            return False

        try:
            self.embeddings = np.load(self.vectors_path)
            with open(self.chunks_path, encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [DocumentChunk.from_dict(d) for d in chunks_data]
            logger.info(f"Loaded {len(self.chunks)} chunks and vectors from '{self.persist_dir}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from '{self.persist_dir}': {e}")
            self.chunks = []
            self.embeddings = None
            return False
