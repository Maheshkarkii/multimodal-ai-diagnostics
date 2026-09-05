"""
RAG Configuration dataclasses for Document Ingestion, Embedding, Storage, and Retrieval.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DocumentIngestionConfig:
    """Document parsing and loading configurations."""

    supported_extensions: list[str] = field(default_factory=lambda: [".pdf", ".txt", ".md", ".docx"])
    detect_scanned_pdf: bool = True
    min_page_chars_threshold: int = 40
    clean_whitespace: bool = True
    preserve_page_boundaries: bool = True


@dataclass
class ChunkingConfig:
    """Deliberate chunking configuration preserving structure and provenance."""

    strategy: str = "hierarchical"  # "hierarchical", "fixed_window", "sentence"
    chunk_size: int = 500  # Target characters per chunk
    chunk_overlap: int = 100  # Character overlap
    split_on_headings: bool = True
    split_on_paragraphs: bool = True
    split_on_tables: bool = True


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model_type: str = "sentence_transformer"  # "sentence_transformer", "deterministic", "tfidf"
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """Vector database storage configuration."""

    store_type: str = "numpy_flat"  # "numpy_flat", "chroma", "faiss"
    persist_directory: str = "data/rag/vector_store"
    collection_name: str = "equipment_manuals"
    distance_metric: str = "cosine"  # "cosine", "l2", "inner_product"


@dataclass
class RetrievalConfig:
    """Retrieval parameters and constraints."""

    top_k: int = 5
    similarity_threshold: float | None = 0.20  # Minimum similarity score required
    enable_hybrid: bool = True  # Combine dense embedding search with BM25 keyword search
    dense_weight: float = 0.65  # Weight for vector similarity in hybrid mode
    sparse_weight: float = 0.35  # Weight for BM25 score in hybrid mode
    max_context_chars: int = 3000  # Maximum character budget for constructed context
    max_context_chunks: int = 5


@dataclass
class RAGConfig:
    """Master RAG pipeline configuration."""

    system_name: str = "technical_knowledge_rag"
    documents_dir: str = "data/rag/documents"
    manifest_path: str = "data/rag/vector_store/documents_manifest.json"
    ingestion: DocumentIngestionConfig = field(default_factory=DocumentIngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RAGConfig":
        """Load RAG config from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RAGConfig":
        """Construct RAGConfig from nested dictionary."""
        return cls(
            system_name=d.get("system_name", "technical_knowledge_rag"),
            documents_dir=d.get("documents_dir", "data/rag/documents"),
            manifest_path=d.get("manifest_path", "data/rag/vector_store/documents_manifest.json"),
            ingestion=DocumentIngestionConfig(**d.get("ingestion", {})),
            chunking=ChunkingConfig(**d.get("chunking", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            vector_store=VectorStoreConfig(**d.get("vector_store", {})),
            retrieval=RetrievalConfig(**d.get("retrieval", {})),
        )

    def to_yaml(self, save_path: str | Path) -> None:
        """Serialize configuration to a YAML file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
