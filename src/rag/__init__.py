"""
RAG module main exports.
"""

from src.rag.config import (
    RAGConfig,
    DocumentIngestionConfig,
    ChunkingConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
)
from src.rag.schema import (
    DocumentMetadata,
    RawDocumentPage,
    DocumentChunk,
    RetrievedEvidence,
    StructuredEvidenceContext,
)
from src.rag.ingestion import DocumentParser, DocumentIngestionPipeline, compute_file_hash
from src.rag.chunking import TechnicalDocumentChunker
from src.rag.embeddings import (
    BaseEmbeddingModel,
    DeterministicDenseEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    create_embedding_model,
)
from src.rag.vectorstore import BaseVectorStore, NumpyFlatVectorStore
from src.rag.retrieval import (
    TechnicalRetriever,
    KnowledgeBaseService,
    preprocess_query,
    tokenize_text,
)
from src.rag.evaluation import EvaluationSample, EvaluationMetrics, RAGEvaluator

__all__ = [
    "RAGConfig",
    "DocumentIngestionConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "RetrievalConfig",
    "DocumentMetadata",
    "RawDocumentPage",
    "DocumentChunk",
    "RetrievedEvidence",
    "StructuredEvidenceContext",
    "DocumentParser",
    "DocumentIngestionPipeline",
    "compute_file_hash",
    "TechnicalDocumentChunker",
    "BaseEmbeddingModel",
    "DeterministicDenseEmbeddingModel",
    "SentenceTransformerEmbeddingModel",
    "create_embedding_model",
    "BaseVectorStore",
    "NumpyFlatVectorStore",
    "TechnicalRetriever",
    "KnowledgeBaseService",
    "preprocess_query",
    "tokenize_text",
    "EvaluationSample",
    "EvaluationMetrics",
    "RAGEvaluator",
]
