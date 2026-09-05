"""
RAG module main exports.
"""

from src.rag.chunking import TechnicalDocumentChunker
from src.rag.config import (
    ChunkingConfig,
    DocumentIngestionConfig,
    EmbeddingConfig,
    RAGConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from src.rag.embeddings import (
    BaseEmbeddingModel,
    DeterministicDenseEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    create_embedding_model,
)
from src.rag.evaluation import EvaluationMetrics, EvaluationSample, RAGEvaluator
from src.rag.ingestion import DocumentIngestionPipeline, DocumentParser, compute_file_hash
from src.rag.retrieval import (
    KnowledgeBaseService,
    TechnicalRetriever,
    preprocess_query,
    tokenize_text,
)
from src.rag.schema import (
    DocumentChunk,
    DocumentMetadata,
    RawDocumentPage,
    RetrievedEvidence,
    StructuredEvidenceContext,
)
from src.rag.vectorstore import BaseVectorStore, NumpyFlatVectorStore

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
