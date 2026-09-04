"""
Technical Evidence Retriever with Hybrid (Vector + BM25) Search,
Metadata Filtering, Similarity Thresholding, and Structured Context Building.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

from src.rag.config import RAGConfig, RetrievalConfig
from src.rag.embeddings.model import BaseEmbeddingModel, create_embedding_model
from src.rag.schema import DocumentChunk, RetrievedEvidence, StructuredEvidenceContext
from src.rag.vectorstore.store import BaseVectorStore, NumpyFlatVectorStore

logger = logging.getLogger(__name__)


def preprocess_query(query: str) -> str:
    """Normalize query text, remove redundant whitespaces, and clean special punctuation."""
    if not query:
        return ""
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 matching."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]


class TechnicalRetriever:
    """
    Evidence retrieval engine supporting pure semantic dense retrieval,
    sparse BM25 exact-keyword search, and weighted hybrid fusion.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        config: Optional[RetrievalConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()
        self.bm25_index: Optional[BM25Okapi] = None
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Initialize BM25 index over all stored document chunks."""
        if hasattr(self.vector_store, "chunks") and self.vector_store.chunks:
            corpus = [tokenize_text(c.text) for c in self.vector_store.chunks]
            if any(len(doc) > 0 for doc in corpus):
                self.bm25_index = BM25Okapi(corpus)
            else:
                self.bm25_index = None
        else:
            self.bm25_index = None

    def refresh_sparse_index(self) -> None:
        """Rebuild BM25 index after vector store updates."""
        self._build_bm25_index()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedEvidence]:
        """
        Retrieve relevant technical evidence chunks for a query.
        Returns empty list if no chunks meet the relevance threshold.
        """
        cleaned_query = preprocess_query(query)
        if not cleaned_query:
            return []

        k = top_k or self.config.top_k
        min_thresh = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold

        if self.config.enable_hybrid and self.bm25_index is not None and len(getattr(self.vector_store, "chunks", [])) > 1:
            evidence = self._hybrid_retrieve(cleaned_query, k, filters, min_thresh)
        else:
            evidence = self._dense_retrieve(cleaned_query, k, filters, min_thresh)

        return evidence

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        min_threshold: Optional[float],
    ) -> List[RetrievedEvidence]:
        """Perform dense embedding vector retrieval."""
        query_emb = self.embedding_model.embed_text(query)
        matches = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k,
            filters=filters,
            min_score=min_threshold,
        )

        results: List[RetrievedEvidence] = []
        for chunk, score in matches:
            results.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=float(score),
                    retrieval_mode="dense",
                    source_path=chunk.source_path,
                    equipment_type=chunk.equipment_type,
                    manufacturer=chunk.manufacturer,
                    model=chunk.model,
                    metadata=chunk.metadata,
                )
            )
        return results

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        min_threshold: Optional[float],
    ) -> List[RetrievedEvidence]:
        """
        Combine Dense Vector similarity and Sparse BM25 scores via Reciprocal / Weighted Score Fusion.
        """
        if not hasattr(self.vector_store, "chunks") or not self.vector_store.chunks:
            return []

        chunks: List[DocumentChunk] = self.vector_store.chunks
        query_tokens = tokenize_text(query)

        # 1. Compute BM25 scores (clamp non-negative)
        bm25_scores = [0.0] * len(chunks)
        if self.bm25_index and query_tokens:
            raw_scores = [max(0.0, float(s)) for s in self.bm25_index.get_scores(query_tokens)]
            max_bm25 = max(raw_scores) if raw_scores else 0.0
            if max_bm25 > 0:
                bm25_scores = [s / max_bm25 for s in raw_scores]

        # 2. Compute Dense Vector similarities across all candidates
        query_emb = self.embedding_model.embed_text(query)
        dense_matches = self.vector_store.search(query_embedding=query_emb, top_k=len(chunks), filters=None)
        dense_results = {chunk.chunk_id: score for chunk, score in dense_matches}

        # 3. Fuse scores with filter application
        fused_candidates: List[Tuple[DocumentChunk, float]] = []

        w_dense = self.config.dense_weight
        w_sparse = self.config.sparse_weight

        for idx, chunk in enumerate(chunks):
            # Check metadata filters
            if filters:
                match = True
                for key, expected_val in filters.items():
                    actual_val = getattr(chunk, key, None)
                    if actual_val is None:
                        actual_val = chunk.metadata.get(key)
                    if actual_val is None or str(actual_val).lower() != str(expected_val).lower():
                        match = False
                        break
                if not match:
                    continue

            d_score = dense_results.get(chunk.chunk_id, 0.0)
            s_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0

            # Fused score
            fused_score = (w_dense * d_score) + (w_sparse * s_score)

            if min_threshold is not None and fused_score < min_threshold:
                continue

            fused_candidates.append((chunk, fused_score))

        # Sort descending by fused score
        fused_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused_candidates[:top_k]

        results: List[RetrievedEvidence] = []
        for chunk, score in top_candidates:
            results.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=float(score),
                    retrieval_mode="hybrid",
                    source_path=chunk.source_path,
                    equipment_type=chunk.equipment_type,
                    manufacturer=chunk.manufacturer,
                    model=chunk.model,
                    metadata=chunk.metadata,
                )
            )

        return results

    def build_evidence_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> StructuredEvidenceContext:
        """
        Execute retrieval and assemble into a formatted StructuredEvidenceContext respecting character limits.
        """
        evidence_items = self.retrieve(query, top_k=top_k, filters=filters)
        max_chars = self.config.max_context_chars
        max_chunks = self.config.max_context_chunks

        selected_items: List[RetrievedEvidence] = []
        curr_chars = 0
        truncated = False

        for ev in evidence_items[:max_chunks]:
            ev_len = len(ev.text)
            if curr_chars + ev_len <= max_chars or not selected_items:
                selected_items.append(ev)
                curr_chars += ev_len
            else:
                truncated = True
                break

        return StructuredEvidenceContext(
            query=query,
            evidence_items=selected_items,
            total_chunks=len(evidence_items),
            total_characters=curr_chars,
            truncated=truncated,
        )


class KnowledgeBaseService:
    """
    Unified High-Level RAG Interface for Indexing and Querying.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embedding_model = create_embedding_model(self.config.embedding)
        self.vector_store = NumpyFlatVectorStore(self.config.vector_store)
        self.retriever = TechnicalRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            config=self.config.retrieval,
        )

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> StructuredEvidenceContext:
        """Query the knowledge base and return structured provenance-attached context."""
        return self.retriever.build_evidence_context(query=query_text, top_k=top_k, filters=filters)
