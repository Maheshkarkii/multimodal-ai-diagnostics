"""
RAG Evaluation Framework and Metrics.
Computes Recall@k, Precision@k, Mean Reciprocal Rank (MRR), and HitRate@k.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rag.retrieval.retriever import TechnicalRetriever
from src.rag.schema import RetrievedEvidence

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single annotated query with ground-truth target citations or keywords."""
    query_id: str
    query: str
    target_document_name: str
    target_page: Optional[int] = None
    target_section: Optional[str] = None
    target_keywords: List[str] = field(default_factory=list)
    equipment_type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Quantitative metrics computed over evaluation dataset."""
    total_queries: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float  # Mean Reciprocal Rank
    precision_at_k: float
    recall_at_k: float
    average_top1_score: float
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)


class RAGEvaluator:
    """Evaluates technical retrieval quality against curated industrial benchmarks."""

    def __init__(self, retriever: TechnicalRetriever):
        self.retriever = retriever

    def evaluate_benchmark(
        self,
        samples: List[EvaluationSample],
        top_k: int = 5,
    ) -> EvaluationMetrics:
        """Run evaluation benchmark across all ground-truth queries."""
        if not samples:
            return EvaluationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks = []
        precisions = []
        recalls = []
        top1_scores = []
        details = []

        for s in samples:
            results = self.retriever.retrieve(s.query, top_k=top_k)

            # Check matching rank
            match_rank = None
            relevant_retrieved = 0

            for rank_idx, item in enumerate(results, start=1):
                is_doc_match = s.target_document_name.lower() in item.document_name.lower()
                is_page_match = s.target_page is None or item.page_number == s.target_page
                is_sec_match = (
                    s.target_section is None
                    or (item.section and s.target_section.lower() in item.section.lower())
                )

                if is_doc_match and (is_page_match or is_sec_match):
                    if match_rank is None:
                        match_rank = rank_idx
                    relevant_retrieved += 1

            # Hit rates
            if match_rank == 1:
                hits_at_1 += 1
            if match_rank is not None and match_rank <= 3:
                hits_at_3 += 1
            if match_rank is not None and match_rank <= 5:
                hits_at_5 += 1

            # Reciprocal rank
            rr = 1.0 / match_rank if match_rank is not None else 0.0
            reciprocal_ranks.append(rr)

            # Precision and Recall
            prec = relevant_retrieved / max(len(results), 1)
            rec = 1.0 if relevant_retrieved > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

            top_score = results[0].score if results else 0.0
            top1_scores.append(top_score)

            details.append({
                "query_id": s.query_id,
                "query": s.query,
                "target_doc": s.target_document_name,
                "target_page": s.target_page,
                "target_section": s.target_section,
                "matched_rank": match_rank,
                "reciprocal_rank": rr,
                "retrieved_count": len(results),
                "top_score": top_score,
                "top_result_doc": results[0].document_name if results else None,
                "top_result_page": results[0].page_number if results else None,
            })

        n = len(samples)
        metrics = EvaluationMetrics(
            total_queries=n,
            hit_rate_at_1=hits_at_1 / n,
            hit_rate_at_3=hits_at_3 / n,
            hit_rate_at_5=hits_at_5 / n,
            mrr=sum(reciprocal_ranks) / n,
            precision_at_k=sum(precisions) / n,
            recall_at_k=sum(recalls) / n,
            average_top1_score=sum(top1_scores) / n,
            per_query_results=details,
        )

        return metrics
