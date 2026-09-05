"""
RAG Retrieval & Citation Grounding Evaluator.
Measures HitRate@k, MRR, nDCG, Citation Accuracy, and Unsupported Claim Rate.
"""

from typing import Any

import numpy as np

from src.evaluation.schemas import RAGRetrievalMetrics


def evaluate_rag_and_citations(
    retrieval_queries: list[dict[str, Any]],
    generated_claims: list[dict[str, Any]],
) -> RAGRetrievalMetrics:
    """
    Evaluates retrieval ranking accuracy and diagnostic claim citation validity.
    """
    if not retrieval_queries:
        return RAGRetrievalMetrics()

    hits_at_1 = []
    hits_at_5 = []
    reciprocal_ranks = []
    ndcgs = []

    for item in retrieval_queries:
        target_doc = item.get("target_doc_id", "")
        retrieved_ids = item.get("retrieved_doc_ids", [])

        # Hit @ 1
        h1 = 1.0 if retrieved_ids and retrieved_ids[0] == target_doc else 0.0
        hits_at_1.append(h1)

        # Hit @ 5
        h5 = 1.0 if target_doc in retrieved_ids[:5] else 0.0
        hits_at_5.append(h5)

        # Reciprocal Rank
        if target_doc in retrieved_ids:
            rank = retrieved_ids.index(target_doc) + 1
            reciprocal_ranks.append(1.0 / rank)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)

    # Citation Grounding Analysis
    valid_citations = 0
    total_citations = 0
    unsupported_claims = 0
    total_claims = len(generated_claims)

    for claim in generated_claims:
        is_supported = claim.get("is_supported", False)
        has_valid_citation = claim.get("has_valid_citation", False)

        if not is_supported:
            unsupported_claims += 1

        if "citation" in claim:
            total_citations += 1
            if has_valid_citation:
                valid_citations += 1

    citation_acc = (valid_citations / total_citations) if total_citations > 0 else 1.0
    unsupported_rate = (unsupported_claims / total_claims) if total_claims > 0 else 0.0

    return RAGRetrievalMetrics(
        hit_rate_at_1=round(float(np.mean(hits_at_1)), 4),
        hit_rate_at_5=round(float(np.mean(hits_at_5)), 4),
        mrr=round(float(np.mean(reciprocal_ranks)), 4),
        ndcg_at_5=round(float(np.mean(ndcgs)), 4),
        citation_accuracy=round(citation_acc, 4),
        unsupported_claim_rate=round(unsupported_rate, 4),
        evaluated_queries=len(retrieval_queries),
    )
