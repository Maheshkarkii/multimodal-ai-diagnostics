"""
Deterministic CI/CD Regression Gate Engine.
Compares new candidate evaluation results against golden baseline metrics.
"""

from typing import Any, Dict, Tuple
from src.evaluation.schemas import EvaluationSummary


DEFAULT_REGRESSION_THRESHOLDS = {
    "max_accuracy_drop": 0.03,
    "max_macro_f1_drop": 0.03,
    "max_unsupported_claim_increase": 0.02,
    "max_latency_increase_percent": 25.0,
}


def check_regression_against_baseline(
    candidate: EvaluationSummary,
    baseline: EvaluationSummary,
    thresholds: Dict[str, float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates whether candidate model passes non-regression criteria.
    """
    t = thresholds or DEFAULT_REGRESSION_THRESHOLDS
    details: Dict[str, Any] = {}
    passed = True

    # 1. Vision & Multimodal F1 Checks
    if candidate.fusion_metrics and baseline.fusion_metrics:
        f1_delta = baseline.fusion_metrics.f1_macro - candidate.fusion_metrics.f1_macro
        details["fusion_f1_delta"] = round(f1_delta, 4)
        if f1_delta > t["max_macro_f1_drop"]:
            passed = False
            details["fusion_f1_regressed"] = True

    # 2. RAG Unsupported Claim Rate Check
    if candidate.rag_metrics and baseline.rag_metrics:
        claim_increase = candidate.rag_metrics.unsupported_claim_rate - baseline.rag_metrics.unsupported_claim_rate
        details["unsupported_claim_increase"] = round(claim_increase, 4)
        if claim_increase > t["max_unsupported_claim_increase"]:
            passed = False
            details["hallucination_regressed"] = True

    # 3. Latency Check
    if candidate.latency_profile and baseline.latency_profile and baseline.latency_profile.total_pipeline_ms > 0:
        base_lat = baseline.latency_profile.total_pipeline_ms
        cand_lat = candidate.latency_profile.total_pipeline_ms
        pct_increase = ((cand_lat - base_lat) / base_lat) * 100.0
        details["latency_increase_pct"] = round(pct_increase, 2)
        if pct_increase > t["max_latency_increase_percent"]:
            passed = False
            details["latency_regressed"] = True

    details["gate_passed"] = passed
    return passed, details
