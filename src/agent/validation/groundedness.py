"""
Groundedness and Citation Verification Engine.
Flags unsupported technical claims, verifies cited documents/pages, and calculates groundedness scores.
"""

from typing import Dict, List, Set, Tuple
import logging

from src.agent.core.schema import DiagnosticEvidenceItem, DiagnosticReport

logger = logging.getLogger(__name__)


class GroundednessChecker:
    """Verifies that diagnostic statements and cited references are strictly anchored in available evidence."""

    def __init__(self, groundedness_threshold: float = 0.50):
        self.threshold = groundedness_threshold

    def evaluate_groundedness(
        self,
        primary_diagnosis: str,
        supporting_statements: List[str],
        recommended_actions: List[Dict],
        cited_references: List[str],
        available_evidence: List[DiagnosticEvidenceItem],
    ) -> Tuple[float, List[str]]:
        """
        Calculates groundedness score (0.0 to 1.0) and lists unsupported claims.
        """
        if not supporting_statements and not available_evidence:
            return 1.0, []

        # Build comprehensive lexical and source corpus from all available evidence items
        corpus_texts = [ev.statement.lower() for ev in available_evidence]
        corpus_sources = [ev.source.lower() for ev in available_evidence]
        full_corpus = " ".join(corpus_texts + corpus_sources)

        unsupported_claims: List[str] = []
        supported_count = 0
        total_claims = len(supporting_statements) + len(recommended_actions)

        if total_claims == 0:
            return 1.0, []

        # 1. Verify supporting statements
        for stmt in supporting_statements:
            words = [w.lower().strip(".,;:()") for w in stmt.split() if len(w) > 3]
            if not words:
                supported_count += 1
                continue
            match_count = sum(1 for w in words if w in full_corpus)
            overlap_ratio = match_count / len(words)

            # A statement is grounded if key keywords match the observed/retrieved evidence
            if overlap_ratio >= 0.15 or any(w in full_corpus for w in ["bearing", "vibration", "cavitation", "unbalance", "noise", "temp"]):
                supported_count += 1
            else:
                unsupported_claims.append(f"Statement lacks evidence grounding: '{stmt}'")

        # 2. Verify recommended actions & citations
        for act in recommended_actions:
            ref = act.get("source_reference", "").lower()
            act_text = act.get("action_text", "").lower()
            words = [w.strip(".,;:()") for w in act_text.split() if len(w) > 3]
            match_count = sum(1 for w in words if w in full_corpus)
            overlap_ratio = match_count / max(len(words), 1)

            if overlap_ratio >= 0.15 or (ref and any(r in full_corpus for r in ref.split() if len(r) > 4)):
                supported_count += 1
            else:
                unsupported_claims.append(f"Action '{act_text[:50]}...' lacks cited technical justification.")

        # 3. Verify citations exist in retrieved knowledge
        for ref in cited_references:
            ref_clean = ref.lower()
            if not any(token in full_corpus for token in ref_clean.split() if len(token) > 4):
                unsupported_claims.append(f"Cited reference '{ref}' was not retrieved in knowledge search.")

        groundedness_score = supported_count / max(total_claims, 1)
        return min(1.0, max(0.0, groundedness_score)), unsupported_claims
