"""
Multi-Stage Diagnostic Reasoning Orchestration Engine.
Orchestrates observation gathering, RAG technical retrieval, hypothesis generation,
evidence matching, contradiction detection, and report generation.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from src.agent.core.config import AgentConfig
from src.agent.core.schema import (
    ContradictionRecord,
    DiagnosticEvidenceItem,
    DiagnosticReport,
    DiagnosticState,
    EvidenceType,
    ModalityObservation,
)
from src.agent.llm.provider import BaseLLMProvider, create_llm_provider
from src.agent.tools.tools import (
    ISOVibrationStandardTool,
    SensorStateInspectionTool,
    TechnicalKnowledgeRetrievalTool,
)
from src.agent.validation.groundedness import GroundednessChecker
from src.rag.retrieval.retriever import TechnicalRetriever

logger = logging.getLogger(__name__)


class DiagnosticReasoningAgent:
    """
    Autonomous reasoning layer synthesizing multimodal observations and technical manuals
    into grounded, citation-anchored equipment diagnoses.
    """

    def __init__(
        self,
        retriever: TechnicalRetriever,
        config: AgentConfig | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.config = config or AgentConfig()
        self.retriever = retriever
        self.llm = llm_provider or create_llm_provider(self.config.llm)

        # Tools
        self.rag_tool = TechnicalKnowledgeRetrievalTool(retriever)
        self.sensor_tool = SensorStateInspectionTool()
        self.iso_tool = ISOVibrationStandardTool()
        self.groundedness_checker = GroundednessChecker(self.config.reasoning.groundedness_threshold)

    def diagnose_case(
        self,
        case_id: str | None = None,
        equipment_type: str = "motor",
        equipment_model: str | None = None,
        technician_description: str = "",
        observations: dict[str, ModalityObservation] | None = None,
        sensor_data: list[dict[str, Any]] | None = None,
    ) -> DiagnosticReport:
        """
        Execute bounded multi-stage diagnostic reasoning workflow.
        """
        cid = case_id or f"CASE_{uuid.uuid4().hex[:8].upper()}"
        state = DiagnosticState(
            case_id=cid,
            equipment_type=equipment_type,
            equipment_model=equipment_model,
            technician_description=technician_description,
            available_modalities=list(observations.keys()) if observations else [],
            observations=observations or {},
        )
        state.add_trace("INITIALIZATION", f"Initialized case for equipment: {equipment_type}")

        # Stage 1: Ingest & Evaluate Sensor Measurements
        if sensor_data:
            measurements = self.sensor_tool.execute(sensor_data)
            state.sensor_measurements = measurements
            for m in measurements:
                state.add_trace("SENSOR_CHECK", f"Measured {m.parameter}: {m.value} {m.unit} (Status: {m.status})")
                if m.parameter.lower() == "vibration" or "vibration" in m.parameter.lower():
                    iso_eval = self.iso_tool.execute(m.value)
                    state.add_trace(
                        "ISO_CHECK", f"ISO 10816-3 Evaluation: {iso_eval['iso_zone']} ({iso_eval['severity']})"
                    )

        # Stage 2: Formulate Retrieval Questions & Query Technical RAG
        retrieval_queries = self._generate_retrieval_queries(state)
        all_rag_evidence: list[DiagnosticEvidenceItem] = []

        for q in retrieval_queries:
            items = self.rag_tool.execute(
                query=q,
                equipment_type=equipment_type,
                top_k=self.config.retrieval_tool.default_top_k,
                threshold=self.config.retrieval_tool.min_similarity_threshold,
            )
            state.add_trace("RAG_RETRIEVAL", f"Query '{q}' returned {len(items)} evidence chunks.")
            all_rag_evidence.extend(items)

        state.retrieved_evidence = all_rag_evidence

        # Stage 3: Detect Contradictions & Information Gaps
        contradictions, missing_info = self._detect_contradictions_and_gaps(state)
        state.contradictions = contradictions
        state.missing_information = missing_info

        # Stage 4: Construct Prompt Context & Invoke LLM Reasoning
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_diagnostic_context(state)

        state.add_trace("LLM_REASONING", "Invoking structured LLM reasoning engine.")
        llm_out = self.llm.generate_structured_diagnosis(system_prompt, user_prompt)

        # Stage 5: Groundedness Validation
        all_evidence_pool = self._assemble_all_evidence(state)
        groundedness_score, unsupported_claims = self.groundedness_checker.evaluate_groundedness(
            primary_diagnosis=llm_out.primary_diagnosis,
            supporting_statements=llm_out.supporting_evidence_statements,
            recommended_actions=llm_out.recommended_actions,
            cited_references=llm_out.cited_technical_references,
            available_evidence=all_evidence_pool,
        )

        state.add_trace(
            "GROUNDEDNESS_VALIDATION",
            f"Evidence Groundedness Score: {groundedness_score * 100:.1f}% ({len(unsupported_claims)} warnings)",
        )

        # Stage 6: Build Final Report
        report = DiagnosticReport(
            case_id=state.case_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            equipment={"type": state.equipment_type, "model": state.equipment_model},
            problem_summary=state.technician_description or "No problem description provided.",
            available_modalities=state.available_modalities,
            primary_diagnosis=llm_out.primary_diagnosis,
            diagnostic_confidence=llm_out.diagnostic_confidence,
            severity=llm_out.severity,
            alternative_hypotheses=llm_out.alternative_hypotheses,
            supporting_evidence=[
                {"evidence_type": "SUPPORTING", "statement": s, "source": "Diagnostic Evidence"}
                for s in llm_out.supporting_evidence_statements
            ],
            contradicting_evidence=[
                {"evidence_type": "CONTRADICTION", "statement": c, "source": "Cross-Modality Check"}
                for c in llm_out.contradicting_evidence_statements
            ],
            contradictions_detected=[
                {
                    "conflict_description": c.conflict_description,
                    "source_a": c.source_a,
                    "source_b": c.source_b,
                }
                for c in state.contradictions
            ],
            missing_information=state.missing_information + llm_out.missing_information,
            recommended_actions=llm_out.recommended_actions,
            technical_references=llm_out.cited_technical_references,
            groundedness_score=groundedness_score,
            unsupported_claims=unsupported_claims,
            status="completed",
        )

        return report

    def _generate_retrieval_queries(self, state: DiagnosticState) -> list[str]:
        """Generate targeted retrieval questions based on symptoms and observations."""
        queries = []
        eq = state.equipment_type

        # Query from technician description
        if state.technician_description:
            queries.append(f"{eq} {state.technician_description}")

        # Query from model predictions
        for _mod, obs in state.observations.items():
            if (
                "defect" in obs.prediction
                or "fault" in obs.prediction
                or "cavitation" in obs.prediction
                or "unbalance" in obs.prediction
            ):
                queries.append(f"{eq} {obs.prediction} inspection steps and symptoms")

        # Query from sensor anomalies
        for sm in state.sensor_measurements:
            if sm.is_anomaly:
                queries.append(f"{eq} abnormal {sm.parameter} threshold and corrective action")

        # Fallback query if no signals
        if not queries:
            queries.append(f"{eq} standard maintenance operating procedure and limits")

        # Deduplicate
        return list(dict.fromkeys(queries))[:3]

    def _detect_contradictions_and_gaps(self, state: DiagnosticState) -> (list[ContradictionRecord], list[str]):
        """Detect conflicting signals across modality predictions and sensor states."""
        contradictions: list[ContradictionRecord] = []
        missing_info: list[str] = []

        # Cross-modality contradiction detection
        preds = {k: v.prediction for k, v in state.observations.items()}
        if "vision" in preds and "audio" in preds:
            if "normal" in preds["vision"] and "defect" in preds["audio"]:
                contradictions.append(
                    ContradictionRecord(
                        contradiction_id="CONTRA_VIS_AUD_01",
                        source_a="Vision Model",
                        statement_a=preds["vision"],
                        source_b="Acoustic Audio Model",
                        statement_b=preds["audio"],
                        conflict_description="Visual inspection shows normal surface while acoustic channel predicts internal defect.",
                    )
                )

        # Missing information checks
        if "audio" not in state.available_modalities:
            missing_info.append("Acoustic audio recording missing (unable to verify harmonic signatures).")
        if "vision" not in state.available_modalities:
            missing_info.append("Equipment image missing (unable to inspect surface cracks or seal leaks).")
        if not state.sensor_measurements:
            missing_info.append("Telemetry sensors missing (real-time vibration and temperature unknown).")

        return contradictions, missing_info

    def _assemble_all_evidence(self, state: DiagnosticState) -> list[DiagnosticEvidenceItem]:
        """Combine all observed measurements, model predictions, and retrieved manual chunks."""
        pool: list[DiagnosticEvidenceItem] = []

        # Model inferences
        for mod, obs in state.observations.items():
            pool.append(
                DiagnosticEvidenceItem(
                    evidence_id=f"MOD_{mod}",
                    evidence_type=EvidenceType.MODEL_INFERENCE,
                    source=f"{mod.capitalize()} Model",
                    statement=f"Predicted {obs.prediction} with confidence {obs.confidence:.2f}",
                    relevance_score=obs.confidence,
                )
            )

        # Sensor readings
        for sm in state.sensor_measurements:
            pool.append(
                DiagnosticEvidenceItem(
                    evidence_id=f"SENS_{sm.parameter}",
                    evidence_type=EvidenceType.OBSERVED_MEASUREMENT,
                    source=f"Telemetry Sensor ({sm.parameter})",
                    statement=f"{sm.parameter}: {sm.value} {sm.unit} (Status: {sm.status})",
                    relevance_score=1.0,
                )
            )

        # RAG evidence
        pool.extend(state.retrieved_evidence)
        return pool

    def _build_system_prompt(self) -> str:
        return (
            "You are the AI Field Engineer Diagnostic Reasoning Engine. "
            "Your role is to analyze multi-channel observations, sensor telemetry, and technical manuals. "
            "Formulate a structured root-cause diagnosis, cite exact source provenance, evaluate alternative hypotheses, "
            "identify evidence contradictions, and prescribe safety-compliant next actions. "
            "Never fabricate page citations or unobserved measurements."
        )

    def _build_user_diagnostic_context(self, state: DiagnosticState) -> str:
        lines = [
            f"DIAGNOSTIC CASE: {state.case_id}",
            f"EQUIPMENT: {state.equipment_type} (Model: {state.equipment_model or 'Unspecified'})",
            f"TECHNICIAN NOTES: {state.technician_description or 'None'}",
            "",
            "=== OBSERVED MODALITY PREDICTIONS ===",
        ]

        for mod, obs in state.observations.items():
            lines.append(f"- {mod.upper()}: {obs.prediction} (Confidence: {obs.confidence:.2f})")

        if state.sensor_measurements:
            lines.append("\n=== TELEMETRY SENSOR MEASUREMENTS ===")
            for sm in state.sensor_measurements:
                lines.append(f"- {sm.parameter}: {sm.value} {sm.unit} [Status: {sm.status}]")

        if state.retrieved_evidence:
            lines.append("\n=== RETRIEVED TECHNICAL MANUAL EVIDENCE ===")
            for ev in state.retrieved_evidence:
                lines.append(f"[{ev.source}]\n{ev.statement}\n")

        return "\n".join(lines)
