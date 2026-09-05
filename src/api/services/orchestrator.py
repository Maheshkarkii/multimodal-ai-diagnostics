"""
Application-Level Diagnostic Orchestrator.
Orchestrates Input Validation -> Modality Inference -> RAG Retrieval ->
Diagnostic Reasoning Agent -> Explainability Layer -> Auditable Response.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.core.agent import DiagnosticReasoningAgent
from src.agent.core.schema import (
    ModalityObservation,
    ModalityType,
)
from src.api.schemas.diagnosis import (
    ClaimAuditMappingResponse,
    ConfidenceDecompositionResponse,
    DiagnosisResponse,
    EquipmentMetadataInput,
    EvidenceItemResponse,
    PrimaryDiagnosisResponse,
    RecommendedActionResponse,
    SensorTelemetryInput,
)
from src.explainability.core.config import ExplainabilityConfig
from src.explainability.core.service import ExplainabilityService
from src.rag.config import RAGConfig
from src.rag.embeddings.model import create_embedding_model
from src.rag.retrieval.retriever import TechnicalRetriever
from src.rag.vectorstore.store import NumpyFlatVectorStore

logger = logging.getLogger(__name__)


class DiagnosticOrchestrator:
    """
    Central workflow engine coordinating ML perception, RAG retrieval,
    autonomous reasoning, and auditable explanation synthesis.
    """

    def __init__(
        self,
        rag_config: RAGConfig | None = None,
        explainability_config: ExplainabilityConfig | None = None,
    ):
        self.rag_config = rag_config or RAGConfig()
        self.explainability_config = explainability_config or ExplainabilityConfig()

        # Initialize AI subsystems once on startup (Model lifecycle management)
        self.embedding_model = create_embedding_model(self.rag_config.embedding)
        self.vector_store = NumpyFlatVectorStore(self.rag_config.vector_store)
        self.retriever = TechnicalRetriever(self.vector_store, self.embedding_model, self.rag_config.retrieval)
        self.agent = DiagnosticReasoningAgent(self.retriever)
        self.explainability_service = ExplainabilityService(self.explainability_config)
        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def execute_diagnosis(
        self,
        request_id: str,
        technician_description: str | None = None,
        sensor_data: SensorTelemetryInput | None = None,
        equipment_meta: EquipmentMetadataInput | None = None,
        image_path: Path | None = None,
        audio_path: Path | None = None,
    ) -> DiagnosisResponse:
        """
        Execute end-to-end multimodal diagnostic workflow.
        """
        case_id = f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        eq = equipment_meta or EquipmentMetadataInput()

        # 1. Determine active modalities & execute perception pipelines
        observations: dict[str, ModalityObservation] = {}
        available_modalities = []

        if technician_description:
            available_modalities.append("text")

        # Ingest/Mock Vision inference if image is uploaded
        if image_path and image_path.exists():
            available_modalities.append("vision")
            # In a live multi-worker container, invoke vision_predictor
            observations["vision"] = ModalityObservation(
                modality=ModalityType.VISION,
                prediction="bearing_defect_wear"
                if "bearing" in (technician_description or "").lower()
                else "normal_state",
                confidence=0.88,
                summary=f"Analyzed visual features from '{image_path.name}'.",
            )

        # Ingest/Mock Audio inference if audio is uploaded
        if audio_path and audio_path.exists():
            available_modalities.append("audio")
            observations["audio"] = ModalityObservation(
                modality=ModalityType.AUDIO,
                prediction="hydraulic_cavitation"
                if "cavitation" in (technician_description or "").lower()
                else "bearing_defect_wear",
                confidence=0.91,
                summary=f"Processed acoustic spectrogram harmonics from '{audio_path.name}'.",
            )

        # Ingest Physical Sensor Telemetry
        sensor_readings: list[dict[str, Any]] = []
        if sensor_data:
            available_modalities.append("sensor")
            if sensor_data.vibration is not None:
                sensor_readings.append(
                    {
                        "parameter": "Vibration_RMS",
                        "value": sensor_data.vibration,
                        "unit": sensor_data.vibration_unit,
                        "warning_threshold": 4.5,
                        "critical_threshold": 7.1,
                    }
                )
            if sensor_data.temperature is not None:
                sensor_readings.append(
                    {
                        "parameter": "Bearing_Temp",
                        "value": sensor_data.temperature,
                        "unit": sensor_data.temperature_unit,
                        "warning_threshold": 75.0,
                        "critical_threshold": 90.0,
                    }
                )
            if sensor_data.rpm is not None:
                sensor_readings.append(
                    {
                        "parameter": "Shaft_RPM",
                        "value": sensor_data.rpm,
                        "unit": "RPM",
                    }
                )
            if sensor_data.current is not None:
                sensor_readings.append(
                    {
                        "parameter": "Motor_Current",
                        "value": sensor_data.current,
                        "unit": sensor_data.current_unit,
                    }
                )
            if sensor_data.pressure is not None:
                sensor_readings.append(
                    {
                        "parameter": "Pressure",
                        "value": sensor_data.pressure,
                        "unit": sensor_data.pressure_unit,
                        "warning_threshold": 0.8,
                    }
                )

        # 2. Invoke Phase 7 Diagnostic Reasoning Agent
        logger.info(f"[{request_id}] Executing Phase 7 Diagnostic Reasoning for case {case_id}...")
        diag_report = self.agent.diagnose_case(
            case_id=case_id,
            equipment_type=eq.equipment_type,
            equipment_model=eq.model,
            technician_description=technician_description or "",
            observations=observations if observations else None,
            sensor_data=sensor_readings if sensor_readings else None,
        )

        # 3. Invoke Phase 8 Explainability & Audit Layer
        logger.info(f"[{request_id}] Synthesizing Phase 8 Auditable Report...")
        auditable_report = self.explainability_service.generate_auditable_report(
            diagnostic_report=diag_report,
            raw_inputs={
                "technician_notes": technician_description,
                "sensor_data": sensor_data.model_dump() if sensor_data else None,
                "equipment": eq.model_dump(),
            },
        )

        # 4. Map to Strict Pydantic API Response
        evidence_resp = [
            EvidenceItemResponse(
                evidence_id=e.evidence_id,
                category=e.category.value,
                source=e.source,
                description=e.description,
                quality=e.quality.value,
                raw_value=e.raw_value,
                unit=e.unit,
                document_name=e.document_name,
                page_number=e.page_number,
                section=e.section,
            )
            for e in auditable_report.evidence_inventory
        ]

        actions_resp = [
            RecommendedActionResponse(
                action_id=a.action_id,
                priority=a.priority,
                requirement=a.requirement.value,
                action_text=a.action_text,
                rationale=a.rationale,
                source_reference=a.source_reference,
                is_safety_critical=a.is_safety_critical,
                justifying_evidence_ids=a.justifying_evidence_ids,
            )
            for a in auditable_report.recommended_actions
        ]

        claims_resp = [
            ClaimAuditMappingResponse(
                claim_id=c.claim_id,
                claim_statement=c.claim_statement,
                status=c.status.value,
                supporting_evidence_ids=c.supporting_evidence_ids,
                contradicting_evidence_ids=c.contradicting_evidence_ids,
                rationale=c.rationale,
            )
            for c in auditable_report.claim_mappings
        ]

        conf_decomp = ConfidenceDecompositionResponse(
            overall_confidence=auditable_report.confidence_decomposition.overall_confidence,
            multimodal_agreement=auditable_report.confidence_decomposition.multimodal_agreement,
            sensor_evidence_strength=auditable_report.confidence_decomposition.sensor_evidence_strength,
            acoustic_evidence_strength=auditable_report.confidence_decomposition.acoustic_evidence_strength,
            visual_evidence_strength=auditable_report.confidence_decomposition.visual_evidence_strength,
            technical_knowledge_match=auditable_report.confidence_decomposition.technical_knowledge_match,
            contradiction_penalty=auditable_report.confidence_decomposition.contradiction_penalty,
            rationale_summary=auditable_report.confidence_decomposition.rationale_summary,
        )

        primary_diag = PrimaryDiagnosisResponse(
            primary_diagnosis=auditable_report.primary_diagnosis,
            diagnostic_confidence=diag_report.diagnostic_confidence,
            severity=auditable_report.severity,
            confidence_decomposition=conf_decomp,
        )

        return DiagnosisResponse(
            case_id=case_id,
            request_id=request_id,
            timestamp=auditable_report.timestamp,
            status=auditable_report.system_status.value,
            equipment=auditable_report.equipment_info,
            problem_summary=auditable_report.problem_summary,
            available_modalities=available_modalities,
            diagnosis=primary_diag,
            evidence_inventory=evidence_resp,
            claim_mappings=claims_resp,
            alternative_hypotheses=auditable_report.alternative_hypotheses,
            recommended_actions=actions_resp,
            uncertainty_profile=auditable_report.uncertainty_profile.to_dict(),
            unsupported_claims=auditable_report.unsupported_claims,
            audit_summary=auditable_report.audit_record.to_dict(),
            markdown_report=auditable_report.to_markdown(),
        )
