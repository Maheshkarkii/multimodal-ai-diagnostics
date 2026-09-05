"""
Agent Tool Interfaces.
Allows the reasoning agent to interact safely with RAG retrieval, sensor state checkers, and ISO standards.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.agent.core.schema import DiagnosticEvidenceItem, EvidenceType, SensorMeasurement
from src.rag.retrieval.retriever import TechnicalRetriever

logger = logging.getLogger(__name__)


class BaseAgentTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass


class TechnicalKnowledgeRetrievalTool(BaseAgentTool):
    """Retrieves relevant manual excerpts from the Phase 6 persistent RAG vector store."""

    def __init__(self, retriever: TechnicalRetriever):
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "retrieve_technical_evidence"

    @property
    def description(self) -> str:
        return "Query technical manuals, SOPs, and repair guides for troubleshooting steps, tolerances, and procedures."

    def execute(
        self,
        query: str,
        equipment_type: str | None = None,
        top_k: int = 3,
        threshold: float = 0.15,
    ) -> list[DiagnosticEvidenceItem]:
        filters = {"equipment_type": equipment_type} if equipment_type else None
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            similarity_threshold=threshold,
        )

        evidence_items: list[DiagnosticEvidenceItem] = []
        for r in results:
            citation = f"{r.document_name} (Page {r.page_number}{', Section: ' + r.section if r.section else ''})"
            evidence_items.append(
                DiagnosticEvidenceItem(
                    evidence_id=r.chunk_id,
                    evidence_type=EvidenceType.RETRIEVED_KNOWLEDGE,
                    source=citation,
                    statement=r.text.strip(),
                    provenance_detail={
                        "document_id": r.document_id,
                        "document_name": r.document_name,
                        "page_number": r.page_number,
                        "section": r.section,
                        "score": r.score,
                    },
                    relevance_score=r.score,
                )
            )

        return evidence_items


class SensorStateInspectionTool(BaseAgentTool):
    """Evaluates raw physical sensor telemetry against engineering operating thresholds."""

    @property
    def name(self) -> str:
        return "inspect_sensor_state"

    @property
    def description(self) -> str:
        return "Inspect and evaluate numerical sensor measurements against normal, warning, and critical thresholds."

    def execute(self, measurements: list[dict[str, Any]]) -> list[SensorMeasurement]:
        analyzed: list[SensorMeasurement] = []
        for m in measurements:
            param = m.get("parameter", "unknown")
            val = float(m.get("value", 0.0))
            unit = m.get("unit", "")
            norm_max = m.get("normal_max")
            warn_th = m.get("warning_threshold")
            crit_th = m.get("critical_threshold")

            status = "NORMAL"
            is_anomaly = False

            if crit_th is not None and val >= crit_th:
                status = "CRITICAL"
                is_anomaly = True
            elif warn_th is not None and val >= warn_th:
                status = "WARNING"
                is_anomaly = True
            elif norm_max is not None and val > norm_max:
                status = "WARNING"
                is_anomaly = True

            analyzed.append(
                SensorMeasurement(
                    parameter=param,
                    value=val,
                    unit=unit,
                    normal_min=m.get("normal_min"),
                    normal_max=norm_max,
                    warning_threshold=warn_th,
                    critical_threshold=crit_th,
                    is_anomaly=is_anomaly,
                    status=status,
                )
            )
        return analyzed


class ISOVibrationStandardTool(BaseAgentTool):
    """Evaluates vibration velocity against ISO 10816-3 industrial severity standards."""

    @property
    def name(self) -> str:
        return "check_iso_vibration_limits"

    @property
    def description(self) -> str:
        return "Map vibration RMS velocity (mm/s) to ISO 10816-3 severity zones (A: Good, B: Acceptable, C: Warning, D: Danger)."

    def execute(self, rms_velocity_mms: float) -> dict[str, Any]:
        v = float(rms_velocity_mms)
        if v < 2.3:
            zone = "Zone A (Good)"
            action = "Normal operation. No maintenance required."
            severity = "LOW"
        elif v <= 4.5:
            zone = "Zone B (Acceptable)"
            action = "Continuous unrestricted operation permitted."
            severity = "LOW"
        elif v <= 7.1:
            zone = "Zone C (Warning)"
            action = "Restricted operation. Schedule maintenance inspection soon."
            severity = "HIGH"
        else:
            zone = "Zone D (Critical Danger)"
            action = "Immediate emergency shutdown to prevent catastrophic failure."
            severity = "CRITICAL"

        return {
            "rms_velocity_mms": v,
            "iso_zone": zone,
            "recommended_action": action,
            "severity": severity,
            "standard_reference": "ISO 10816-3 (Mechanical vibration evaluation on non-rotating parts)",
        }
