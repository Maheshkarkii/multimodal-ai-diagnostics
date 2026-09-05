"""
Structured Data Models and Schemas for Phase 7 Diagnostic Reasoning.
Defines explicit hierarchies for Observations, Hypotheses, Evidence, State, and Reports.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class ModalityType(str, Enum):
    VISION = "vision"
    AUDIO = "audio"
    SENSOR = "sensor"
    TEXT = "text"
    MULTIMODAL_FUSED = "multimodal_fused"


class EvidenceType(str, Enum):
    OBSERVED_MEASUREMENT = "observed_measurement"
    MODEL_INFERENCE = "model_inference"
    RETRIEVED_KNOWLEDGE = "retrieved_knowledge"
    TECHNICIAN_REPORT = "technician_report"


@dataclass
class ModalityObservation:
    """Standardized observation produced by a modality perception model or sensor."""

    modality: ModalityType
    prediction: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    anomaly_score: float | None = None
    extracted_features: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value
        return d


@dataclass
class SensorMeasurement:
    """Individual physical sensor reading with unit and operational thresholds."""

    parameter: str
    value: float
    unit: str
    normal_min: float | None = None
    normal_max: float | None = None
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    is_anomaly: bool = False
    status: str = "NORMAL"  # "NORMAL", "WARNING", "CRITICAL"


@dataclass
class DiagnosticEvidenceItem:
    """Typed evidence item with provenance attached."""

    evidence_id: str
    evidence_type: EvidenceType
    source: str  # e.g., "motor_m4500_maintenance_manual.pdf (Page 2)" or "Acoustic CNN"
    statement: str
    provenance_detail: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_type"] = self.evidence_type.value
        return d


@dataclass
class DiagnosticHypothesis:
    """A competing failure mode hypothesis evaluated against evidence."""

    hypothesis_id: str
    failure_mode: str
    description: str
    likelihood_score: float  # 0.0 to 1.0
    supporting_evidence: list[DiagnosticEvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[DiagnosticEvidenceItem] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    status: str = "active"  # "active", "rejected", "confirmed"


@dataclass
class ContradictionRecord:
    """Explicit record of conflicting evidence between modalities, sensors, or documentation."""

    contradiction_id: str
    source_a: str
    statement_a: str
    source_b: str
    statement_b: str
    conflict_description: str
    impact_on_confidence: float = 0.15


@dataclass
class RecommendedAction:
    """Actionable, safety-grounded next step for field engineers."""

    action_id: str
    priority: int  # 1 (immediate) to 5 (preventive)
    action_text: str
    rationale: str
    is_safety_critical: bool
    source_reference: str | None = None


@dataclass
class DiagnosticState:
    """Complete internal representation of a diagnostic investigation case."""

    case_id: str
    equipment_type: str
    equipment_model: str | None = None
    technician_description: str = ""
    available_modalities: list[str] = field(default_factory=list)
    observations: dict[str, ModalityObservation] = field(default_factory=dict)
    sensor_measurements: list[SensorMeasurement] = field(default_factory=list)
    retrieved_evidence: list[DiagnosticEvidenceItem] = field(default_factory=list)
    hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)
    contradictions: list[ContradictionRecord] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    selected_diagnosis: str | None = None
    diagnostic_confidence: float = 0.0
    severity: SeverityLevel = SeverityLevel.UNKNOWN
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    status: str = "initialized"

    def add_trace(self, step_name: str, message: str) -> None:
        self.reasoning_trace.append(f"[{step_name}] {message}")


@dataclass
class DiagnosticReport:
    """Final auditable diagnostic assessment report."""

    case_id: str
    timestamp: str
    equipment: dict[str, Any]
    problem_summary: str
    available_modalities: list[str]
    primary_diagnosis: str
    diagnostic_confidence: float
    severity: SeverityLevel
    alternative_hypotheses: list[dict[str, Any]]
    supporting_evidence: list[dict[str, Any]]
    contradicting_evidence: list[dict[str, Any]]
    contradictions_detected: list[dict[str, Any]]
    missing_information: list[str]
    recommended_actions: list[dict[str, Any]]
    technical_references: list[str]
    groundedness_score: float
    unsupported_claims: list[str]
    status: str

    def to_markdown(self) -> str:
        """Render diagnostic report into ASCII-safe Markdown format."""
        lines = [
            "# AI FIELD ENGINEER -- DIAGNOSTIC ASSESSMENT REPORT",
            f"**Case ID**: `{self.case_id}` | **Timestamp**: {self.timestamp} | **Status**: {self.status.upper()}",
            "---",
            "## 1. Equipment & Problem Summary",
            f"- **Equipment Type**: {self.equipment.get('type', 'Unknown').capitalize()}",
            f"- **Equipment Model**: {self.equipment.get('model', 'Not Specified')}",
            f"- **Technician Description**: {self.problem_summary}",
            f"- **Available Diagnostic Modalities**: {', '.join(self.available_modalities) if self.available_modalities else 'None'}",
            "",
            "## 2. Primary Diagnostic Assessment",
            f"- **Leading Diagnosis**: **{self.primary_diagnosis.upper()}**",
            f"- **Diagnostic Confidence**: **{self.diagnostic_confidence * 100:.1f}%**",
            f"- **Severity Level**: `{self.severity.value}`",
            f"- **Evidence Groundedness Score**: {self.groundedness_score * 100:.1f}%",
            "",
            "## 3. Key Supporting Evidence",
        ]

        if self.supporting_evidence:
            for ev in self.supporting_evidence:
                lines.append(
                    f"- **[{ev.get('evidence_type', 'EVIDENCE')}]** {ev.get('statement', '')} *(Source: {ev.get('source', 'Unknown')})*"
                )
        else:
            lines.append("- *No direct supporting evidence established.*")

        lines.append("\n## 4. Alternative Competing Hypotheses")
        if self.alternative_hypotheses:
            for hyp in self.alternative_hypotheses:
                lines.append(
                    f"- **{hyp.get('failure_mode', 'Unknown')}** (Likelihood: {hyp.get('likelihood_score', 0.0) * 100:.1f}%): {hyp.get('description', '')}"
                )
        else:
            lines.append("- *No alternative hypotheses considered.*")

        if self.contradictions_detected:
            lines.append("\n## 5. Detected Evidence Contradictions")
            for c in self.contradictions_detected:
                lines.append(
                    f"- [CONFLICT] {c.get('conflict_description')} (`{c.get('source_a')}` vs `{c.get('source_b')}`)"
                )

        lines.append("\n## 6. Missing Information & Investigation Gaps")
        if self.missing_information:
            for m in self.missing_information:
                lines.append(f"- [GAP] {m}")
        else:
            lines.append("- *No critical information gaps identified.*")

        lines.append("\n## 7. Recommended Action Plan")
        if self.recommended_actions:
            for act in sorted(self.recommended_actions, key=lambda x: x.get("priority", 99)):
                crit_badge = "[SAFETY CRITICAL] " if act.get("is_safety_critical") else ""
                ref_str = f" *(Ref: {act.get('source_reference')})*" if act.get("source_reference") else ""
                lines.append(
                    f"1. **{crit_badge}{act.get('action_text')}**{ref_str}\n   - *Rationale*: {act.get('rationale')}"
                )
        else:
            lines.append("- *No specific actions recommended.*")

        lines.append("\n## 8. Technical Manual & Knowledge References")
        if self.technical_references:
            for ref in self.technical_references:
                lines.append(f"- [REFERENCE] {ref}")
        else:
            lines.append("- *No technical manual references cited.*")

        if self.unsupported_claims:
            lines.append("\n## Groundedness Warnings")
            for u in self.unsupported_claims:
                lines.append(f"- [UNVERIFIED] {u}")

        lines.append("\n---")
        lines.append(
            "> **Disclaimer**: *This automated diagnostic report is generated by AI for advisory troubleshooting support. Field engineers must observe all plant safety procedures, lockout-tagout (LOTO) protocols, and OEM safety guidelines before performing physical maintenance.*"
        )

        return "\n".join(lines)
