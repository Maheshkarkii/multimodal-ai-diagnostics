"""
Data Models and Structured Schemas for Phase 8 Explainability and Auditable Reports.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceCategory(str, Enum):
    VISUAL = "VISUAL"
    ACOUSTIC = "ACOUSTIC"
    SENSOR = "SENSOR"
    TECHNICIAN = "TECHNICIAN"
    TECHNICAL_DOCUMENT = "TECHNICAL_DOCUMENT"
    MODEL_FUSION = "MODEL_FUSION"


class EvidenceQuality(str, Enum):
    HIGH = "HIGH"  # Direct physical sensor or validated OEM manual citation
    MEDIUM = "MEDIUM"  # High-confidence model prediction or technician symptom
    LOW = "LOW"  # Low-confidence model inference or uncorroborated claim


class ClaimSupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class ActionRequirement(str, Enum):
    REQUIRED = "REQUIRED"  # Mandatory safety-critical protocol
    RECOMMENDED = "RECOMMENDED"  # High-yield inspection procedure
    OPTIONAL = "OPTIONAL"  # Preventive optimization


class DiagnosticSystemStatus(str, Enum):
    DIAGNOSIS_SUPPORTED = "DIAGNOSIS_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    REQUIRES_HUMAN_INSPECTION = "REQUIRES_HUMAN_INSPECTION"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class AuditableEvidenceItem:
    """Standardized atomic evidence unit with stable ID and full provenance."""

    evidence_id: str  # e.g., "SEN-001", "DOC-002", "AUD-001"
    category: EvidenceCategory
    source: str  # e.g., "Vibration Sensor", "Motor Manual (Page 2)"
    description: str  # Human-readable observation description
    quality: EvidenceQuality = EvidenceQuality.MEDIUM
    raw_value: float | None = None  # e.g., 6.8
    unit: str | None = None  # e.g., "mm/s"
    model_name: str | None = None  # e.g., "AcousticCNN"
    prediction: str | None = None  # e.g., "bearing_defect_wear"
    confidence: float | None = None  # e.g., 0.88
    document_name: str | None = None  # e.g., "motor_m4500_maintenance_manual.pdf"
    page_number: int | None = None  # e.g., 2
    section: str | None = None  # e.g., "BEARING INSPECTION"
    chunk_id: str | None = None
    visualization_artifact: str | None = None  # File path to Grad-CAM heatmap or spectrogram
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["quality"] = self.quality.value
        return d


@dataclass
class ClaimEvidenceMapping:
    """Explicit bidirectional connection between a diagnostic claim and supporting evidence."""

    claim_id: str
    claim_statement: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    status: ClaimSupportStatus = ClaimSupportStatus.SUPPORTED
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ConfidenceDecomposition:
    """Multifactorial explanatory decomposition of overall diagnostic confidence."""

    overall_confidence: float  # e.g., 0.86 (86.0%)
    multimodal_agreement: str  # "HIGH", "MEDIUM", "LOW"
    sensor_evidence_strength: str  # "HIGH", "MEDIUM", "LOW", "UNAVAILABLE"
    acoustic_evidence_strength: str  # "HIGH", "MEDIUM", "LOW", "UNAVAILABLE"
    visual_evidence_strength: str  # "HIGH", "MEDIUM", "LOW", "UNAVAILABLE"
    technical_knowledge_match: str  # "HIGH", "MEDIUM", "LOW", "UNAVAILABLE"
    contradiction_penalty: float = 0.0  # Confidence reduction amount from detected conflicts
    rationale_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceableRecommendedAction:
    """Actionable recommendation with explicit requirement level and technical citation."""

    action_id: str  # e.g., "ACT-001"
    priority: int  # 1 (Immediate) to 5 (Long-term)
    requirement: ActionRequirement
    action_text: str
    rationale: str
    justifying_evidence_ids: list[str] = field(default_factory=list)
    source_reference: str | None = None
    is_safety_critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["requirement"] = self.requirement.value
        return d


@dataclass
class UncertaintyProfile:
    """Transparent cataloging of known facts, missing signals, and uncertainty reduction steps."""

    confirmed_facts: list[str] = field(default_factory=list)
    unknown_parameters: list[str] = field(default_factory=list)
    recommended_investigation_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditTrailRecord:
    """Immutable operational record of an entire diagnostic execution for regulatory audit."""

    case_id: str
    timestamp: str
    system_version: str
    report_version: str
    model_versions: dict[str, str]
    knowledge_base_version: str
    input_hashes: dict[str, str]
    available_modalities: list[str]
    retrieval_queries: list[str]
    retrieved_chunk_ids: list[str]
    final_diagnosis: str
    diagnostic_confidence: float
    status: str
    execution_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditableDiagnosticReport:
    """Master structured and auditable diagnostic report."""

    case_id: str
    timestamp: str
    report_version: str
    system_status: DiagnosticSystemStatus
    equipment_info: dict[str, Any]
    problem_summary: str
    primary_diagnosis: str
    severity: str
    confidence_decomposition: ConfidenceDecomposition
    evidence_inventory: list[AuditableEvidenceItem]
    claim_mappings: list[ClaimEvidenceMapping]
    alternative_hypotheses: list[dict[str, Any]]
    recommended_actions: list[TraceableRecommendedAction]
    uncertainty_profile: UncertaintyProfile
    unsupported_claims: list[str]
    audit_record: AuditTrailRecord
    limitations: list[str] = field(
        default_factory=lambda: [
            "This automated diagnostic report is an AI-assisted decision support tool, not a certified structural engineer.",
            "Feature attribution heatmaps and spectrogram overlays reflect model attention, not physical causal proof.",
            "All physical maintenance interventions must adhere strictly to plant Lockout-Tagout (LOTO) procedures and OEM manuals.",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["system_status"] = self.system_status.value
        d["confidence_decomposition"] = self.confidence_decomposition.to_dict()
        d["evidence_inventory"] = [e.to_dict() for e in self.evidence_inventory]
        d["claim_mappings"] = [c.to_dict() for c in self.claim_mappings]
        d["recommended_actions"] = [a.to_dict() for a in self.recommended_actions]
        d["uncertainty_profile"] = self.uncertainty_profile.to_dict()
        d["audit_record"] = self.audit_record.to_dict()
        return d

    def to_markdown(self) -> str:
        """Render complete, professional auditable Markdown report."""
        eq_type = self.equipment_info.get("type", "Unknown").capitalize()
        eq_model = self.equipment_info.get("model", "Unspecified")

        lines = [
            "# AI FIELD ENGINEER -- AUDITABLE DIAGNOSTIC ASSESSMENT REPORT",
            f"**Case ID**: `{self.case_id}` | **Timestamp**: {self.timestamp} | **Report Version**: `v{self.report_version}`",
            f"**Diagnostic System Status**: `{self.system_status.value}`",
            "---",
            "## 1. Equipment & Problem Context",
            f"- **Equipment Type**: {eq_type}",
            f"- **Model Identifier**: {eq_model}",
            f"- **Field Technician Description**: {self.problem_summary}",
            f"- **Diagnostic Knowledge Base**: `{self.audit_record.knowledge_base_version}`",
            "",
            "## 2. Primary Diagnostic Assessment",
            f"- **Leading Diagnosis**: **{self.primary_diagnosis.upper()}**",
            f"- **Diagnostic Confidence**: **{self.confidence_decomposition.overall_confidence * 100:.1f}%**",
            f"- **Operational Severity**: `{self.severity.upper()}`",
            f"- **Confidence Rationale**: {self.confidence_decomposition.rationale_summary}",
            "",
            "### Confidence Contributing Factors:",
            "| Factor | Assessment | Description |",
            "| :--- | :---: | :--- |",
            f"| **Multimodal Agreement** | `{self.confidence_decomposition.multimodal_agreement}` | Cross-modality prediction consistency |",
            f"| **Sensor Telemetry** | `{self.confidence_decomposition.sensor_evidence_strength}` | Physical telemetry threshold evaluation |",
            f"| **Acoustic Audio** | `{self.confidence_decomposition.acoustic_evidence_strength}` | Harmonic & acoustic spectrum features |",
            f"| **Visual Inspection** | `{self.confidence_decomposition.visual_evidence_strength}` | Surface defect & camera observations |",
            f"| **Technical Manual Grounding** | `{self.confidence_decomposition.technical_knowledge_match}` | OEM SOP & specification retrieval |",
            "",
            "## 3. Auditable Evidence Inventory",
            "| Evidence ID | Category | Quality | Source / Provenance | Observation Description |",
            "| :--- | :---: | :---: | :--- | :--- |",
        ]

        for ev in self.evidence_inventory:
            val_str = f" ({ev.raw_value} {ev.unit})" if ev.raw_value is not None else ""
            lines.append(
                f"| **[{ev.evidence_id}]** | `{ev.category.value}` | `{ev.quality.value}` | {ev.source} | {ev.description}{val_str} |"
            )

        lines.append("\n## 4. Claim-to-Evidence Audit Trace")
        if self.claim_mappings:
            for cm in self.claim_mappings:
                sup_str = (
                    ", ".join([f"`[{e}]`" for e in cm.supporting_evidence_ids])
                    if cm.supporting_evidence_ids
                    else "*None*"
                )
                con_str = (
                    ", ".join([f"`[{e}]`" for e in cm.contradicting_evidence_ids])
                    if cm.contradicting_evidence_ids
                    else "*None*"
                )
                lines.append(f'### Claim: "{cm.claim_statement}"')
                lines.append(f"- **Verification Status**: `{cm.status.value}`")
                lines.append(f"- **Supporting Evidence**: {sup_str}")
                lines.append(f"- **Contradicting Evidence**: {con_str}")
                if cm.rationale:
                    lines.append(f"- **Audit Rationale**: {cm.rationale}")
                lines.append("")
        else:
            lines.append("- *No claim mappings registered.*")

        lines.append("## 5. Alternative Competing Hypotheses")
        if self.alternative_hypotheses:
            for idx, hyp in enumerate(self.alternative_hypotheses, start=1):
                prob = (
                    f" (Likelihood: {hyp.get('likelihood_score', 0.0) * 100:.1f}%)" if "likelihood_score" in hyp else ""
                )
                lines.append(f"{idx}. **{hyp.get('failure_mode', 'Unknown')}**{prob}: {hyp.get('description', '')}")
        else:
            lines.append("- *No alternative hypotheses considered.*")

        lines.append("\n## 6. Traceable Action Plan")
        if self.recommended_actions:
            for act in sorted(self.recommended_actions, key=lambda x: x.priority):
                crit = "[SAFETY CRITICAL] " if act.is_safety_critical else ""
                req_badge = f"`[{act.requirement.value}]` "
                ref = f" *(Ref: {act.source_reference})*" if act.source_reference else ""
                ev_ids = (
                    f" *(Evidence: {', '.join([f'[{e}]' for e in act.justifying_evidence_ids])})*"
                    if act.justifying_evidence_ids
                    else ""
                )
                lines.append(
                    f"1. {req_badge}**{crit}{act.action_text}**{ref}{ev_ids}\n   - *Technical Rationale*: {act.rationale}"
                )
        else:
            lines.append("- *No specific actions prescribed.*")

        lines.append("\n## 7. Uncertainty & Investigation Gaps")
        lines.append("### What the System Confirmed:")
        for fact in self.uncertainty_profile.confirmed_facts:
            lines.append(f"- [CONFIRMED] {fact}")
        lines.append("\n### What is Currently Unknown:")
        for unk in self.uncertainty_profile.unknown_parameters:
            lines.append(f"- [UNKNOWN] {unk}")
        lines.append("\n### Recommended Steps to Reduce Diagnostic Uncertainty:")
        for step in self.uncertainty_profile.recommended_investigation_steps:
            lines.append(f"- [ACTION] {step}")

        if self.unsupported_claims:
            lines.append("\n## ⚠️ Groundedness Warnings")
            for u in self.unsupported_claims:
                lines.append(f"- [UNVERIFIED] {u}")

        lines.append("\n## 8. Audit Trail & Reproducibility Record")
        lines.append(f"- **Execution Timestamp**: `{self.audit_record.timestamp}`")
        lines.append(f"- **Execution Latency**: `{self.audit_record.execution_duration_ms:.2f} ms`")
        lines.append(f"- **Vision Model Version**: `{self.audit_record.model_versions.get('vision', 'N/A')}`")
        lines.append(f"- **Acoustic Model Version**: `{self.audit_record.model_versions.get('audio', 'N/A')}`")
        lines.append(f"- **Sensor Model Version**: `{self.audit_record.model_versions.get('sensor', 'N/A')}`")
        lines.append(f"- **Retrieved Knowledge Chunks**: `{len(self.audit_record.retrieved_chunk_ids)} chunks indexed`")

        lines.append("\n## 9. Limitations & Advisory Disclaimer")
        for lim in self.limitations:
            lines.append(f"- {lim}")

        lines.append("\n---")
        lines.append("*Report generated by AI Field Engineer Explainability & Audit Engine.*")

        return "\n".join(lines)
