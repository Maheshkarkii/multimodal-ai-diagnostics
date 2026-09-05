"""
Pydantic Request & Response Schemas for Phase 9 FastAPI Service.
"""

from typing import Any

from pydantic import BaseModel, Field


# --- Health & Readiness Schemas ---
class HealthResponse(BaseModel):
    status: str = Field(json_schema_extra={"example": "healthy"})
    service: str = Field(json_schema_extra={"example": "ai-field-engineer-api"})
    version: str = Field(json_schema_extra={"example": "1.0.0"})
    environment: str = Field("production", json_schema_extra={"example": "production"})
    git_sha: str = Field("unknown", json_schema_extra={"example": "5963611"})
    timestamp: str


class ReadinessResponse(BaseModel):
    ready: bool
    status: str
    components: dict[str, bool] = Field(
        json_schema_extra={
            "example": {
                "vision_service": True,
                "audio_service": True,
                "sensor_service": True,
                "multimodal_fusion": True,
                "rag_retrieval": True,
                "diagnostic_agent": True,
                "explainability_service": True,
            }
        }
    )
    timestamp: str


# --- Sensor & Telemetry Input Schemas ---
class SensorTelemetryInput(BaseModel):
    temperature: float | None = Field(
        None, description="Bearing or casing temperature", json_schema_extra={"example": 84.0}
    )
    temperature_unit: str = Field("degC", description="Temperature unit (degC or degF)")
    vibration: float | None = Field(None, description="RMS vibration velocity", json_schema_extra={"example": 6.8})
    vibration_unit: str = Field("mm/s", description="Vibration unit (mm/s or in/s)")
    rpm: float | None = Field(
        None, description="Operating shaft rotational speed", json_schema_extra={"example": 1480.0}
    )
    current: float | None = Field(None, description="Motor operating current", json_schema_extra={"example": 8.4})
    current_unit: str = Field("A", description="Current unit")
    pressure: float | None = Field(
        None, description="Suction or discharge pressure", json_schema_extra={"example": 1.2}
    )
    pressure_unit: str = Field("bar", description="Pressure unit")
    custom_parameters: dict[str, float] = Field(
        default_factory=dict, description="Additional arbitrary sensor readings"
    )


# --- Equipment Metadata ---
class EquipmentMetadataInput(BaseModel):
    equipment_type: str = Field(
        default="motor",
        description="Equipment classification (e.g. motor, pump, gearbox)",
        json_schema_extra={"example": "motor"},
    )
    manufacturer: str | None = Field(
        default=None, description="Equipment manufacturer", json_schema_extra={"example": "Siemens"}
    )
    model: str | None = Field(
        default=None, description="Equipment model identifier", json_schema_extra={"example": "M-4500"}
    )
    serial_number: str | None = Field(
        default=None, description="Asset serial number", json_schema_extra={"example": "SN-94821"}
    )
    operating_mode: str | None = Field(default="continuous", description="Duty cycle mode")


# --- Evidence & Action Output Schemas ---
class EvidenceItemResponse(BaseModel):
    evidence_id: str
    category: str
    source: str
    description: str
    quality: str
    raw_value: float | None = None
    unit: str | None = None
    document_name: str | None = None
    page_number: int | None = None
    section: str | None = None


class RecommendedActionResponse(BaseModel):
    action_id: str
    priority: int
    requirement: str
    action_text: str
    rationale: str
    source_reference: str | None = None
    is_safety_critical: bool
    justifying_evidence_ids: list[str] = Field(default_factory=list)


class ClaimAuditMappingResponse(BaseModel):
    claim_id: str
    claim_statement: str
    status: str
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    rationale: str | None = None


class ConfidenceDecompositionResponse(BaseModel):
    overall_confidence: float
    multimodal_agreement: str
    sensor_evidence_strength: str
    acoustic_evidence_strength: str
    visual_evidence_strength: str
    technical_knowledge_match: str
    contradiction_penalty: float
    rationale_summary: str


class PrimaryDiagnosisResponse(BaseModel):
    primary_diagnosis: str
    diagnostic_confidence: float
    severity: str
    confidence_decomposition: ConfidenceDecompositionResponse


# --- Main Diagnostic Response Schema ---
class DiagnosisResponse(BaseModel):
    case_id: str
    request_id: str
    timestamp: str
    status: str
    equipment: dict[str, Any]
    problem_summary: str
    available_modalities: list[str]
    diagnosis: PrimaryDiagnosisResponse
    evidence_inventory: list[EvidenceItemResponse]
    claim_mappings: list[ClaimAuditMappingResponse]
    alternative_hypotheses: list[dict[str, Any]]
    recommended_actions: list[RecommendedActionResponse]
    uncertainty_profile: dict[str, Any]
    unsupported_claims: list[str]
    audit_summary: dict[str, Any]
    markdown_report: str


# --- Error Response Schema ---
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None
    timestamp: str
