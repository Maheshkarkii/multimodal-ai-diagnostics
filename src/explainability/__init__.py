"""
Explainability module main exports.
"""

from src.explainability.audio.spectrogram import generate_spectrogram_visualization
from src.explainability.audit.audit_service import AuditService
from src.explainability.core.config import (
    AudioExplainabilityConfig,
    AuditConfig,
    CitationValidationConfig,
    ExplainabilityConfig,
    SensorExplainabilityConfig,
    VisionExplainabilityConfig,
)
from src.explainability.core.schema import (
    ActionRequirement,
    AuditableDiagnosticReport,
    AuditableEvidenceItem,
    AuditTrailRecord,
    ClaimEvidenceMapping,
    ClaimSupportStatus,
    ConfidenceDecomposition,
    DiagnosticSystemStatus,
    EvidenceCategory,
    EvidenceQuality,
    TraceableRecommendedAction,
    UncertaintyProfile,
)
from src.explainability.core.service import ExplainabilityService
from src.explainability.sensor.telemetry_plot import generate_sensor_threshold_plot
from src.explainability.vision.gradcam import generate_gradcam_visualization

__all__ = [
    "ExplainabilityConfig",
    "VisionExplainabilityConfig",
    "AudioExplainabilityConfig",
    "SensorExplainabilityConfig",
    "CitationValidationConfig",
    "AuditConfig",
    "EvidenceCategory",
    "EvidenceQuality",
    "ClaimSupportStatus",
    "ActionRequirement",
    "DiagnosticSystemStatus",
    "AuditableEvidenceItem",
    "ClaimEvidenceMapping",
    "ConfidenceDecomposition",
    "TraceableRecommendedAction",
    "UncertaintyProfile",
    "AuditTrailRecord",
    "AuditableDiagnosticReport",
    "ExplainabilityService",
    "AuditService",
    "generate_gradcam_visualization",
    "generate_spectrogram_visualization",
    "generate_sensor_threshold_plot",
]
