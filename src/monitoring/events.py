"""
Structured Production Monitoring & Ingestion Event Schema.
Sanitizes raw user data and exposes latency, confidence, status, and error taxonomy.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureTaxonomy(str, Enum):
    INPUT_INVALID = "INPUT_INVALID"
    INPUT_LOW_QUALITY = "INPUT_LOW_QUALITY"
    MODEL_MISCLASSIFICATION = "MODEL_MISCLASSIFICATION"
    SENSOR_ANOMALY_FALSE_POSITIVE = "SENSOR_ANOMALY_FALSE_POSITIVE"
    SENSOR_ANOMALY_FALSE_NEGATIVE = "SENSOR_ANOMALY_FALSE_NEGATIVE"
    MULTIMODAL_CONFLICT = "MULTIMODAL_CONFLICT"
    MISSING_MODALITY = "MISSING_MODALITY"
    RAG_NO_EVIDENCE = "RAG_NO_EVIDENCE"
    RAG_IRRELEVANT_EVIDENCE = "RAG_IRRELEVANT_EVIDENCE"
    RAG_CITATION_ERROR = "RAG_CITATION_ERROR"
    DIAGNOSTIC_ERROR = "DIAGNOSTIC_ERROR"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    HALLUCINATION = "HALLUCINATION"
    ABSTENTION_ERROR = "ABSTENTION_ERROR"
    TIMEOUT = "TIMEOUT"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class DiagnosticMonitoringEvent(BaseModel):
    """Sanitized telemetry event emitted on every diagnostic inference."""
    event_id: str
    request_id: str
    case_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    environment: str = "production"
    
    # Subsystem Availability & Status
    available_modalities: Dict[str, bool] = Field(default_factory=dict)
    diagnostic_status: str = "COMPLETED"
    primary_diagnosis: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    
    # RAG & Verification Metrics
    retrieval_count: int = 0
    citation_count: int = 0
    
    # Latency Breakdown (ms)
    pipeline_latency_ms: float = 0.0
    
    # Failure categorization
    error_type: Optional[FailureTaxonomy] = None
    is_abstention: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
