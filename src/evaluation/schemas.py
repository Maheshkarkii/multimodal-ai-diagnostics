"""
Standardized Evaluation Schemas and Manifests for Phase 11.
Defines typed containers for evaluation cases, ground truth, predictions, metrics, and regression gates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class GroundTruthState(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ModalityAvailability(BaseModel):
    has_image: bool = False
    has_audio: bool = False
    has_sensor: bool = False
    has_text: bool = False


class EvaluationCase(BaseModel):
    """Unified evaluation case definition preventing data leakage."""
    case_id: str
    machine_id: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    equipment_type: str = "motor"
    equipment_model: Optional[str] = "M-4500"
    
    # Raw / Structured Modality Inputs
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    sensor_data: Optional[Dict[str, Any]] = None
    technician_description: Optional[str] = None
    
    # Ground Truth Annotations
    ground_truth_fault: Optional[str] = None
    ground_truth_severity: Optional[str] = None
    ground_truth_evidence_ids: List[str] = Field(default_factory=list)
    is_anomaly: Optional[bool] = None
    is_ood: bool = False
    
    # Modality Availability Flags
    available_modalities: ModalityAvailability = Field(default_factory=ModalityAvailability)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModalityMetrics(BaseModel):
    """Standardized classification & discrimination metrics."""
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    per_class_f1: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    roc_auc: Optional[float] = None
    sample_count: int = 0


class AnomalyMetrics(BaseModel):
    """Quantitative anomaly detection evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_alarm_rate: float = 0.0
    missed_anomaly_rate: float = 0.0
    threshold_applied: float = 0.0
    normal_count: int = 0
    anomalous_count: int = 0


class RAGRetrievalMetrics(BaseModel):
    """Technical Knowledge RAG evaluation metrics."""
    hit_rate_at_1: float = 0.0
    hit_rate_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    citation_accuracy: float = 0.0
    unsupported_claim_rate: float = 0.0
    evaluated_queries: int = 0


class CalibrationMetrics(BaseModel):
    """Confidence calibration metrics."""
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0
    confidence_vs_accuracy_gap: float = 0.0
    bin_confidences: List[float] = Field(default_factory=list)
    bin_accuracies: List[float] = Field(default_factory=list)


class RobustnessMetrics(BaseModel):
    """Metrics under missing/corrupted modalities."""
    full_modality_f1: float = 0.0
    missing_vision_f1: float = 0.0
    missing_audio_f1: float = 0.0
    missing_sensor_f1: float = 0.0
    missing_text_f1: float = 0.0
    corrupted_sensor_f1: float = 0.0
    abstention_rate_on_insufficient_data: float = 0.0


class LatencyBreakdown(BaseModel):
    """Latency profiling in milliseconds."""
    preprocessing_ms: float = 0.0
    vision_inference_ms: float = 0.0
    audio_inference_ms: float = 0.0
    sensor_inference_ms: float = 0.0
    fusion_inference_ms: float = 0.0
    rag_retrieval_ms: float = 0.0
    agent_reasoning_ms: float = 0.0
    explainability_ms: float = 0.0
    total_pipeline_ms: float = 0.0


class EvaluationSummary(BaseModel):
    """Master structured evaluation result for Phase 11."""
    evaluation_id: str
    timestamp: str
    dataset_version: str
    code_git_sha: str
    machine_split_strategy: str
    total_cases: int
    train_cases: int
    val_cases: int
    test_cases: int
    unique_machines: int
    unique_sessions: int
    
    # Subsystem Evaluation Results
    vision_metrics: Optional[ModalityMetrics] = None
    audio_metrics: Optional[ModalityMetrics] = None
    sensor_metrics: Optional[ModalityMetrics] = None
    anomaly_metrics: Optional[AnomalyMetrics] = None
    fusion_metrics: Optional[ModalityMetrics] = None
    rag_metrics: Optional[RAGRetrievalMetrics] = None
    calibration_metrics: Optional[CalibrationMetrics] = None
    robustness_metrics: Optional[RobustnessMetrics] = None
    latency_profile: Optional[LatencyBreakdown] = None
    
    # Regression Gate
    regression_passed: bool = True
    regression_details: Dict[str, Any] = Field(default_factory=dict)
