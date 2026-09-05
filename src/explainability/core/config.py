"""
Configuration Dataclasses for Phase 8 Explainability and Auditable Reporting.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VisionExplainabilityConfig:
    """Settings for vision saliency and feature attribution."""

    method: str = "gradcam"  # "gradcam", "saliency", "occlusion", "none"
    target_layer: str = "features"  # Backbone feature extraction layer
    colormap: str = "inferno"
    alpha_overlay: float = 0.5
    save_visualizations: bool = True
    output_dir: str = "reports/explainability/vision"


@dataclass
class AudioExplainabilityConfig:
    """Settings for acoustic feature and spectrogram visualization."""

    generate_mel_spectrogram: bool = True
    highlight_frequency_bands: bool = True
    save_visualizations: bool = True
    output_dir: str = "reports/explainability/audio"


@dataclass
class SensorExplainabilityConfig:
    """Settings for sensor threshold analysis and deviation plotting."""

    generate_radar_plot: bool = True
    generate_envelope_plot: bool = True
    save_visualizations: bool = True
    output_dir: str = "reports/explainability/sensor"


@dataclass
class CitationValidationConfig:
    """Settings for technical document citation verification."""

    strict_verification: bool = True
    require_page_match: bool = True
    require_section_match: bool = False
    lexical_overlap_threshold: float = 0.20


@dataclass
class AuditConfig:
    """Settings for auditable diagnostic execution tracking."""

    enable_audit_trail: bool = True
    audit_storage_dir: str = "reports/audit"
    record_input_hashes: bool = True
    record_model_checkpoints: bool = True


@dataclass
class ExplainabilityConfig:
    """Master Explainability and Auditable Diagnostic Report Configuration."""

    system_name: str = "explainability_and_auditable_reports"
    report_version: str = "1.0.0"
    vision_model_version: str = "vision_mobilenetv2_v1"
    audio_model_version: str = "audio_cnn_v1"
    sensor_model_version: str = "sensor_mlp_v1"
    knowledge_base_version: str = "rag_manuals_v1_2026_09"
    reports_output_dir: str = "reports/diagnostics"
    max_evidence_items: int = 25
    vision: VisionExplainabilityConfig = field(default_factory=VisionExplainabilityConfig)
    audio: AudioExplainabilityConfig = field(default_factory=AudioExplainabilityConfig)
    sensor: SensorExplainabilityConfig = field(default_factory=SensorExplainabilityConfig)
    citation_validation: CitationValidationConfig = field(default_factory=CitationValidationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExplainabilityConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Explainability config file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExplainabilityConfig":
        return cls(
            system_name=d.get("system_name", "explainability_and_auditable_reports"),
            report_version=d.get("report_version", "1.0.0"),
            vision_model_version=d.get("vision_model_version", "vision_mobilenetv2_v1"),
            audio_model_version=d.get("audio_model_version", "audio_cnn_v1"),
            sensor_model_version=d.get("sensor_model_version", "sensor_mlp_v1"),
            knowledge_base_version=d.get("knowledge_base_version", "rag_manuals_v1_2026_09"),
            reports_output_dir=d.get("reports_output_dir", "reports/diagnostics"),
            max_evidence_items=d.get("max_evidence_items", 25),
            vision=VisionExplainabilityConfig(**d.get("vision", {})),
            audio=AudioExplainabilityConfig(**d.get("audio", {})),
            sensor=SensorExplainabilityConfig(**d.get("sensor", {})),
            citation_validation=CitationValidationConfig(**d.get("citation_validation", {})),
            audit=AuditConfig(**d.get("audit", {})),
        )

    def to_yaml(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
