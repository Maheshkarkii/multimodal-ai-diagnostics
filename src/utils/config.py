"""
Unified Experiment Configuration system supporting Vision, Audio, Sensor, and Multimodal Fusion.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SystemConfig:
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    output_dir: str = "reports/experiments"
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    rotation_degrees: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    freq_mask: int = 8
    time_mask: int = 16


@dataclass
class AnomalyDetectorConfig:
    method: str = "isolation_forest"
    contamination: float = 0.05
    n_estimators: int = 100


@dataclass
class MultimodalConfig:
    vision_dim: int = 1280
    audio_dim: int = 512
    sensor_dim: int = 256
    text_dim: int = 256
    shared_projection_dim: int = 256
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    unified_embedding_dim: int = 256
    dropout: float = 0.25
    modality_dropout_prob: float = 0.20
    enabled_modalities: list[str] = field(default_factory=lambda: ["vision", "audio", "sensor", "text"])


@dataclass
class DatasetConfig:
    name: str = "industrial_diagnostics"
    dataset_dir: str = "data/raw"
    manifest_path: str | None = None
    image_size: int = 224
    sample_rate: int = 16000
    duration: float = 3.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    f_min: float = 50.0
    f_max: float = 8000.0
    val_split: float = 0.15
    test_split: float = 0.15
    num_classes: int = 5
    classes: list[str] = field(
        default_factory=lambda: [
            "normal",
            "fault_1",
            "fault_2",
            "fault_3",
            "fault_4",
        ]
    )
    group_by: str | None = None
    timestamp_column: str | None = None
    target_column: str | None = None
    feature_columns: list[str] | None = None
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ModelConfig:
    name: str = "multimodal_fusion"
    num_classes: int = 5
    in_channels: int = 3
    in_features: int = 6
    hidden_dims: list[int] = field(default_factory=lambda: [128, 256, 128])
    embedding_dim: int = 256
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_layers: int | None = None
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    backbone_learning_rate: float | None = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    early_stopping_patience: int = 4
    gradient_clip_val: float | None = 1.0
    use_class_weights: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "diagnostics_baseline"
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    anomaly_detector: AnomalyDetectorConfig = field(default_factory=AnomalyDetectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        dataset_data = d.get("dataset", {}).copy()
        aug_data = dataset_data.pop("augmentations", {})
        if isinstance(aug_data, dict):
            aug_cfg = AugmentationConfig(**aug_data)
        else:
            aug_cfg = AugmentationConfig()

        anom_data = d.get("anomaly_detector", {})
        anom_cfg = AnomalyDetectorConfig(**anom_data) if isinstance(anom_data, dict) else AnomalyDetectorConfig()

        mm_data = d.get("multimodal", {})
        mm_cfg = MultimodalConfig(**mm_data) if isinstance(mm_data, dict) else MultimodalConfig()

        model_data = d.get("model", {})
        training_data = d.get("training", {})
        system_data = d.get("system", {})

        return cls(
            experiment_name=d.get("experiment_name", "diagnostics_baseline"),
            system=SystemConfig(**system_data),
            dataset=DatasetConfig(augmentations=aug_cfg, **dataset_data),
            model=ModelConfig(**model_data),
            multimodal=mm_cfg,
            anomaly_detector=anom_cfg,
            training=TrainingConfig(**training_data),
        )

    def to_yaml(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
