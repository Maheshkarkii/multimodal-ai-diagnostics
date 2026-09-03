"""
Hierarchical, typed experiment configuration system for industrial diagnostics.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import yaml


@dataclass
class SystemConfig:
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 2
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


@dataclass
class DatasetConfig:
    name: str = "industrial_fault_inspection"
    dataset_dir: str = "data/industrial_inspection"
    image_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    num_classes: int = 5
    classes: List[str] = field(default_factory=lambda: [
        "normal",
        "bearing_fault",
        "corrosion",
        "surface_crack",
        "damaged_component",
    ])
    group_by: Optional[str] = None  # e.g., "equipment_id" to prevent data leakage
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ModelConfig:
    name: str = "mobilenet_v2"
    num_classes: int = 5
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_layers: Optional[int] = None  # Number of deeper layers/blocks to unfreeze
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    backbone_learning_rate: Optional[float] = 1e-4  # Discriminative LR for fine-tuning
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    early_stopping_patience: int = 4
    gradient_clip_val: Optional[float] = 1.0
    use_class_weights: bool = True  # Handle real-world class imbalance


@dataclass
class ExperimentConfig:
    experiment_name: str = "industrial_vision_baseline"
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        dataset_data = d.get("dataset", {})
        aug_data = dataset_data.pop("augmentations", {})
        if isinstance(aug_data, dict):
            aug_cfg = AugmentationConfig(**aug_data)
        else:
            aug_cfg = AugmentationConfig()

        return cls(
            experiment_name=d.get("experiment_name", "industrial_vision_baseline"),
            system=SystemConfig(**d.get("system", {})),
            dataset=DatasetConfig(augmentations=aug_cfg, **dataset_data),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    def to_yaml(self, save_path: Union[str, Path]) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
