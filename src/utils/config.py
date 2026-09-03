"""
Configuration management utilities.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


@dataclass
class SystemConfig:
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    output_dir: str = "experiments/outputs"
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"


@dataclass
class DatasetConfig:
    name: str = "fashion_mnist"
    data_dir: str = "data/raw"
    image_size: int = 224
    val_split: float = 0.15
    num_classes: int = 10


@dataclass
class ModelConfig:
    name: str = "mobilenet_v2"
    num_classes: int = 10
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    early_stopping_patience: int = 3
    gradient_clip_val: Optional[float] = 1.0


@dataclass
class ExperimentConfig:
    experiment_name: str = "mobilenetv2_fashionmnist_baseline"
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
        return cls(
            experiment_name=d.get("experiment_name", "mobilenetv2_fashionmnist_baseline"),
            system=SystemConfig(**d.get("system", {})),
            dataset=DatasetConfig(**d.get("dataset", {})),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    def to_yaml(self, save_path: Union[str, Path]) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
