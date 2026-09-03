"""
Configuration management system.

Supports dataclass-based schemas with YAML serialization/deserialization,
default fallbacks, and validation for reproducible experiments.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


@dataclass
class SystemConfig:
    """Core hardware and execution configuration."""
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    output_dir: str = "experiments/outputs"
    log_level: str = "INFO"


@dataclass
class ModelConfig:
    """Base model architecture settings."""
    name: str = "resnet18"
    num_classes: int = 5
    pretrained: bool = True
    in_channels: int = 3
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Optimization and training hyperparameter settings."""
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "none"
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    gradient_clip_val: Optional[float] = 1.0


@dataclass
class DatasetConfig:
    """Dataset and path configurations."""
    name: str = "synthetic_industrial_vision"
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    image_size: int = 224
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ExperimentConfig:
    """Top-level unified experiment configuration."""
    experiment_name: str = "baseline_experiment"
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        """Load experiment config from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Construct ExperimentConfig from a nested dictionary."""
        system_data = d.get("system", {})
        dataset_data = d.get("dataset", {})
        model_data = d.get("model", {})
        training_data = d.get("training", {})

        return cls(
            experiment_name=d.get("experiment_name", "baseline_experiment"),
            system=SystemConfig(**system_data),
            dataset=DatasetConfig(**dataset_data),
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
        )

    def to_yaml(self, save_path: Union[str, Path]) -> None:
        """Serialize configuration to a YAML file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
