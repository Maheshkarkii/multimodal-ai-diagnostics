"""
Unit tests for core configuration, logging, and device resolution.
"""

import tempfile
from pathlib import Path

import torch

from field_engineer.core.config import DatasetConfig, ExperimentConfig, ModelConfig, SystemConfig, TrainingConfig
from field_engineer.core.device import resolve_device, set_seed
from field_engineer.core.logging import setup_logger


def test_default_config():
    cfg = ExperimentConfig()
    assert cfg.experiment_name == "baseline_experiment"
    assert cfg.system.seed == 42
    assert cfg.model.num_classes == 5
    assert cfg.training.optimizer == "adamw"


def test_config_yaml_roundtrip():
    cfg = ExperimentConfig(
        experiment_name="test_run",
        system=SystemConfig(seed=123, device="cpu"),
        model=ModelConfig(name="efficientnet_b0", num_classes=10),
        training=TrainingConfig(batch_size=64, learning_rate=3e-4),
        dataset=DatasetConfig(image_size=256),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_yaml = Path(tmp_dir) / "test_config.yaml"
        cfg.to_yaml(tmp_yaml)
        assert tmp_yaml.exists()

        loaded_cfg = ExperimentConfig.from_yaml(tmp_yaml)
        assert loaded_cfg.experiment_name == "test_run"
        assert loaded_cfg.system.seed == 123
        assert loaded_cfg.model.name == "efficientnet_b0"
        assert loaded_cfg.model.num_classes == 10
        assert loaded_cfg.training.batch_size == 64
        assert loaded_cfg.dataset.image_size == 256


def test_resolve_device_cpu():
    dev = resolve_device("cpu")
    assert dev == torch.device("cpu")


def test_set_seed_execution():
    set_seed(99)
    t1 = torch.rand(5)
    set_seed(99)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)


def test_logger_setup():
    logger = setup_logger("test_logger", level="DEBUG")
    assert logger.name == "test_logger"
    assert logger.level == 10  # DEBUG level int
