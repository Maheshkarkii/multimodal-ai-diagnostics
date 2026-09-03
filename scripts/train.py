"""
Unified Training Script for Industrial Fault Diagnostics.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.data.dataset import create_industrial_dataloaders
from src.data.generate_sample_dataset import generate_synthetic_industrial_dataset
from src.vision.model import build_vision_model
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Industrial Vision Diagnostic Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/frozen_baseline.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("IndustrialTraining", level=config.system.log_level)
    logger.info("Starting experiment: %s", config.experiment_name)

    data_dir = Path(config.dataset.dataset_dir)
    # Check if raw files exist
    has_files = any(data_dir.glob("*/*.*")) if data_dir.exists() else False
    if not has_files:
        logger.info("Generating verified baseline industrial dataset at %s...", data_dir)
        generate_synthetic_industrial_dataset(data_dir, samples_per_class=40, seed=config.system.seed)

    logger.info("Constructing group-aware industrial DataLoaders (leakage prevention: group_by=%s)...", config.dataset.group_by)
    train_loader, val_loader, test_loader, class_to_idx, class_weights = create_industrial_dataloaders(
        dataset_cfg=config.dataset,
        system_cfg=config.system,
        training_cfg=config.training,
    )

    logger.info(
        "DataLoaders ready: Train batches=%d, Val batches=%d, Test batches=%d, Classes=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        len(class_to_idx),
    )

    model = build_vision_model(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
        unfreeze_layers=config.model.unfreeze_layers,
        dropout=config.model.dropout,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights,
        logger=logger,
    )

    history = trainer.train(resume_path=args.resume)
    logger.info("Training cycle finished successfully for %s.", config.experiment_name)


if __name__ == "__main__":
    main()
