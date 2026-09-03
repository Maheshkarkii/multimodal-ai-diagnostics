"""
Training Script for Phase 1 Vision Diagnostic Baseline.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.data.dataset import create_fashionmnist_dataloaders
from src.vision.model import build_vision_model
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Diagnostic Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vision.yaml",
        help="Path to YAML experiment configuration",
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
    logger = setup_logger("VisionTraining", level=config.system.log_level)
    logger.info("Loaded configuration from %s", args.config)

    logger.info("Initializing dataset pipelines...")
    train_loader, val_loader, test_loader, class_names = create_fashionmnist_dataloaders(
        dataset_cfg=config.dataset,
        system_cfg=config.system,
        training_cfg=config.training,
    )
    logger.info(
        "DataLoaders ready: Train batches=%d, Val batches=%d, Test batches=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    logger.info(
        "Building %s classifier (num_classes=%d, pretrained=%s, freeze_backbone=%s)...",
        config.model.name,
        config.model.num_classes,
        config.model.pretrained,
        config.model.freeze_backbone,
    )
    model = build_vision_model(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
        dropout=config.model.dropout,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
    )

    history = trainer.train(resume_path=args.resume)
    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
