"""
Training CLI for Phase 3 Acoustic Intelligence.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.audio.data.audio_dataset import create_audio_dataloaders
from src.audio.data.generate_sample_audio import generate_synthetic_acoustic_dataset
from src.audio.models.audio_cnn import build_audio_model
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Acoustic Diagnostic CNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/audio.yaml",
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
    logger = setup_logger("AudioTraining", level=config.system.log_level)
    logger.info("Starting acoustic experiment: %s", config.experiment_name)

    data_dir = Path(config.dataset.dataset_dir)
    has_audio = any(data_dir.glob("*/*.wav")) if data_dir.exists() else False
    if not has_audio:
        logger.info("Generating verified baseline machine sound dataset at %s...", data_dir)
        generate_synthetic_acoustic_dataset(data_dir, samples_per_class=30, seed=config.system.seed)

    logger.info("Constructing group-aware audio DataLoaders (leakage prevention: group_by=%s)...", config.dataset.group_by)
    train_loader, val_loader, test_loader, class_to_idx, class_weights = create_audio_dataloaders(config)

    logger.info(
        "Audio DataLoaders ready: Train batches=%d, Val batches=%d, Test batches=%d, Classes=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        len(class_to_idx),
    )

    model = build_audio_model(
        num_classes=len(config.dataset.classes),
        in_channels=1,
        embedding_dim=512,
        dropout=0.3,
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
    logger.info("Acoustic training cycle completed successfully for %s.", config.experiment_name)


if __name__ == "__main__":
    main()
