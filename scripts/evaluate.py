"""
Evaluation Script for Phase 1 Vision Diagnostic Baseline.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.data.dataset import create_fashionmnist_dataloaders
from src.vision.model import build_vision_model
from src.evaluation.evaluator import Evaluator
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Vision Diagnostic Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vision.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint .pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("VisionEvaluation", level=config.system.log_level)

    _, _, test_loader, class_names = create_fashionmnist_dataloaders(
        dataset_cfg=config.dataset,
        system_cfg=config.system,
        training_cfg=config.training,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Please train first.", ckpt_path)
        sys.exit(1)

    model = build_vision_model(
        num_classes=config.model.num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    evaluator = Evaluator(
        model=model,
        device=config.system.device,
        class_names=class_names,
        logger=logger,
    )
    metrics = evaluator.evaluate(test_loader)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
