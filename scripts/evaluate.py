"""
Standalone Evaluation & Error Analysis Script.
"""

import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.data.dataset import create_industrial_dataloaders
from src.data.generate_sample_dataset import generate_synthetic_industrial_dataset
from src.vision.model import build_vision_model
from src.evaluation.evaluator import Evaluator
from src.analysis.error_analysis import DiagnosticErrorAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Industrial Vision Diagnostic Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/frozen_baseline.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/exp_frozen_baseline_best.pt",
        help="Path to trained model checkpoint .pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("IndustrialEvaluation", level=config.system.log_level)

    data_dir = Path(config.dataset.dataset_dir)
    has_files = any(data_dir.glob("*/*.*")) if data_dir.exists() else False
    if not has_files:
        generate_synthetic_industrial_dataset(data_dir, samples_per_class=40, seed=config.system.seed)

    _, _, test_loader, class_to_idx, _ = create_industrial_dataloaders(
        dataset_cfg=config.dataset,
        system_cfg=config.system,
        training_cfg=config.training,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Please train first.", ckpt_path)
        sys.exit(1)

    class_names = config.dataset.classes
    model = build_vision_model(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Standard Metrics Evaluation
    evaluator = Evaluator(
        model=model,
        device=config.system.device,
        class_names=class_names,
        logger=logger,
    )
    metrics = evaluator.evaluate(test_loader)

    # 2. Detailed Error & Confidence Analysis
    analyzer = DiagnosticErrorAnalyzer(
        model=model,
        class_names=class_names,
        device=config.system.device,
        logger=logger,
    )
    analysis = analyzer.run_comprehensive_analysis(test_loader)
    logger.info("Evaluation and Error Analysis complete.")


if __name__ == "__main__":
    main()
