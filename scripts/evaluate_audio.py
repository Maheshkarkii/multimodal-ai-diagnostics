"""
Evaluation and Acoustic Error Analysis CLI.
"""

import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.audio.data.audio_dataset import create_audio_dataloaders
from src.audio.data.generate_sample_audio import generate_synthetic_acoustic_dataset
from src.audio.models.audio_cnn import build_audio_model
from src.evaluation.evaluator import Evaluator
from src.analysis.error_analysis import DiagnosticErrorAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Acoustic Diagnostic Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/audio.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/acoustic_fault_baseline_best.pt",
        help="Path to trained model checkpoint .pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("AudioEvaluation", level=config.system.log_level)

    data_dir = Path(config.dataset.dataset_dir)
    has_audio = any(data_dir.glob("*/*.wav")) if data_dir.exists() else False
    if not has_audio:
        generate_synthetic_acoustic_dataset(data_dir, samples_per_class=30, seed=config.system.seed)

    _, _, test_loader, class_to_idx, _ = create_audio_dataloaders(config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Please train first.", ckpt_path)
        sys.exit(1)

    class_names = config.dataset.classes
    model = build_audio_model(
        num_classes=len(class_names),
        in_channels=1,
        embedding_dim=512,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Standard Multiclass Metrics
    evaluator = Evaluator(
        model=model,
        device=config.system.device,
        class_names=class_names,
        logger=logger,
    )
    metrics = evaluator.evaluate(test_loader)

    # 2. Detailed Diagnostic Error & Confidence Analysis
    analyzer = DiagnosticErrorAnalyzer(
        model=model,
        class_names=class_names,
        device=config.system.device,
        logger=logger,
    )
    analysis = analyzer.run_comprehensive_analysis(test_loader)
    logger.info("Acoustic evaluation and error analysis complete.")


if __name__ == "__main__":
    main()
