"""
Sensor Telemetry Evaluation and Error Analysis CLI.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.sensor.data.sensor_dataset import prepare_sensor_splits_and_loaders
from src.sensor.data.generate_sample_telemetry import generate_synthetic_telemetry_dataset
from src.sensor.models.sensor_mlp import build_sensor_model
from src.evaluation.evaluator import Evaluator
from src.analysis.error_analysis import DiagnosticErrorAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sensor Telemetry Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sensor.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sensor_state_and_anomaly_baseline_best.pt",
        help="Path to trained model checkpoint .pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("SensorEvaluation", level=config.system.log_level)

    csv_path = Path(config.dataset.dataset_dir)
    if not csv_path.exists():
        generate_synthetic_telemetry_dataset(csv_path, num_machines=8, records_per_machine=100, seed=config.system.seed)

    df = pd.read_csv(csv_path)
    feature_cols = getattr(config.dataset, "feature_columns", [
        "temperature_c", "vibration_rms_g", "rotational_speed_rpm", "motor_current_a", "hydraulic_pressure_bar", "load_percentage"
    ])
    target_col = getattr(config.dataset, "target_column", "fault_label")
    group_col = getattr(config.dataset, "group_by", "machine_id")

    _, _, test_loader, _, class_to_idx, _, _ = prepare_sensor_splits_and_loaders(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        classes=config.dataset.classes,
        group_col=group_col,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        batch_size=config.training.batch_size,
        seed=config.system.seed,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Please train first.", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", config.dataset.classes)

    model = build_sensor_model(
        in_features=len(feature_cols),
        num_classes=len(class_names),
        embedding_dim=256,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Evaluate Metrics
    evaluator = Evaluator(
        model=model,
        device=config.system.device,
        class_names=class_names,
        logger=logger,
    )
    metrics = evaluator.evaluate(test_loader)

    # 2. Error Analysis
    analyzer = DiagnosticErrorAnalyzer(
        model=model,
        class_names=class_names,
        device=config.system.device,
        logger=logger,
    )
    analysis = analyzer.run_comprehensive_analysis(test_loader)
    logger.info("Sensor evaluation and state analysis complete.")


if __name__ == "__main__":
    main()
