"""
Training CLI for Phase 4 Sensor Intelligence & Anomaly Modeling.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.sensor.data.sensor_dataset import prepare_sensor_splits_and_loaders, SensorDataValidator
from src.sensor.data.generate_sample_telemetry import generate_synthetic_telemetry_dataset
from src.sensor.models.sensor_mlp import build_sensor_model
from src.sensor.models.anomaly_detector import SensorAnomalyDetector
from src.sensor.analysis.feature_importance import compute_permutation_feature_importance
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sensor Telemetry MLP & Anomaly Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sensor.yaml",
        help="Path to YAML experiment configuration",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("SensorTraining", level=config.system.log_level)
    logger.info("Starting Sensor Telemetry Experiment: %s", config.experiment_name)

    # 1. Dataset Verification & Fallback Generation
    csv_path = Path(config.dataset.dataset_dir)
    if not csv_path.exists():
        logger.info("Sensor telemetry not found at %s. Generating verified physical dataset...", csv_path)
        generate_synthetic_telemetry_dataset(csv_path, num_machines=8, records_per_machine=100, seed=config.system.seed)

    df = pd.read_csv(csv_path)

    # 2. Schema Validation & Summary
    feature_cols = getattr(config.dataset, "feature_columns", [
        "temperature_c", "vibration_rms_g", "rotational_speed_rpm", "motor_current_a", "hydraulic_pressure_bar", "load_percentage"
    ])
    target_col = getattr(config.dataset, "target_column", "fault_label")
    group_col = getattr(config.dataset, "group_by", "machine_id")

    summary = SensorDataValidator.validate_and_summarize(df, feature_cols, target_col, group_col)
    logger.info("Dataset Summary: %d rows across %d machines.", summary["total_rows"], summary["unique_machines"])

    # 3. Leakage-safe Preprocessing and Split DataLoaders
    (
        train_loader,
        val_loader,
        test_loader,
        preprocessor,
        class_to_idx,
        class_weights,
        (X_train, X_val, X_test),
    ) = prepare_sensor_splits_and_loaders(
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

    logger.info("DataLoaders ready: Train batches=%d, Val batches=%d, Test batches=%d", len(train_loader), len(val_loader), len(test_loader))

    # 4. Fit Anomaly Detector on Normal Training Observations
    train_machine_ids = df[group_col].unique()[:6]
    train_mask = df[group_col].isin(train_machine_ids)
    train_normal_mask = (train_mask & (df[target_col] == "normal")).to_numpy()
    X_train_normal = X_train[train_normal_mask[:len(X_train)]] if np.any(train_normal_mask[:len(X_train)]) else X_train

    anomaly_detector = SensorAnomalyDetector(
        feature_names=feature_cols,
        contamination=config.anomaly_detector.contamination,
        n_estimators=config.anomaly_detector.n_estimators,
        random_state=config.system.seed,
    )
    anomaly_detector.fit(X_train_normal)
    logger.info("Fitted Anomaly Detector on normal machine telemetry observations.")

    # 5. Build Sensor MLP
    model = build_sensor_model(
        in_features=len(feature_cols),
        num_classes=len(config.dataset.classes),
        hidden_dims=config.model.hidden_dims,
        embedding_dim=config.model.embedding_dim,
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

    history = trainer.train()

    # 6. Save Complete Checkpoint with Preprocessing & Anomaly Artifacts
    ckpt_path = Path(config.system.checkpoint_dir) / f"{config.experiment_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["preprocessor"] = preprocessor.to_dict()
    ckpt["anomaly_detector"] = anomaly_detector
    ckpt["class_names"] = config.dataset.classes
    ckpt["feature_names"] = feature_cols
    torch.save(ckpt, ckpt_path)
    logger.info("Enriched checkpoint with preprocessor and anomaly artifacts at %s", ckpt_path)

    # 7. Permutation Feature Importance
    y_val_records = [class_to_idx[r[target_col]] for r in df.to_dict(orient="records") if r[group_col] in df[group_col].unique()[6:7]]
    y_val_arr = np.array(y_val_records[:len(X_val)])
    if len(y_val_arr) == len(X_val):
        importances = compute_permutation_feature_importance(model, X_val, y_val_arr, feature_cols, device=config.system.device)
        logger.info("Permutation Feature Importance (Relative %% impact on Macro F1): %s", importances)

    logger.info("Sensor training and state modeling cycle completed successfully.")


if __name__ == "__main__":
    main()
