"""
Sensor Telemetry Dataset with Preprocessing Leakage Prevention and Feature Grouping.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.sensor.preprocessing.sensor_scaler import SensorPreprocessor
from src.data.dataset import split_samples_group_aware, compute_class_weights
from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger

logger = setup_logger("SensorData")


class SensorTelemetryDataset(Dataset):
    """
    PyTorch Dataset wrapping scaled numerical sensor features and target class indices.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.X[index], self.y[index].item()


class SensorDataValidator:
    """Validates raw sensor telemetry CSV schemas, checks missing counts, and verifies machine IDs."""

    @staticmethod
    def validate_and_summarize(
        df: pd.DataFrame, feature_cols: List[str], target_col: str, group_col: Optional[str] = None
    ) -> Dict[str, Any]:
        missing_counts = df[feature_cols].isnull().sum().to_dict()
        feature_stats = {}
        for col in feature_cols:
            feature_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "missing": int(missing_counts[col]),
            }

        class_dist = df[target_col].value_counts().to_dict() if target_col in df.columns else {}
        machines = list(df[group_col].unique()) if group_col and group_col in df.columns else []

        return {
            "total_rows": len(df),
            "feature_statistics": feature_stats,
            "class_distribution": class_dist,
            "unique_machines": len(machines),
            "machine_ids": machines,
        }


def prepare_sensor_splits_and_loaders(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    classes: List[str],
    group_col: Optional[str] = "machine_id",
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, SensorPreprocessor, Dict[str, int], torch.Tensor, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Leakage-safe end-to-end splitting, fitting, and DataLoader generation for sensor telemetry.

    Flow:
    1. Group-aware splitting into Train, Val, and Test DataFrames.
    2. Fit `SensorPreprocessor` exclusively on Train DataFrame.
    3. Transform Val and Test using Train parameters.
    4. Compute inverse class weights on Train labels.
    """
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    # 1. Convert DataFrame rows into sample records for group-aware splitting
    samples = df.to_dict(orient="records")
    for s in samples:
        s["label"] = s[target_col]

    train_records, val_records, test_records = split_samples_group_aware(
        samples=samples,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        group_key=group_col,
    )

    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    # 2. Strict Preprocessing Isolation: Fit scaler ONLY on train
    preprocessor = SensorPreprocessor(feature_columns=feature_cols)
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    y_train = np.array([class_to_idx[r[target_col]] for r in train_records], dtype=np.int64)
    y_val = np.array([class_to_idx[r[target_col]] for r in val_records], dtype=np.int64)
    y_test = np.array([class_to_idx[r[target_col]] for r in test_records], dtype=np.int64)

    # 3. Class Weights
    class_weights = compute_class_weights(train_records, class_to_idx)

    # 4. PyTorch DataLoaders
    train_loader = DataLoader(
        SensorTelemetryDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        SensorTelemetryDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        SensorTelemetryDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        preprocessor,
        class_to_idx,
        class_weights,
        (X_train, X_val, X_test),
    )
