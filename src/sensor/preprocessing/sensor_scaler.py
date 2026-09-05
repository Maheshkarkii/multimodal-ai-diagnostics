"""
Sensor telemetry preprocessing, missing-value imputation, and leakage-safe standard scaling.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SensorPreprocessor:
    """
    Leakage-safe Preprocessor for multivariate sensor telemetry.

    Ensures that statistics (imputation medians and scaling mean/variance) are fitted
    EXCLUSIVELY on the training partition and transformed onto validation, test, and inference queries.
    """

    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.impute_values: dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "SensorPreprocessor":
        """Fit imputation medians and scaling parameters on training split only."""
        # 1. Compute robust medians for missing value imputation
        self.impute_values = {
            col: float(df_train[col].median()) if col in df_train else 0.0 for col in self.feature_columns
        }

        # 2. Impute and fit standard scaler
        X_imputed = df_train[self.feature_columns].fillna(self.impute_values).to_numpy()
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe or dict into standardized feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("SensorPreprocessor must be fitted on training data before calling transform().")

        # Validate columns
        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

        X_imputed = df[self.feature_columns].fillna(self.impute_values).to_numpy(dtype=np.float32)
        X_scaled = self.scaler.transform(X_imputed).astype(np.float32)
        return X_scaled

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)

    def to_dict(self) -> dict[str, Any]:
        """Serialize preprocessor state for checkpoint preservation."""
        return {
            "feature_columns": self.feature_columns,
            "impute_values": self.impute_values,
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else [],
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else [],
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SensorPreprocessor":
        """Reconstruct preprocessor state from serialized parameters."""
        instance = cls(feature_columns=d["feature_columns"])
        instance.impute_values = d.get("impute_values", {})
        instance.is_fitted = d.get("is_fitted", False)
        if instance.is_fitted and d.get("scaler_mean"):
            instance.scaler.mean_ = np.array(d["scaler_mean"], dtype=np.float64)
            instance.scaler.scale_ = np.array(d["scaler_scale"], dtype=np.float64)
            instance.scaler.var_ = instance.scaler.scale_**2
        return instance
