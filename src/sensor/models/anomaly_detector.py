"""
Sensor Telemetry Anomaly Detection & Operating Envelope Modeling.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


class OperatingEnvelopeDetector:
    """
    Computes statistical operational envelopes (regime-aware [min, max, mean, std]) on normal baseline data
    and scores deviations based on Mahalanobis/Z-score distance.
    """

    def __init__(self, feature_names: list[str], std_threshold: float = 3.0):
        self.feature_names = feature_names
        self.std_threshold = std_threshold
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}

    def fit(self, X_normal: np.ndarray) -> "OperatingEnvelopeDetector":
        """Compute normal bounds from baseline normal telemetry."""
        for i, name in enumerate(self.feature_names):
            col = X_normal[:, i]
            self.means[name] = float(np.mean(col))
            self.stds[name] = float(np.std(col)) + 1e-6
            self.mins[name] = float(np.min(col))
            self.maxs[name] = float(np.max(col))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute max normalized deviation across all sensor channels."""
        scores = []
        for row in X:
            row_devs = [abs(row[i] - self.means[name]) / self.stds[name] for i, name in enumerate(self.feature_names)]
            scores.append(max(row_devs))
        return np.array(scores, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomalous (outside envelope), 0 for normal."""
        scores = self.score_samples(X)
        return (scores > self.std_threshold).astype(np.int32)


class SensorAnomalyDetector:
    """
    Multivariate Anomaly Detection using Isolation Forest & Normal Envelopes.
    """

    def __init__(
        self,
        feature_names: list[str],
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.envelope = OperatingEnvelopeDetector(feature_names=feature_names)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "SensorAnomalyDetector":
        """Fit isolation forest and operational envelope on normal training observations."""
        self.iso_forest.fit(X_train)
        self.envelope.fit(X_train)
        self.is_fitted = True
        return self

    def evaluate_sample(self, x_vector: np.ndarray) -> dict[str, Any]:
        """
        Evaluate a single telemetry observation.

        Returns:
            Dictionary with anomaly flag (bool), continuous anomaly score [0.0..1.0], and envelope boundary violations.
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted prior to evaluation.")

        if x_vector.ndim == 1:
            x_mat = x_vector.reshape(1, -1)
        else:
            x_mat = x_vector

        # Raw decision function: lower means more anomalous
        raw_score = float(self.iso_forest.decision_function(x_mat)[0])
        # Normalized anomaly score [0.0 = completely normal, 1.0 = extreme anomaly]
        norm_anomaly_score = float(1.0 / (1.0 + np.exp(raw_score * 5.0)))
        is_anomaly = bool(self.iso_forest.predict(x_mat)[0] == -1)

        # Envelope check
        envelope_dev = float(self.envelope.score_samples(x_mat)[0])

        return {
            "is_anomalous": is_anomaly,
            "anomaly_score": norm_anomaly_score,
            "max_envelope_deviation_sigma": envelope_dev,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "envelope_means": self.envelope.means,
            "envelope_stds": self.envelope.stds,
            "is_fitted": self.is_fitted,
        }
