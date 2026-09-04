"""
Sensor Anomaly Detection Evaluator.
Measures Precision, Recall, F1, False Alarm Rate, and Missed Anomaly Rate.
"""

from typing import List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.evaluation.schemas import AnomalyMetrics


def evaluate_anomaly_detector(
    y_true_anomaly: List[bool],
    y_pred_anomaly: List[bool],
    threshold_applied: float = 0.5,
) -> AnomalyMetrics:
    """
    Evaluates binary anomaly detection performance.
    """
    if len(y_true_anomaly) == 0:
        return AnomalyMetrics()

    y_t = np.array(y_true_anomaly, dtype=bool)
    y_p = np.array(y_pred_anomaly, dtype=bool)

    # Anomaly is positive class (1 / True)
    p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
    
    # False Alarm Rate: False Positives / Actual Normals
    actual_normals = np.sum(~y_t)
    false_positives = np.sum((~y_t) & y_p)
    false_alarm_rate = float(false_positives / actual_normals) if actual_normals > 0 else 0.0

    # Missed Anomaly Rate: False Negatives / Actual Anomalies
    actual_anomalies = np.sum(y_t)
    false_negatives = np.sum(y_t & (~y_p))
    missed_rate = float(false_negatives / actual_anomalies) if actual_anomalies > 0 else 0.0

    return AnomalyMetrics(
        precision=round(float(p), 4),
        recall=round(float(r), 4),
        f1=round(float(f1), 4),
        false_alarm_rate=round(false_alarm_rate, 4),
        missed_anomaly_rate=round(missed_rate, 4),
        threshold_applied=threshold_applied,
        normal_count=int(actual_normals),
        anomalous_count=int(actual_anomalies),
    )
