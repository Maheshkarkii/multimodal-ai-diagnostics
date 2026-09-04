"""
Confidence Calibration and Reliability Curve Engine.
Computes Expected Calibration Error (ECE), Brier Score, and Confidence vs Accuracy Bins.
"""

from typing import List, Tuple
import numpy as np
from src.evaluation.schemas import CalibrationMetrics


def compute_calibration_metrics(
    y_true: List[int],
    y_prob_max: List[float],
    y_pred: List[int],
    num_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute Expected Calibration Error (ECE) and Brier score.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(y_true) == 0:
        return CalibrationMetrics()

    confidences = np.array(y_prob_max)
    accuracies = np.array(y_true) == np.array(y_pred)
    N = len(y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_acc_in_bin = float(np.mean(accuracies[in_bin]))
            avg_conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(avg_acc_in_bin - avg_conf_in_bin) * prop_in_bin
            bin_confidences.append(round(avg_conf_in_bin, 4))
            bin_accuracies.append(round(avg_acc_in_bin, 4))
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    # Brier Score: Mean squared difference between predicted probability and actual binary outcome (1 or 0)
    # Binary outcome is 1 if prediction was correct, 0 if incorrect
    brier_score = float(np.mean((confidences - accuracies.astype(float)) ** 2))
    overall_gap = float(np.abs(np.mean(confidences) - np.mean(accuracies)))

    return CalibrationMetrics(
        expected_calibration_error=round(float(ece), 4),
        brier_score=round(brier_score, 4),
        confidence_vs_accuracy_gap=round(overall_gap, 4),
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
    )
