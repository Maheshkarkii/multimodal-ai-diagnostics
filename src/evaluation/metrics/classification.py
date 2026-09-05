"""
Classification, Discrimination, and Multi-Class Evaluation Metrics.
Computes Accuracy, Macro/Weighted F1, Recall, Precision, and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.evaluation.schemas import ModalityMetrics


def calculate_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    y_prob: np.ndarray | None = None,
) -> ModalityMetrics:
    """
    Compute full multi-class classification and discrimination metrics.
    """
    if len(y_true) == 0:
        return ModalityMetrics(sample_count=0)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # Per-class F1
    _, _, per_class_f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = {lbl: float(score) for lbl, score in zip(labels, per_class_f1_arr)}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # ROC AUC if probabilities available and >1 class
    roc_auc_val = None
    if y_prob is not None and len(labels) > 1:
        try:
            # Map string labels to integer indices
            lbl_to_idx = {lbl: i for i, lbl in enumerate(labels)}
            y_true_idx = np.array([lbl_to_idx[y] for y in y_true if y in lbl_to_idx])
            if len(np.unique(y_true_idx)) > 1:
                roc_auc_val = float(roc_auc_score(y_true_idx, y_prob, multi_class="ovr"))
        except Exception:
            roc_auc_val = None

    return ModalityMetrics(
        accuracy=round(acc, 4),
        precision_macro=round(float(p_macro), 4),
        recall_macro=round(float(r_macro), 4),
        f1_macro=round(float(f1_macro), 4),
        f1_weighted=round(float(f1_weighted), 4),
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
        sample_count=len(y_true),
    )
