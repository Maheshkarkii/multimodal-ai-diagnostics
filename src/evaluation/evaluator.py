"""
Model Evaluation and Diagnostic Performance Reporting Module.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.utils.device import resolve_device
from src.utils.logging import setup_logger


class Evaluator:
    """Evaluates vision diagnostic model performance on isolated test splits."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        class_names: Optional[List[str]] = None,
        logger: Optional[Any] = None,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.class_names = class_names or [str(i) for i in range(10)]
        self.logger = logger or setup_logger("Evaluator")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test DataLoader and compute comprehensive multiclass metrics.

        Returns dictionary with:
        - accuracy
        - precision (macro & weighted)
        - recall (macro & weighted)
        - f1_score (macro & weighted)
        - confusion_matrix
        - classification_report
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for images, targets in test_loader:
            images = images.to(self.device, non_blocking=True)
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        report_str = classification_report(
            y_true, y_pred, target_names=self.class_names, zero_division=0
        )

        results = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "precision_weighted": prec_weighted,
            "recall_macro": rec_macro,
            "recall_weighted": rec_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": report_str,
        }

        self.logger.info("Test Evaluation Results:")
        self.logger.info("Accuracy:           %.4f", acc)
        self.logger.info("F1 Score (Macro):   %.4f", f1_macro)
        self.logger.info("F1 Score (Weighted):%.4f", f1_weighted)
        self.logger.info("\n%s", report_str)

        return results
