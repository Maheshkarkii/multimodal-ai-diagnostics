"""
Diagnostic Error Analysis and Confidence Calibration Module.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.device import resolve_device
from src.utils.logging import setup_logger


class DiagnosticErrorAnalyzer:
    """
    Analyzes model diagnostic failures, confusion patterns, low-confidence predictions,
    and statistical confidence distributions.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = "auto",
        logger: Optional[Any] = None,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.class_names = class_names
        self.logger = logger or setup_logger("ErrorAnalyzer")

    @torch.no_grad()
    def run_comprehensive_analysis(
        self, test_loader: DataLoader, confidence_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Execute detailed error and confidence analysis across test dataset.

        Identifies:
        - False Positives & False Negatives per class
        - Top Confused Class Pairs
        - Low-confidence predictions (where model is uncertain)
        - Confidence calibration stats (mean confidence on correct vs incorrect predictions)
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_confidences = []
        misclassifications = []

        for images, targets in test_loader:
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            probabilities = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probabilities, dim=1)

            for i in range(images.size(0)):
                pred_idx = int(preds[i].item())
                true_idx = int(targets[i].item())
                conf_val = float(confs[i].item())

                all_preds.append(pred_idx)
                all_targets.append(true_idx)
                all_confidences.append(conf_val)

                if pred_idx != true_idx:
                    misclassifications.append({
                        "true_class": self.class_names[true_idx],
                        "predicted_class": self.class_names[pred_idx],
                        "confidence": conf_val,
                        "is_low_confidence": conf_val < confidence_threshold,
                    })

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        confidences = np.array(all_confidences)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))

        # 2. Confusion Pairs
        confusion_pairs = {}
        for item in misclassifications:
            pair = f"{item['true_class']} -> {item['predicted_class']}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

        # 3. Confidence Stats
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        mean_conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
        mean_conf_incorrect = float(np.mean(confidences[incorrect_mask])) if np.any(incorrect_mask) else 0.0

        low_conf_predictions = [
            m for m in misclassifications if m["is_low_confidence"]
        ]

        report = {
            "total_evaluated": len(y_true),
            "total_errors": len(misclassifications),
            "error_rate": float(len(misclassifications) / max(len(y_true), 1)),
            "mean_confidence_when_correct": mean_conf_correct,
            "mean_confidence_when_incorrect": mean_conf_incorrect,
            "top_confusion_pairs": sorted_confusions[:5],
            "low_confidence_error_count": len(low_conf_predictions),
            "confusion_matrix": cm.tolist(),
        }

        self.logger.info("=== Diagnostic Error Analysis ===")
        self.logger.info("Total Evaluated: %d | Total Errors: %d (Error Rate: %.2f%%)",
                         report["total_evaluated"], report["total_errors"], report["error_rate"] * 100)
        self.logger.info("Confidence Calibration: Correct Mean=%.3f, Incorrect Mean=%.3f",
                         mean_conf_correct, mean_conf_incorrect)
        self.logger.info("Top Failure Confusions: %s", sorted_confusions[:3])

        return report
