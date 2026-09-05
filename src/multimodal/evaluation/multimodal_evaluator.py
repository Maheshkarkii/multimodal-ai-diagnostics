"""
Multimodal Evaluation & Modality-Ablation Benchmark Framework.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.utils.device import resolve_device
from src.utils.logging import setup_logger


class MultimodalEvaluator:
    """Evaluates Multimodal Fusion models and benchmarks unimodal & ablated combinations."""

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        device: str = "auto",
        logger: Any | None = None,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.class_names = class_names
        self.logger = logger or setup_logger("MultimodalEval")

    @torch.no_grad()
    def evaluate_combination(
        self,
        test_loader: DataLoader,
        active_modalities: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate performance when only a subset of modalities are supplied (missing-modality test).
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for embs, masks, targets in test_loader:
            embs = {k: v.to(self.device) for k, v in embs.items()}

            # Override masks if evaluating explicit combination
            if active_modalities is not None:
                custom_masks = {
                    m: torch.ones_like(masks[m]).to(self.device)
                    if m in active_modalities
                    else torch.zeros_like(masks[m]).to(self.device)
                    for m in ["vision", "audio", "sensor", "text"]
                }
            else:
                custom_masks = {k: v.to(self.device) for k, v in masks.items()}

            outputs = self.model(embs, masks=custom_masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def run_full_ablation_study(self, test_loader: DataLoader) -> dict[str, dict[str, float]]:
        """
        Benchmark all combinations:
        - Unimodal: Vision only, Audio only, Sensor only, Text only
        - Bimodal: Vision+Audio, Vision+Sensor, Audio+Sensor
        - Trimodal: Vision+Audio+Sensor (Physical signals)
        - Full Quad-Modal: Vision+Audio+Sensor+Text
        """
        combinations = {
            "Vision Only": ["vision"],
            "Audio Only": ["audio"],
            "Sensor Only": ["sensor"],
            "Text Only": ["text"],
            "Vision + Audio": ["vision", "audio"],
            "Vision + Sensor": ["vision", "sensor"],
            "Audio + Sensor": ["audio", "sensor"],
            "Vision + Audio + Sensor": ["vision", "audio", "sensor"],
            "All Modalities (Vision+Audio+Sensor+Text)": ["vision", "audio", "sensor", "text"],
        }

        results = {}
        self.logger.info("=== Multimodal Ablation & Modality Contribution Benchmark ===")
        for name, active_mods in combinations.items():
            metrics = self.evaluate_combination(test_loader, active_modalities=active_mods)
            results[name] = metrics
            self.logger.info(
                "%-42s | Accuracy: %6.2f%% | Macro F1: %6.4f | Weighted F1: %6.4f",
                name,
                metrics["accuracy"] * 100,
                metrics["macro_f1"],
                metrics["weighted_f1"],
            )

        return results
