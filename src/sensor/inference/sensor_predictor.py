"""
Sensor Telemetry Inference Engine with Anomaly Scoring and 256-dim Feature Embedding Extraction.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.sensor.models.anomaly_detector import SensorAnomalyDetector
from src.sensor.models.sensor_mlp import build_sensor_model
from src.sensor.preprocessing.sensor_scaler import SensorPreprocessor
from src.utils.device import resolve_device


class SensorPredictor:
    """Production inference engine for multivariate machine sensor telemetry."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: nn.Module | None = None,
        preprocessor: SensorPreprocessor | None = None,
        anomaly_detector: SensorAnomalyDetector | None = None,
        class_names: list[str] | None = None,
        device: str = "auto",
    ):
        self.device = resolve_device(device)
        self.model = model
        self.preprocessor = preprocessor
        self.anomaly_detector = anomaly_detector
        self.class_names = class_names

        if checkpoint_path is not None:
            self._load_from_checkpoint(Path(checkpoint_path))

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _load_from_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Sensor checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.class_names = ckpt.get("class_names", [f"class_{i}" for i in range(5)])
        self.preprocessor = SensorPreprocessor.from_dict(ckpt["preprocessor"])

        if "anomaly_detector" in ckpt and ckpt["anomaly_detector"] is not None:
            self.anomaly_detector = ckpt["anomaly_detector"]

        in_features = len(self.preprocessor.feature_columns)
        self.model = build_sensor_model(
            in_features=in_features,
            num_classes=len(self.class_names),
            embedding_dim=256,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        sensor_input: dict[str, float] | pd.DataFrame | np.ndarray,
        top_k: int = 3,
        return_embedding: bool = False,
    ) -> dict[str, Any]:
        """
        Run inference on structured sensor measurements.

        Args:
            sensor_input: Dictionary of {feature_name: value} or DataFrame row.
            top_k: Top candidate machine state conditions to return.
            return_embedding: If True, returns 256-dim embedding vector for multimodal fusion.

        Returns:
            Structured diagnostic & anomaly assessment.
        """
        # 1. Format input to DataFrame
        if isinstance(sensor_input, dict):
            df = pd.DataFrame([sensor_input])
        elif isinstance(sensor_input, pd.DataFrame):
            df = sensor_input
        elif isinstance(sensor_input, np.ndarray):
            df = pd.DataFrame(sensor_input, columns=self.preprocessor.feature_columns)
        else:
            raise ValueError(f"Unsupported sensor input type: {type(sensor_input)}")

        # 2. Preprocess / Scale using fitted parameters
        X_scaled = self.preprocessor.transform(df)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # 3. Model Forward Pass
        if return_embedding:
            logits, embeddings = self.model(x_tensor, return_features=True)
            emb_vector = embeddings.squeeze(0).cpu().numpy().tolist()
        else:
            logits = self.model(x_tensor)
            emb_vector = None

        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probabilities, dim=0)

        top1_idx = int(pred_idx.item())
        top1_conf = float(confidence.item())
        top1_class = self.class_names[top1_idx]

        # Top-K
        k = min(top_k, len(self.class_names))
        topk_probs, topk_indices = torch.topk(probabilities, k)
        topk_list = [
            {
                "rank": i + 1,
                "class_index": int(topk_indices[i].item()),
                "class_name": self.class_names[topk_indices[i].item()],
                "confidence": float(topk_probs[i].item()),
            }
            for i in range(k)
        ]

        result = {
            "predicted_machine_state": top1_class,
            "confidence": top1_conf,
            "top_candidates": topk_list,
        }

        # 4. Anomaly Evaluation
        if self.anomaly_detector is not None and self.anomaly_detector.is_fitted:
            anomaly_eval = self.anomaly_detector.evaluate_sample(X_scaled[0])
            result["anomaly_assessment"] = anomaly_eval

        # 5. Multimodal Embedding
        if return_embedding:
            result["embedding_dim"] = len(emb_vector)
            result["sensor_embedding"] = emb_vector

        return result
