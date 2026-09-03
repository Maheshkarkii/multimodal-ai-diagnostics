"""
Acoustic Inference Engine with 512-dim Feature Embedding Extraction.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
import torch.nn as nn

from src.audio.preprocessing.audio_transforms import AudioPreprocessor
from src.audio.models.audio_cnn import build_audio_model
from src.utils.device import resolve_device


DEFAULT_AUDIO_CLASSES = [
    "normal_operation",
    "bearing_defect",
    "loose_component",
    "rotor_imbalance",
    "cavitation_anomaly",
]


class AudioPredictor:
    """Production acoustic inference engine for machine sound diagnostics."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        sample_rate: int = 16000,
        duration: float = 3.0,
        device: str = "auto",
    ):
        self.device = resolve_device(device)
        self.class_names = class_names or DEFAULT_AUDIO_CLASSES
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_model_from_checkpoint(Path(checkpoint_path))
        else:
            raise ValueError("Either 'checkpoint_path' or 'model' must be provided.")

        self.model.eval()

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = build_audio_model(
            num_classes=len(self.class_names),
            in_channels=1,
            embedding_dim=512,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        return model

    @torch.no_grad()
    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor],
        src_sr: Optional[int] = None,
        top_k: int = 3,
        return_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on a single audio recording (.wav).

        Returns:
            Dictionary with predicted fault, confidence, top-k candidates, and 512-dim embedding.
        """
        # Transform into standardized Log-Mel Spectrogram (1, 1, n_mels, T)
        spec = self.preprocessor.process(audio_input, src_sr=src_sr)
        spec = spec.unsqueeze(0).to(self.device)

        if return_embedding:
            logits, embeddings = self.model(spec, return_features=True)
            emb_vector = embeddings.squeeze(0).cpu().numpy().tolist()
        else:
            logits = self.model(spec)
            emb_vector = None

        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probabilities, dim=0)

        top1_idx = int(pred_idx.item())
        top1_conf = float(confidence.item())
        top1_class = self.class_names[top1_idx]

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
            "predicted_sound": top1_class,
            "predicted_class": top1_class,
            "predicted_index": top1_idx,
            "confidence": top1_conf,
            "top_candidates": topk_list,
            "top_k": topk_list,
        }

        if return_embedding:
            result["embedding_dim"] = len(emb_vector)
            result["acoustic_embedding"] = emb_vector

        return result
