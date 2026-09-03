"""
Multimodal Diagnostic Inference Engine with Modality Attribution and Unified Representation Extraction.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.vision.model import build_vision_model
from src.audio.models.audio_cnn import build_audio_model
from src.sensor.models.sensor_mlp import build_sensor_model
from src.multimodal.text.text_encoder import build_text_encoder
from src.multimodal.models.fusion_model import build_multimodal_model
from src.preprocessing.transforms import get_industrial_eval_transforms
from src.audio.preprocessing.audio_transforms import AudioPreprocessor
from src.sensor.preprocessing.sensor_scaler import SensorPreprocessor
from src.utils.device import resolve_device


class MultimodalPredictor:
    """
    End-to-end multimodal inference engine.

    Accepts any combination of available evidence:
    - image: Image file path, PIL Image, or precomputed embedding
    - audio: WAV file path or precomputed embedding
    - sensor_data: Dictionary of telemetry or precomputed embedding
    - technician_notes: String or precomputed embedding

    Returns unified diagnostic prediction, confidence score, available modalities,
    modality attribution, and unified machine embedding.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        fusion_model: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        device: str = "auto",
    ):
        self.device = resolve_device(device)
        self.class_names = class_names or [
            "normal_state",
            "bearing_defect_wear",
            "structural_crack_loose",
            "rotor_unbalance",
            "hydraulic_cavitation",
        ]

        # 1. Encoders and Preprocessors
        self.vision_encoder = build_vision_model(num_classes=5, pretrained=False).to(self.device).eval()
        self.audio_encoder = build_audio_model(num_classes=5, in_channels=1, embedding_dim=512).to(self.device).eval()
        self.sensor_encoder = build_sensor_model(in_features=6, num_classes=5, embedding_dim=256).to(self.device).eval()
        self.text_encoder = build_text_encoder(embedding_dim=256).to(self.device).eval()

        self.vision_tf = get_industrial_eval_transforms(image_size=224)
        self.audio_prep = AudioPreprocessor(sample_rate=16000, duration=3.0)
        self.sensor_prep = SensorPreprocessor(feature_columns=[
            "temperature_c", "vibration_rms_g", "rotational_speed_rpm", "motor_current_a", "hydraulic_pressure_bar", "load_percentage"
        ])

        # 2. Fusion Model
        if fusion_model is not None:
            self.fusion_model = fusion_model.to(self.device).eval()
        elif checkpoint_path is not None:
            self.fusion_model = self._load_fusion_model(Path(checkpoint_path))
        else:
            self.fusion_model = build_multimodal_model(num_classes=len(self.class_names)).to(self.device).eval()

    def _load_fusion_model(self, checkpoint_path: Path) -> nn.Module:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names = ckpt.get("class_names", self.class_names)
        model = build_multimodal_model(num_classes=len(self.class_names))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        image: Optional[Union[str, Path, Image.Image, np.ndarray, torch.Tensor]] = None,
        audio: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        sensor_data: Optional[Union[Dict[str, float], pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        technician_notes: Optional[Union[str, torch.Tensor]] = None,
        top_k: int = 3,
        return_unified_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute unified multimodal diagnostic inference.
        """
        # Validate that at least one modality is provided
        if image is None and audio is None and sensor_data is None and technician_notes is None:
            raise ValueError("Invalid input: At least one modality (image, audio, sensor, or text) must be provided.")

        active_modalities = []
        embs = {}
        masks = {}

        # 1. Vision
        if image is not None:
            active_modalities.append("vision")
            masks["vision"] = torch.tensor([[1]], dtype=torch.long, device=self.device)
            if isinstance(image, torch.Tensor) and image.shape[-1] == 1280:
                embs["vision"] = image.to(self.device)
            else:
                pil_img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image
                v_t = self.vision_tf(pil_img).unsqueeze(0).to(self.device)
                embs["vision"] = self.vision_encoder.extract_features(v_t)
        else:
            masks["vision"] = torch.tensor([[0]], dtype=torch.long, device=self.device)
            embs["vision"] = torch.zeros(1, 1280, device=self.device)

        # 2. Audio
        if audio is not None:
            active_modalities.append("audio")
            masks["audio"] = torch.tensor([[1]], dtype=torch.long, device=self.device)
            if isinstance(audio, torch.Tensor) and audio.shape[-1] == 512:
                embs["audio"] = audio.to(self.device)
            else:
                spec = self.audio_prep.process(audio).unsqueeze(0).to(self.device)
                embs["audio"] = self.audio_encoder.extract_features(spec)
        else:
            masks["audio"] = torch.tensor([[0]], dtype=torch.long, device=self.device)
            embs["audio"] = torch.zeros(1, 512, device=self.device)

        # 3. Sensor
        if sensor_data is not None:
            active_modalities.append("sensor")
            masks["sensor"] = torch.tensor([[1]], dtype=torch.long, device=self.device)
            if isinstance(sensor_data, torch.Tensor) and sensor_data.shape[-1] == 256:
                embs["sensor"] = sensor_data.to(self.device)
            else:
                df = pd.DataFrame([sensor_data]) if isinstance(sensor_data, dict) else sensor_data
                s_scaled = self.sensor_prep.transform(df) if self.sensor_prep.is_fitted else df.to_numpy(dtype=np.float32)
                s_t = torch.tensor(s_scaled, dtype=torch.float32).to(self.device)
                embs["sensor"] = self.sensor_encoder.extract_features(s_t)
        else:
            masks["sensor"] = torch.tensor([[0]], dtype=torch.long, device=self.device)
            embs["sensor"] = torch.zeros(1, 256, device=self.device)

        # 4. Text
        if technician_notes is not None:
            active_modalities.append("text")
            masks["text"] = torch.tensor([[1]], dtype=torch.long, device=self.device)
            if isinstance(technician_notes, torch.Tensor) and technician_notes.shape[-1] == 256:
                embs["text"] = technician_notes.to(self.device)
            else:
                embs["text"] = self.text_encoder.encode([technician_notes]).to(self.device)
        else:
            masks["text"] = torch.tensor([[0]], dtype=torch.long, device=self.device)
            embs["text"] = torch.zeros(1, 256, device=self.device)

        # 5. Fusion Forward Pass
        logits, unified_emb = self.fusion_model(embs, masks=masks, return_features=True)
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
            "predicted_machine_condition": top1_class,
            "confidence": top1_conf,
            "available_modalities": active_modalities,
            "top_candidates": topk_list,
        }

        if return_unified_embedding:
            emb_arr = unified_emb.squeeze(0).cpu().numpy().tolist()
            result["unified_embedding_dim"] = len(emb_arr)
            result["unified_machine_embedding"] = emb_arr

        return result
