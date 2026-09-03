"""
Vision Diagnostics Inference Predictor.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.preprocessing.transforms import get_eval_transforms
from src.vision.model import build_vision_model
from src.data.dataset import FASHION_MNIST_CLASSES
from src.utils.device import resolve_device


class VisionPredictor:
    """Production inference engine for single-image and batch diagnostic predictions."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        image_size: int = 224,
        device: str = "auto",
    ):
        self.device = resolve_device(device)
        self.class_names = class_names or FASHION_MNIST_CLASSES
        self.image_size = image_size
        self.transform = get_eval_transforms(image_size=self.image_size)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_model_from_checkpoint(Path(checkpoint_path))
        else:
            raise ValueError("Either 'checkpoint_path' or 'model' must be provided.")

        self.model.eval()

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model = build_vision_model(
            num_classes=len(self.class_names),
            pretrained=False,
            freeze_backbone=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        return model

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_input: Filepath, PIL Image, or NumPy Array.
            top_k: Number of top ranked predictions to return.

        Returns:
            Dictionary containing predicted class, index, confidence, and top-k rankings.
        """
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        tensor_img = self.transform(pil_image).unsqueeze(0).to(self.device)

        logits = self.model(tensor_img)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # Top-1
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

        return {
            "predicted_class": top1_class,
            "predicted_index": top1_idx,
            "confidence": top1_conf,
            "top_k": topk_list,
        }
