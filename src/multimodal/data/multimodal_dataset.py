"""
Aligned Multimodal Dataset with Synchronized Modality Cache & Mask Handling.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import split_samples_group_aware, compute_class_weights
from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger

logger = setup_logger("MultimodalDataset")


class AlignedMultimodalDataset(Dataset):
    """
    Synchronized Multimodal PyTorch Dataset.

    Returns:
    - Dict of modality tensors: {'vision': Tensor, 'audio': Tensor, 'sensor': Tensor, 'text': Tensor}
    - Dict of presence masks: {'vision': 1/0, 'audio': 1/0, 'sensor': 1/0, 'text': 1/0}
    - Numerical class index
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        class_to_idx: Dict[str, int],
        vision_embeddings: np.ndarray,
        audio_embeddings: np.ndarray,
        sensor_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        masks: Dict[str, np.ndarray],
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.v_emb = torch.tensor(vision_embeddings, dtype=torch.float32)
        self.a_emb = torch.tensor(audio_embeddings, dtype=torch.float32)
        self.s_emb = torch.tensor(sensor_embeddings, dtype=torch.float32)
        self.t_emb = torch.tensor(text_embeddings, dtype=torch.float32)

        self.masks = {
            m: torch.tensor(masks[m], dtype=torch.long).unsqueeze(1)
            for m in ["vision", "audio", "sensor", "text"]
        }

        self.labels = torch.tensor(
            [class_to_idx[s["label"]] for s in samples], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        embs = {
            "vision": self.v_emb[index],
            "audio": self.a_emb[index],
            "sensor": self.s_emb[index],
            "text": self.t_emb[index],
        }
        masks = {
            "vision": self.masks["vision"][index],
            "audio": self.masks["audio"][index],
            "sensor": self.masks["sensor"][index],
            "text": self.masks["text"][index],
        }
        return embs, masks, self.labels[index].item()


def custom_multimodal_collate(batch):
    """Custom collate to properly batch dictionaries of modality tensors and masks."""
    v_list, a_list, s_list, t_list = [], [], [], []
    mv_list, ma_list, ms_list, mt_list = [], [], [], []
    targets = []

    for embs, masks, target in batch:
        v_list.append(embs["vision"])
        a_list.append(embs["audio"])
        s_list.append(embs["sensor"])
        t_list.append(embs["text"])

        mv_list.append(masks["vision"])
        ma_list.append(masks["audio"])
        ms_list.append(masks["sensor"])
        mt_list.append(masks["text"])

        targets.append(target)

    batch_embs = {
        "vision": torch.stack(v_list, dim=0),
        "audio": torch.stack(a_list, dim=0),
        "sensor": torch.stack(s_list, dim=0),
        "text": torch.stack(t_list, dim=0),
    }

    batch_masks = {
        "vision": torch.stack(mv_list, dim=0),
        "audio": torch.stack(ma_list, dim=0),
        "sensor": torch.stack(ms_list, dim=0),
        "text": torch.stack(mt_list, dim=0),
    }

    return batch_embs, batch_masks, torch.tensor(targets, dtype=torch.long)
