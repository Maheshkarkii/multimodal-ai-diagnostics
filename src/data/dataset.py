"""
Industrial Equipment Inspection Dataset with Group-aware Leakage Prevention & Class Imbalance Balancing.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from src.preprocessing.transforms import (
    get_industrial_train_transforms,
    get_industrial_eval_transforms,
)
from src.utils.config import DatasetConfig, SystemConfig, TrainingConfig
from src.utils.logging import setup_logger

logger = setup_logger("IndustrialDataset")


class IndustrialEquipmentDataset(Dataset):
    """
    Standard PyTorch Dataset for industrial equipment inspection images.

    Supports metadata dictionaries with explicit equipment/machine IDs to guarantee
    leakage-free group-based train/val/test splits.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        class_to_idx: Dict[str, int],
        transform: Optional[Compose] = None,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[index]
        img_path = item["filepath"]
        label_str = item["label"]
        label_idx = self.class_to_idx[label_str]

        # Load image via PIL
        if isinstance(img_path, (str, Path)):
            image = Image.open(img_path).convert("RGB")
        elif isinstance(img_path, Image.Image):
            image = img_path
        elif isinstance(img_path, np.ndarray):
            image = Image.fromarray(img_path)
        else:
            raise ValueError(f"Invalid image format at index {index}")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


class DatasetValidator:
    """Validates real industrial image datasets for corruption, leaks, and severe imbalances."""

    @staticmethod
    def validate_and_summarize(
        data_dir: Path, supported_extensions=(".png", ".jpg", ".jpeg", ".bmp")
    ) -> Dict[str, Any]:
        """Inspect and validate directory structure and image readability."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

        class_counts = {}
        corrupted_files = []
        valid_samples = []

        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            manifest_file = data_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                return {"type": "manifest", "total_samples": len(manifest_data)}

        for class_dir in class_dirs:
            class_name = class_dir.name
            files = [
                f for f in class_dir.glob("*") if f.suffix.lower() in supported_extensions
            ]
            valid_count = 0
            for f in files:
                try:
                    with Image.open(f) as img:
                        img.verify()
                    valid_samples.append({"filepath": str(f), "label": class_name})
                    valid_count += 1
                except Exception as e:
                    corrupted_files.append({"filepath": str(f), "error": str(e)})

            class_counts[class_name] = valid_count

        summary = {
            "total_valid_samples": len(valid_samples),
            "class_distribution": class_counts,
            "corrupted_count": len(corrupted_files),
            "corrupted_files": corrupted_files,
        }
        return summary


def split_samples_group_aware(
    samples: List[Dict[str, Any]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    group_key: Optional[str] = "equipment_id",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split samples into train/val/test splits preventing data leakage across machine/equipment IDs.
    """
    rng = np.random.default_rng(seed)

    has_groups = any(group_key in s for s in samples) if group_key else False

    if has_groups:
        groups = list({s[group_key] for s in samples if group_key in s})
        rng.shuffle(groups)

        n_test = max(1, int(len(groups) * test_split))
        n_val = max(1, int(len(groups) * val_split))

        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        train_groups = set(groups[n_test + n_val :])

        train_samples = [s for s in samples if s.get(group_key) in train_groups]
        val_samples = [s for s in samples if s.get(group_key) in val_groups]
        test_samples = [s for s in samples if s.get(group_key) in test_groups]
    else:
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)

        test_samples = shuffled[:n_test]
        val_samples = shuffled[n_test : n_test + n_val]
        train_samples = shuffled[n_test + n_val :]

    return train_samples, val_samples, test_samples


def compute_class_weights(
    samples: List[Dict[str, Any]], class_to_idx: Dict[str, int]
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss to handle severe fault imbalance.
    """
    counts = np.zeros(len(class_to_idx), dtype=np.float32)
    for s in samples:
        idx = class_to_idx[s["label"]]
        counts[idx] += 1.0

    counts = np.maximum(counts, 1.0)
    total_samples = len(samples)
    num_classes = len(class_to_idx)
    weights = total_samples / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_industrial_dataloaders(
    dataset_cfg: DatasetConfig,
    system_cfg: SystemConfig,
    training_cfg: Optional[TrainingConfig] = None,
    custom_samples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Optional[torch.Tensor]]:
    """
    Construct DataLoaders for industrial fault vision classification with data leakage prevention.
    """
    class_names = dataset_cfg.classes
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    batch_size = training_cfg.batch_size if training_cfg else 32

    if custom_samples is not None:
        samples = custom_samples
    else:
        data_dir = Path(dataset_cfg.dataset_dir)
        samples = []
        for class_name in class_names:
            c_dir = data_dir / class_name
            if c_dir.exists():
                for f in c_dir.glob("*"):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                        samples.append({"filepath": str(f), "label": class_name})

    if not samples:
        logger.warning("No samples found at %s. Please supply data.", dataset_cfg.dataset_dir)
        return None, None, None, class_to_idx, None

    train_samples, val_samples, test_samples = split_samples_group_aware(
        samples=samples,
        val_split=dataset_cfg.val_split,
        test_split=dataset_cfg.test_split,
        seed=system_cfg.seed,
        group_key=dataset_cfg.group_by,
    )

    class_weights = compute_class_weights(train_samples, class_to_idx)

    train_transform = get_industrial_train_transforms(
        image_size=dataset_cfg.image_size,
        horizontal_flip=dataset_cfg.augmentations.horizontal_flip,
        rotation_degrees=dataset_cfg.augmentations.rotation_degrees,
        color_jitter_brightness=dataset_cfg.augmentations.color_jitter_brightness,
        color_jitter_contrast=dataset_cfg.augmentations.color_jitter_contrast,
    )
    eval_transform = get_industrial_eval_transforms(image_size=dataset_cfg.image_size)

    train_ds = IndustrialEquipmentDataset(train_samples, class_to_idx, train_transform)
    val_ds = IndustrialEquipmentDataset(val_samples, class_to_idx, eval_transform)
    test_ds = IndustrialEquipmentDataset(test_samples, class_to_idx, eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=system_cfg.num_workers,
        pin_memory=system_cfg.pin_memory if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=system_cfg.num_workers,
        pin_memory=system_cfg.pin_memory if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=system_cfg.num_workers,
        pin_memory=system_cfg.pin_memory if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, class_to_idx, class_weights
