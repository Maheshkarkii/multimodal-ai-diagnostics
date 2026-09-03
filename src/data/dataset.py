"""
Dataset pipeline for Fashion-MNIST pipeline validation and future visual datasets.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose

from src.preprocessing.transforms import get_train_transforms, get_eval_transforms
from src.utils.config import DatasetConfig, SystemConfig, TrainingConfig


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class TransformedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a specific transformation pipeline."""

    def __init__(self, subset: torch.utils.data.Subset, transform: Compose):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


def create_fashionmnist_dataloaders(
    dataset_cfg: DatasetConfig,
    system_cfg: SystemConfig,
    training_cfg: TrainingConfig = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Download, split, preprocess, and wrap Fashion-MNIST dataset into DataLoaders.

    Args:
        dataset_cfg: Dataset configuration parameters.
        system_cfg: System execution parameters (workers, pin_memory, seed).
        training_cfg: Optional training config for batch size.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(dataset_cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    batch_size = training_cfg.batch_size if training_cfg else 64

    # 1. Download raw PIL images without transforms initially
    full_train_raw = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=None,
    )

    test_raw = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=None,
    )

    # 2. Perform train/validation split
    total_train = len(full_train_raw)
    val_size = int(total_train * dataset_cfg.val_split)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(system_cfg.seed)
    train_subset, val_subset = random_split(
        full_train_raw, [train_size, val_size], generator=generator
    )

    # 3. Attach distinct augmentation/deterministic transforms
    train_transforms = get_train_transforms(image_size=dataset_cfg.image_size)
    eval_transforms = get_eval_transforms(image_size=dataset_cfg.image_size)

    train_dataset = TransformedSubset(train_subset, transform=train_transforms)
    val_dataset = TransformedSubset(val_subset, transform=eval_transforms)
    test_dataset = TransformedSubset(test_raw, transform=eval_transforms)

    # 4. Construct DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=system_cfg.num_workers,
        pin_memory=system_cfg.pin_memory if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=system_cfg.num_workers,
        pin_memory=system_cfg.pin_memory if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=system_cfg.num_workers,
        pin_memory=system_cfg.pin_memory if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, FASHION_MNIST_CLASSES
