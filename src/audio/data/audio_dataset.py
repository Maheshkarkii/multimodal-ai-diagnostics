"""
Acoustic Machine Fault Dataset with Group-aware Leakage Prevention and Inverse Class Weighting.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.audio.preprocessing.audio_transforms import AudioPreprocessor, SpecAugment
from src.data.dataset import split_samples_group_aware, compute_class_weights
from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger

logger = setup_logger("AudioDataset")


class MachineAudioDataset(Dataset):
    """
    PyTorch Dataset for Machine Acoustic Diagnostics.

    Loads audio files, applies AudioPreprocessor on-the-fly to generate Log-Mel Spectrograms,
    and applies optional SpecAugment during training.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        class_to_idx: Dict[str, int],
        preprocessor: AudioPreprocessor,
        spec_augment: Optional[SpecAugment] = None,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.preprocessor = preprocessor
        self.spec_augment = spec_augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[index]
        audio_path = item["filepath"]
        label_str = item["label"]
        label_idx = self.class_to_idx[label_str]

        # 1. Generate Log-Mel Spectrogram (1, n_mels, time_frames)
        spec = self.preprocessor.process(audio_path)

        # 2. Training Augmentation
        if self.spec_augment:
            spec = self.spec_augment(spec)

        return spec, label_idx


def create_audio_dataloaders(
    config: ExperimentConfig,
    custom_samples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Optional[torch.Tensor]]:
    """
    Construct DataLoaders for machine sound diagnosis with group-aware isolation.
    """
    class_names = config.dataset.classes
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    batch_size = config.training.batch_size

    # 1. Gather samples
    if custom_samples is not None:
        samples = custom_samples
    else:
        data_dir = Path(config.dataset.dataset_dir)
        samples = []
        for class_name in class_names:
            c_dir = data_dir / class_name
            if c_dir.exists():
                for f in c_dir.glob("*.wav"):
                    samples.append({
                        "filepath": str(f),
                        "label": class_name,
                        "machine_id": f.stem.split("_")[0] if "_" in f.stem else "unknown",
                    })

    if not samples:
        logger.warning("No audio files found at %s.", config.dataset.dataset_dir)
        return None, None, None, class_to_idx, None

    # 2. Group-aware splitting
    train_samples, val_samples, test_samples = split_samples_group_aware(
        samples=samples,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.system.seed,
        group_key=config.dataset.group_by,
    )

    class_weights = compute_class_weights(train_samples, class_to_idx)

    # 3. Audio Preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=getattr(config.dataset, "sample_rate", 16000),
        duration=getattr(config.dataset, "duration", 3.0),
        n_mels=getattr(config.dataset, "n_mels", 64),
        n_fft=getattr(config.dataset, "n_fft", 1024),
        hop_length=getattr(config.dataset, "hop_length", 512),
        f_min=getattr(config.dataset, "f_min", 50.0),
        f_max=getattr(config.dataset, "f_max", 8000.0),
    )

    train_aug = SpecAugment(freq_mask_param=8, time_mask_param=16, p=0.5)

    # 4. Build Datasets
    train_ds = MachineAudioDataset(train_samples, class_to_idx, preprocessor, spec_augment=train_aug)
    val_ds = MachineAudioDataset(val_samples, class_to_idx, preprocessor, spec_augment=None)
    test_ds = MachineAudioDataset(test_samples, class_to_idx, preprocessor, spec_augment=None)

    # 5. Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, class_to_idx, class_weights
