"""
Training CLI for Phase 5 Multimodal Fusion.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.logging import setup_logger
from src.multimodal.data.generate_aligned_multimodal_dataset import (
    generate_aligned_multimodal_corpus,
    extract_and_cache_multimodal_embeddings,
)
from src.multimodal.data.multimodal_dataset import AlignedMultimodalDataset, custom_multimodal_collate
from src.multimodal.models.fusion_model import build_multimodal_model
from src.multimodal.training.multimodal_trainer import MultimodalTrainer
from src.data.dataset import split_samples_group_aware, compute_class_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Diagnostic Network")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal.yaml",
        help="Path to YAML configuration",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("MultimodalTraining", level=config.system.log_level)
    logger.info("Starting Multimodal Fusion Experiment: %s", config.experiment_name)

    # 1. Dataset Manifest Verification & Cache Generation
    manifest_csv = Path(config.dataset.manifest_path) if hasattr(config.dataset, "manifest_path") else Path("data/multimodal/multimodal_manifest.csv")
    multimodal_dir = manifest_csv.parent

    if not manifest_csv.exists():
        logger.info("Generating verified aligned multimodal dataset at %s...", multimodal_dir)
        df_manifest = generate_aligned_multimodal_corpus(multimodal_dir, num_machines=8, events_per_machine=25, seed=config.system.seed)
    else:
        df_manifest = pd.read_csv(manifest_csv)

    cache_file = multimodal_dir / "multimodal_embeddings.npz"
    if not cache_file.exists():
        logger.info("Extracting and caching frozen modality representations (Vision, Audio, Sensor, Text)...")
        cache = extract_and_cache_multimodal_embeddings(df_manifest, multimodal_dir, device="cpu")
    else:
        cache = np.load(cache_file)

    # 2. Group-aware Splitting by Machine Entity
    samples = df_manifest.to_dict(orient="records")
    for s in samples:
        s["label"] = s["fault_label"]

    class_names = config.dataset.classes
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_recs, val_recs, test_recs = split_samples_group_aware(
        samples=samples,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.system.seed,
        group_key=config.dataset.group_by,
    )

    train_indices = [samples.index(r) for r in train_recs]
    val_indices = [samples.index(r) for r in val_recs]
    test_indices = [samples.index(r) for r in test_recs]

    # Full presence masks by default
    n_total = len(samples)
    masks = {m: np.ones(n_total, dtype=np.int64) for m in ["vision", "audio", "sensor", "text"]}

    # 3. Create Datasets & Loaders
    train_ds = AlignedMultimodalDataset(
        train_recs, class_to_idx,
        cache["vision"][train_indices], cache["audio"][train_indices],
        cache["sensor"][train_indices], cache["text"][train_indices],
        {m: masks[m][train_indices] for m in masks},
    )
    val_ds = AlignedMultimodalDataset(
        val_recs, class_to_idx,
        cache["vision"][val_indices], cache["audio"][val_indices],
        cache["sensor"][val_indices], cache["text"][val_indices],
        {m: masks[m][val_indices] for m in masks},
    )

    batch_size = config.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_multimodal_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_multimodal_collate)

    class_weights = compute_class_weights(train_recs, class_to_idx)

    # 4. Build Fusion Model
    model = build_multimodal_model(
        num_classes=len(class_names),
        vision_dim=1280,
        audio_dim=512,
        sensor_dim=256,
        text_dim=256,
        shared_dim=256,
        fusion_hidden_dims=[512, 256],
        unified_embedding_dim=256,
        dropout=0.25,
        modality_dropout_prob=0.20,
    )

    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights,
        logger=logger,
    )

    history = trainer.train()
    logger.info("Multimodal fusion training completed successfully.")


if __name__ == "__main__":
    main()
