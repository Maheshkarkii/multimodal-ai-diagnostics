"""
Multimodal Evaluation, Ablation Benchmark, and Robustness CLI.
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
from src.multimodal.evaluation.multimodal_evaluator import MultimodalEvaluator
from src.data.dataset import split_samples_group_aware


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Fusion Network & Run Ablations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal.yaml",
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/multimodal_fusion_baseline_best.pt",
        help="Path to trained checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("MultimodalEvaluation", level=config.system.log_level)

    manifest_csv = Path("data/multimodal/multimodal_manifest.csv")
    multimodal_dir = manifest_csv.parent
    if not manifest_csv.exists():
        df_manifest = generate_aligned_multimodal_corpus(multimodal_dir, num_machines=8, events_per_machine=25, seed=config.system.seed)
    else:
        df_manifest = pd.read_csv(manifest_csv)

    cache_file = multimodal_dir / "multimodal_embeddings.npz"
    if not cache_file.exists():
        cache = extract_and_cache_multimodal_embeddings(df_manifest, multimodal_dir, device="cpu")
    else:
        cache = np.load(cache_file)

    samples = df_manifest.to_dict(orient="records")
    for s in samples:
        s["label"] = s["fault_label"]

    class_names = config.dataset.classes
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    _, _, test_recs = split_samples_group_aware(
        samples=samples,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.system.seed,
        group_key=config.dataset.group_by,
    )

    test_indices = [samples.index(r) for r in test_recs]
    masks = {m: np.ones(len(samples), dtype=np.int64) for m in ["vision", "audio", "sensor", "text"]}

    test_ds = AlignedMultimodalDataset(
        test_recs, class_to_idx,
        cache["vision"][test_indices], cache["audio"][test_indices],
        cache["sensor"][test_indices], cache["text"][test_indices],
        {m: masks[m][test_indices] for m in masks},
    )

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=custom_multimodal_collate)

    # Load Model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Please train first.", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_multimodal_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])

    evaluator = MultimodalEvaluator(
        model=model,
        class_names=class_names,
        device=config.system.device,
        logger=logger,
    )

    ablation_results = evaluator.run_full_ablation_study(test_loader)
    logger.info("Multimodal ablation and benchmark evaluation complete.")


if __name__ == "__main__":
    main()
