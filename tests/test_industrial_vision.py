"""
Unit and Integration tests for Phase 2 Industrial Vision components.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import (
    DatasetValidator,
    compute_class_weights,
    split_samples_group_aware,
)
from src.data.generate_sample_dataset import generate_synthetic_industrial_dataset
from src.inference.predictor import VisionPredictor
from src.preprocessing.transforms import get_industrial_eval_transforms, get_industrial_train_transforms
from src.vision.model import build_vision_model


def test_industrial_transforms_and_color_jitter():
    img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))
    train_tf = get_industrial_train_transforms(image_size=224, color_jitter_brightness=0.3)
    eval_tf = get_industrial_eval_transforms(image_size=224)

    t_out = train_tf(img)
    e_out = eval_tf(img)

    assert t_out.shape == (3, 224, 224)
    assert e_out.shape == (3, 224, 224)
    assert t_out.dtype == torch.float32


def test_feature_embedding_extraction():
    model = build_vision_model(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    logits, embeddings = model(x, return_features=True)
    assert logits.shape == (2, 5)
    assert embeddings.shape == (2, 1280)

    standalone_emb = model.extract_features(x)
    assert standalone_emb.shape == (2, 1280)


def test_group_aware_split_leakage_prevention():
    # Synthetic samples with equipment_ids
    samples = [
        {"filepath": "f1.png", "label": "crack", "equipment_id": "unit_01"},
        {"filepath": "f2.png", "label": "crack", "equipment_id": "unit_01"},
        {"filepath": "f3.png", "label": "normal", "equipment_id": "unit_02"},
        {"filepath": "f4.png", "label": "normal", "equipment_id": "unit_02"},
        {"filepath": "f5.png", "label": "corrosion", "equipment_id": "unit_03"},
        {"filepath": "f6.png", "label": "bearing_fault", "equipment_id": "unit_04"},
    ]

    train, val, test = split_samples_group_aware(
        samples, val_split=0.25, test_split=0.25, seed=42, group_key="equipment_id"
    )

    train_groups = {s["equipment_id"] for s in train}
    val_groups = {s["equipment_id"] for s in val}
    test_groups = {s["equipment_id"] for s in test}

    # Verify no equipment overlap exists between any of the splits
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_class_weights_computation():
    samples = [
        {"label": "normal"},
        {"label": "normal"},
        {"label": "normal"},
        {"label": "crack"},
    ]
    class_to_idx = {"normal": 0, "crack": 1}
    weights = compute_class_weights(samples, class_to_idx)
    # Rare class 'crack' must receive higher weight than frequent class 'normal'
    assert weights[1] > weights[0]


def test_dataset_validator_and_generator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        generate_synthetic_industrial_dataset(tmp_path, samples_per_class=5, seed=42)
        summary = DatasetValidator.validate_and_summarize(tmp_path)

        assert summary["total_valid_samples"] == 25
        assert summary["corrupted_count"] == 0
        assert len(summary["class_distribution"]) == 5


def test_predictor_with_embedding_extraction():
    model = build_vision_model(num_classes=5, pretrained=False)
    predictor = VisionPredictor(model=model, device="cpu")

    dummy_img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    res = predictor.predict(dummy_img, top_k=2, return_embedding=True)

    assert "predicted_fault" in res
    assert "feature_embedding" in res
    assert res["embedding_dim"] == 1280
