"""
Comprehensive Unit & Pipeline Tests for Phase 1 Vision Foundation.
"""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.transforms import get_train_transforms, get_eval_transforms
from src.vision.model import MobileNetV2Classifier, build_vision_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.inference.predictor import VisionPredictor
from src.utils.config import ExperimentConfig, SystemConfig, ModelConfig, TrainingConfig, DatasetConfig


def test_preprocessing_transforms_shape_and_channels():
    # 28x28 grayscale PIL Image
    raw_img = Image.fromarray((np.random.rand(28, 28) * 255).astype(np.uint8), mode="L")

    train_tf = get_train_transforms(image_size=224)
    eval_tf = get_eval_transforms(image_size=224)

    t_tensor = train_tf(raw_img)
    e_tensor = eval_tf(raw_img)

    assert t_tensor.shape == (3, 224, 224)
    assert e_tensor.shape == (3, 224, 224)
    assert t_tensor.dtype == torch.float32
    assert e_tensor.dtype == torch.float32


def test_vision_model_forward_pass():
    model = build_vision_model(
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (4, 10)
    assert isinstance(output, torch.Tensor)


def test_backbone_freezing_behavior():
    model = MobileNetV2Classifier(
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )

    # Features should not require grad
    for p in model.features.parameters():
        assert not p.requires_grad

    # Classifier head MUST require grad
    for p in model.classifier.parameters():
        assert p.requires_grad

    # Unfreeze test
    model.unfreeze_feature_extractor()
    for p in model.features.parameters():
        assert p.requires_grad


def test_trainer_synthetic_epoch_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = ExperimentConfig(
            system=SystemConfig(device="cpu", checkpoint_dir=str(tmp_path / "ckpts")),
            training=TrainingConfig(epochs=1, batch_size=2, learning_rate=0.01),
            model=ModelConfig(num_classes=5, pretrained=False),
        )

        model = build_vision_model(num_classes=5, pretrained=False)

        # Synthetic DataLoader
        dummy_x = torch.randn(8, 3, 224, 224)
        dummy_y = torch.randint(0, 5, (8,))
        ds = TensorDataset(dummy_x, dummy_y)
        loader = DataLoader(ds, batch_size=2)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=cfg,
        )

        history = trainer.train()
        assert len(history["train_loss"]) == 1
        assert len(history["val_acc"]) == 1
        assert (tmp_path / "ckpts" / "best_model.pt").exists()
        assert (tmp_path / "ckpts" / "latest_model.pt").exists()


def test_evaluator_metrics_computation():
    model = build_vision_model(num_classes=3, pretrained=False)
    dummy_x = torch.randn(6, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)

    evaluator = Evaluator(
        model=model,
        device="cpu",
        class_names=["ClassA", "ClassB", "ClassC"],
    )

    results = evaluator.evaluate(loader)
    assert "accuracy" in results
    assert "f1_macro" in results
    assert "confusion_matrix" in results
    assert results["confusion_matrix"].shape == (3, 3)


def test_vision_predictor_synthetic_image():
    model = build_vision_model(num_classes=4, pretrained=False)
    classes = ["Fault_A", "Fault_B", "Fault_C", "Normal"]

    predictor = VisionPredictor(
        model=model,
        class_names=classes,
        image_size=224,
        device="cpu",
    )

    dummy_img = Image.fromarray(np.uint8(np.random.rand(28, 28) * 255), mode="L")
    res = predictor.predict(dummy_img, top_k=2)

    assert "predicted_class" in res
    assert res["predicted_class"] in classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 2
    assert res["top_k"][0]["rank"] == 1
