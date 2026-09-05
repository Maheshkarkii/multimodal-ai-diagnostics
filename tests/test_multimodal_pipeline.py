"""
Unit and integration tests for Phase 5 Multimodal Fusion & Unified Representation.
"""

import numpy as np
import pytest
import torch

from src.multimodal.data.multimodal_dataset import AlignedMultimodalDataset, custom_multimodal_collate
from src.multimodal.inference.multimodal_predictor import MultimodalPredictor
from src.multimodal.models.fusion_model import ModalityProjection, build_multimodal_model
from src.multimodal.text.text_encoder import build_text_encoder


def test_text_encoder_deterministic_embeddings():
    encoder = build_text_encoder(embedding_dim=256)
    note1 = "Motor bearings running hot with audible chirping sound."
    note2 = "Regular shift inspection complete. All values normal."

    emb1 = encoder.encode([note1])
    emb2 = encoder.encode([note2])

    assert emb1.shape == (1, 256)
    assert emb2.shape == (1, 256)
    assert not torch.allclose(emb1, emb2)


def test_modality_projection_shapes():
    proj = ModalityProjection(in_dim=1280, out_dim=256)
    dummy_v = torch.randn(4, 1280)
    out = proj(dummy_v)
    assert out.shape == (4, 256)


def test_multimodal_fusion_forward_and_masks():
    model = build_multimodal_model(
        num_classes=5,
        vision_dim=1280,
        audio_dim=512,
        sensor_dim=256,
        text_dim=256,
        shared_dim=256,
        unified_embedding_dim=256,
    )

    B = 2
    embs = {
        "vision": torch.randn(B, 1280),
        "audio": torch.randn(B, 512),
        "sensor": torch.randn(B, 256),
        "text": torch.randn(B, 256),
    }

    # Case 1: Full presence
    masks_full = {m: torch.ones(B, 1, dtype=torch.long) for m in ["vision", "audio", "sensor", "text"]}
    logits, unified_emb = model(embs, masks=masks_full, return_features=True)

    assert logits.shape == (B, 5)
    assert unified_emb.shape == (B, 256)

    # Case 2: Missing Vision & Audio (Sensor + Text only)
    masks_partial = {
        "vision": torch.zeros(B, 1, dtype=torch.long),
        "audio": torch.zeros(B, 1, dtype=torch.long),
        "sensor": torch.ones(B, 1, dtype=torch.long),
        "text": torch.ones(B, 1, dtype=torch.long),
    }
    part_logits, part_emb = model(embs, masks=masks_partial, return_features=True)
    assert part_logits.shape == (B, 5)
    assert part_emb.shape == (B, 256)


def test_multimodal_dataset_and_collate():
    samples = [{"label": "normal_state"}, {"label": "bearing_defect_wear"}]
    class_to_idx = {"normal_state": 0, "bearing_defect_wear": 1}

    v_embs = np.random.randn(2, 1280).astype(np.float32)
    a_embs = np.random.randn(2, 512).astype(np.float32)
    s_embs = np.random.randn(2, 256).astype(np.float32)
    t_embs = np.random.randn(2, 256).astype(np.float32)
    masks = {m: np.ones(2, dtype=np.int64) for m in ["vision", "audio", "sensor", "text"]}

    ds = AlignedMultimodalDataset(samples, class_to_idx, v_embs, a_embs, s_embs, t_embs, masks)
    assert len(ds) == 2

    batch = [ds[0], ds[1]]
    b_embs, b_masks, b_targets = custom_multimodal_collate(batch)

    assert b_embs["vision"].shape == (2, 1280)
    assert b_masks["audio"].shape == (2, 1)
    assert b_targets.shape == (2,)


def test_multimodal_predictor_partial_evidence():
    fusion_model = build_multimodal_model(num_classes=5)
    predictor = MultimodalPredictor(fusion_model=fusion_model, device="cpu")

    # Supply only text and sensor
    sensor_dict = {
        "temperature_c": 92.0,
        "vibration_rms_g": 5.0,
        "rotational_speed_rpm": 1475.0,
        "motor_current_a": 10.2,
        "hydraulic_pressure_bar": 140.0,
        "load_percentage": 70.0,
    }
    notes = "Bearing chirping audible from pump casing."

    res = predictor.predict(
        sensor_data=sensor_dict,
        technician_notes=notes,
        top_k=3,
        return_unified_embedding=True,
    )

    assert "predicted_machine_condition" in res
    assert "confidence" in res
    assert res["available_modalities"] == ["sensor", "text"]
    assert len(res["top_candidates"]) == 3
    assert res["unified_embedding_dim"] == 256


def test_empty_input_graceful_rejection():
    predictor = MultimodalPredictor(device="cpu")
    with pytest.raises(ValueError, match="At least one modality"):
        predictor.predict()
