"""
Unit and integration tests for Phase 3 Acoustic Intelligence pipeline.
"""

from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import torch

from src.audio.preprocessing.audio_transforms import AudioPreprocessor, SpecAugment
from src.audio.models.audio_cnn import AcousticCNN, build_audio_model
from src.audio.data.audio_dataset import MachineAudioDataset
from src.audio.data.generate_sample_audio import generate_synthetic_acoustic_dataset
from src.audio.inference.audio_predictor import AudioPredictor
from src.audio.preprocessing.visualize_audio import plot_waveform_and_spectrogram


def test_audio_preprocessor_synthetic_sine_wave():
    sr = 16000
    duration = 2.0  # 2s input (should be padded to 3s)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "test_sine.wav"
        sf.write(str(wav_path), sine, sr)

        preprocessor = AudioPreprocessor(sample_rate=16000, duration=3.0, n_mels=64)
        spec = preprocessor.process(wav_path)

        # Output must be (1, n_mels, time_frames)
        assert spec.shape[0] == 1
        assert spec.shape[1] == 64
        assert spec.shape[2] > 0
        assert spec.dtype == torch.float32


def test_spec_augment_transform():
    spec = torch.randn(1, 64, 94)
    augment = SpecAugment(freq_mask_param=8, time_mask_param=16, p=1.0)
    augment.train()
    aug_spec = augment(spec.clone())
    assert aug_spec.shape == (1, 64, 94)


def test_acoustic_cnn_forward_and_embedding():
    model = build_audio_model(num_classes=5, in_channels=1, embedding_dim=512)
    dummy_spec = torch.randn(4, 1, 64, 94)

    logits, embeddings = model(dummy_spec, return_features=True)
    assert logits.shape == (4, 5)
    assert embeddings.shape == (4, 512)

    standalone_emb = model.extract_features(dummy_spec)
    assert standalone_emb.shape == (4, 512)


def test_audio_dataset_and_generator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        generate_synthetic_acoustic_dataset(tmp_path, samples_per_class=3, seed=42)

        classes = ["normal_operation", "bearing_defect", "loose_component", "rotor_imbalance", "cavitation_anomaly"]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        samples = []
        for c in classes:
            for f in (tmp_path / c).glob("*.wav"):
                samples.append({"filepath": str(f), "label": c})

        preprocessor = AudioPreprocessor(sample_rate=16000, duration=1.0, n_mels=32)
        ds = MachineAudioDataset(samples, class_to_idx, preprocessor)

        assert len(ds) == 15
        spec, label = ds[0]
        assert spec.shape[0] == 1
        assert spec.shape[1] == 32
        assert isinstance(label, int)


def test_audio_predictor_with_embedding():
    model = build_audio_model(num_classes=5, in_channels=1, embedding_dim=512)
    predictor = AudioPredictor(model=model, device="cpu", sample_rate=16000, duration=1.0)

    # Synthetic waveform tensor
    dummy_audio = np.random.randn(16000).astype(np.float32)
    res = predictor.predict(dummy_audio, src_sr=16000, top_k=2, return_embedding=True)

    assert "predicted_sound" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_candidates"]) == 2
    assert res["embedding_dim"] == 512
    assert len(res["acoustic_embedding"]) == 512


def test_audio_visualization_generation():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "test_vis.wav"
        png_path = Path(tmp_dir) / "spec_plot.png"
        sf.write(str(wav_path), sine, sr)

        plot_waveform_and_spectrogram(wav_path, save_path=png_path)
        assert png_path.exists()
