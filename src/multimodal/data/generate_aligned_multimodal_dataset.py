"""
Synchronized Multimodal Dataset Generator & Manifest Builder.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw
import torch

from src.vision.model import build_vision_model
from src.audio.models.audio_cnn import build_audio_model
from src.sensor.models.sensor_mlp import build_sensor_model
from src.multimodal.text.text_encoder import build_text_encoder
from src.audio.preprocessing.audio_transforms import AudioPreprocessor
from src.preprocessing.transforms import get_industrial_eval_transforms
from src.sensor.preprocessing.sensor_scaler import SensorPreprocessor


def generate_aligned_multimodal_corpus(
    output_dir: Path,
    num_machines: int = 8,
    events_per_machine: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    audio_dir = output_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    classes = [
        "normal_state",
        "bearing_defect_wear",
        "structural_crack_loose",
        "rotor_unbalance",
        "hydraulic_cavitation",
    ]

    manifest_rows = []

    for m_idx in range(1, num_machines + 1):
        machine_id = f"machine_asset_{m_idx:02d}"

        for e_idx in range(events_per_machine):
            event_id = f"{machine_id}_event_{e_idx:03d}"
            label = rng.choice(classes, p=[0.30, 0.20, 0.20, 0.15, 0.15])

            img_path = img_dir / f"{event_id}.png"
            base_col = rng.integers(130, 170, size=3).astype(np.uint8)
            img_arr = np.ones((224, 224, 3), dtype=np.uint8) * base_col
            img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img)

            audio_path = audio_dir / f"{event_id}.wav"
            noise = rng.normal(0, 0.02, len(t))
            motor_hum = 0.08 * np.sin(2 * np.pi * 50 * t)

            temp = rng.normal(55.0, 3.0)
            vib = rng.normal(1.8, 0.3)
            rpm = rng.normal(1495.0, 10.0)
            curr = rng.normal(10.2, 0.6)
            press = rng.normal(140.0, 5.0)
            load = rng.normal(70.0, 5.0)

            if label == "normal_state":
                sig = motor_hum + noise
                text_note = "Regular shift inspection complete. Machine operating within normal acoustic and thermal limits."
            elif label == "bearing_defect_wear":
                draw.ellipse([60, 60, 164, 164], outline=(40, 40, 40), width=5)
                impacts = np.zeros_like(t)
                for p in range(0, len(t), int(sr / 12)):
                    ring_len = min(500, len(t) - p)
                    rt = np.linspace(0, ring_len / sr, ring_len)
                    impacts[p : p + ring_len] += 0.3 * np.exp(-rt * 60) * np.sin(2 * np.pi * 1400 * rt)
                sig = motor_hum + noise + impacts
                temp += rng.uniform(35.0, 50.0)
                vib += rng.uniform(2.5, 4.5)
                text_note = "Audible high-pitched periodic chirping from drive-end bearing. Casing hot to touch."
            elif label == "structural_crack_loose":
                draw.line([(30, 40), (80, 90), (130, 120), (180, 180)], fill=(15, 15, 15), width=4)
                clanks = np.zeros_like(t)
                for _ in range(8):
                    pos = rng.integers(0, len(t) - 600)
                    clanks[pos : pos + 600] += 0.35 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.04, 600))
                sig = motor_hum + noise + clanks
                vib += rng.uniform(4.5, 7.5)
                text_note = "Visible crack propagation on mounting bracket and audible mechanical rattling."
            elif label == "rotor_unbalance":
                draw.polygon([(140, 140), (200, 120), (200, 200), (120, 200)], fill=(20, 20, 20))
                mod = 0.5 * (1 + np.sin(2 * np.pi * 10 * t))
                sig = motor_hum + noise + 0.4 * mod * np.sin(2 * np.pi * 120 * t)
                vib += rng.uniform(6.0, 10.0)
                rpm -= rng.uniform(40.0, 100.0)
                text_note = "Severe 1X rotational vibration harmonics and mild rotational speed drop under load."
            elif label == "hydraulic_cavitation":
                for _ in range(5):
                    cx, cy = rng.integers(50, 170), rng.integers(50, 170)
                    draw.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(150, 60, 25))
                cav = rng.normal(0, 0.18, len(t))
                sig = motor_hum + noise * 0.5 + cav
                press = rng.uniform(30.0, 60.0)
                text_note = "Broadband hissing noise in pump suction line. Hydraulic pressure dropped severely."

            img.save(img_path)
            sig_max = np.max(np.abs(sig)) + 1e-6
            sf.write(str(audio_path), (sig / sig_max * 0.85).astype(np.float32), sr)

            manifest_rows.append({
                "sample_id": event_id,
                "machine_id": machine_id,
                "image_path": str(img_path),
                "audio_path": str(audio_path),
                "temperature_c": round(float(temp), 2),
                "vibration_rms_g": round(float(vib), 3),
                "rotational_speed_rpm": round(float(rpm), 1),
                "motor_current_a": round(float(curr), 2),
                "hydraulic_pressure_bar": round(float(press), 2),
                "load_percentage": round(float(load), 1),
                "technician_notes": text_note,
                "fault_label": label,
            })

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = output_dir / "multimodal_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    return df_manifest


@torch.no_grad()
def extract_and_cache_multimodal_embeddings(
    df_manifest: pd.DataFrame,
    output_dir: Path,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    vision_model = build_vision_model(num_classes=5, pretrained=False).to(dev).eval()
    audio_model = build_audio_model(num_classes=5, in_channels=1, embedding_dim=512).to(dev).eval()
    sensor_model = build_sensor_model(in_features=6, num_classes=5, embedding_dim=256).to(dev).eval()
    text_encoder = build_text_encoder(embedding_dim=256).to(dev).eval()

    img_transform = get_industrial_eval_transforms(image_size=224)
    audio_prep = AudioPreprocessor(sample_rate=16000, duration=3.0)
    feature_cols = ["temperature_c", "vibration_rms_g", "rotational_speed_rpm", "motor_current_a", "hydraulic_pressure_bar", "load_percentage"]
    sensor_prep = SensorPreprocessor(feature_columns=feature_cols).fit(df_manifest)

    v_embs, a_embs, s_embs, t_embs = [], [], [], []

    for idx, row in df_manifest.iterrows():
        pil_img = Image.open(row["image_path"]).convert("RGB")
        v_tensor = img_transform(pil_img).unsqueeze(0).to(dev)
        v_emb = vision_model.extract_features(v_tensor).cpu().numpy()[0]
        v_embs.append(v_emb)

        spec = audio_prep.process(row["audio_path"]).unsqueeze(0).to(dev)
        a_emb = audio_model.extract_features(spec).cpu().numpy()[0]
        a_embs.append(a_emb)

        s_scaled = sensor_prep.transform(pd.DataFrame([row[feature_cols]]))
        s_tensor = torch.tensor(s_scaled, dtype=torch.float32).to(dev)
        s_emb = sensor_model.extract_features(s_tensor).cpu().numpy()[0]
        s_embs.append(s_emb)

        t_emb = text_encoder.encode([row["technician_notes"]]).cpu().numpy()[0]
        t_embs.append(t_emb)

    cache = {
        "vision": np.array(v_embs, dtype=np.float32),
        "audio": np.array(a_embs, dtype=np.float32),
        "sensor": np.array(s_embs, dtype=np.float32),
        "text": np.array(t_embs, dtype=np.float32),
    }

    np.savez_compressed(output_dir / "multimodal_embeddings.npz", **cache)
    return cache
