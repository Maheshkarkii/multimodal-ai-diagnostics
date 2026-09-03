"""
Sample Industrial Machine Acoustic Dataset Generator.

Generates verified physical audio waveforms (.wav at 16kHz):
1. normal_operation: Steady electric motor 50Hz hum + smooth white background noise.
2. bearing_defect: High-frequency periodic impact bursts (spalls/flaking impacts at 1200Hz harmonics).
3. loose_component: Random low-frequency mechanical rattling impulses.
4. rotor_imbalance: 1X rotational unbalance sinusoid (100Hz) with amplitude modulation.
5. cavitation_anomaly: High-frequency broadband burst noise characteristic of collapsing vapor bubbles.
"""

from pathlib import Path
import numpy as np
import soundfile as sf


def generate_synthetic_acoustic_dataset(
    output_dir: Path,
    samples_per_class: int = 30,
    sample_rate: int = 16000,
    duration: float = 3.0,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    classes = [
        "normal_operation",
        "bearing_defect",
        "loose_component",
        "rotor_imbalance",
        "cavitation_anomaly",
    ]

    for class_name in classes:
        c_dir = output_dir / class_name
        c_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            machine_id = f"pump{(i % 6) + 1:02d}"
            filename = f"{machine_id}_{class_name}_{i:03d}.wav"
            file_path = c_dir / filename

            # Base background motor noise
            base_noise = rng.normal(0, 0.02, len(t))
            motor_hum = 0.08 * np.sin(2 * np.pi * 50 * t)

            if class_name == "normal_operation":
                signal = motor_hum + base_noise

            elif class_name == "bearing_defect":
                # Periodic impact bursts (e.g. 10 Hz repetition with 1500 Hz damped resonant ring)
                impacts = np.zeros_like(t)
                period_samples = int(sample_rate / 12)  # 12 Hz impact frequency
                for p in range(0, len(t), period_samples):
                    ring_len = min(600, len(t) - p)
                    ring_t = np.linspace(0, ring_len / sample_rate, ring_len)
                    impacts[p : p + ring_len] += 0.25 * np.exp(-ring_t * 60) * np.sin(2 * np.pi * 1500 * ring_t)
                signal = motor_hum + base_noise + impacts

            elif class_name == "loose_component":
                # Random sporadic mechanical clanks
                rattle = np.zeros_like(t)
                n_clanks = rng.integers(5, 15)
                for _ in range(n_clanks):
                    pos = rng.integers(0, len(t) - 800)
                    clank_t = np.linspace(0, 0.05, 800)
                    rattle[pos : pos + 800] += 0.3 * np.exp(-clank_t * 80) * np.sin(2 * np.pi * 320 * clank_t)
                signal = motor_hum + base_noise + rattle

            elif class_name == "rotor_imbalance":
                # Strong 1X rotational frequency (120 Hz) with 10 Hz amplitude modulation
                mod = 0.5 * (1 + np.sin(2 * np.pi * 10 * t))
                imbalance = 0.35 * mod * np.sin(2 * np.pi * 120 * t)
                signal = motor_hum + base_noise + imbalance

            elif class_name == "cavitation_anomaly":
                # High-frequency hiss / bursting vapor bubble noise (filtered high-freq Gaussian noise)
                cavitation = rng.normal(0, 0.15, len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 30 * t))
                signal = motor_hum + base_noise * 0.5 + cavitation

            # Normalize to avoid clipping
            max_val = np.max(np.abs(signal)) + 1e-6
            signal = (signal / max_val * 0.85).astype(np.float32)

            sf.write(str(file_path), signal, sample_rate)

    return output_dir


if __name__ == "__main__":
    generate_synthetic_acoustic_dataset("data/audio/raw", samples_per_class=30)
