"""
Acoustic signal inspection utility: generates waveform and Log-Mel Spectrogram visualizations for debugging.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.audio.preprocessing.audio_transforms import AudioPreprocessor


def plot_waveform_and_spectrogram(
    audio_path: str | Path,
    save_path: str | Path | None = None,
    sample_rate: int = 16000,
    duration: float = 3.0,
) -> None:
    """Generate side-by-side plot of raw waveform and Log-Mel Spectrogram."""
    preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)
    waveform = preprocessor.load_and_standardize_waveform(audio_path).squeeze(0).numpy()
    log_mel = preprocessor.process(audio_path).squeeze(0).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # 1. Waveform
    time_axis = np.linspace(0, duration, len(waveform))
    axes[0].plot(time_axis, waveform, color="royalblue", lw=0.8)
    axes[0].set_title(f"Acoustic Waveform: {Path(audio_path).name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # 2. Log-Mel Spectrogram
    im = axes[1].imshow(log_mel, aspect="auto", origin="lower", cmap="inferno")
    axes[1].set_title("Log-Mel Spectrogram (64 filterbanks, dB normalized)")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Frequency Bands")
    fig.colorbar(im, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.close(fig)
