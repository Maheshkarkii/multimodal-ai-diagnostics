"""
Audio signal processing and Log-Mel Spectrogram transformation pipeline.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class AudioPreprocessor:
    """
    Standardizes raw acoustic waveforms into Log-Mel Spectrogram representations.

    Processing pipeline:
    1. Load WAV file -> Float32 waveform tensor
    2. Convert multichannel audio to mono
    3. Resample to target sample rate (e.g. 16,000 Hz)
    4. Deterministic padding (for short audio) or center-crop (for long audio)
    5. Compute Mel-Scale Filterbank Spectrogram
    6. Convert power to decibel (Log-Mel) scale
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

        # PyTorch Native MelSpectrogram + AmplitudeToDB (decibel log scaling)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def load_and_standardize_waveform(
        self, audio_input: str | Path | torch.Tensor | np.ndarray, src_sr: int | None = None
    ) -> torch.Tensor:
        """Load, convert to mono, resample, and pad/crop waveform to exactly target_samples."""
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            # soundfile load
            data, sr = sf.read(str(audio_path), dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, T)
            else:
                waveform = waveform.t()  # (channels, T)
            src_sr = sr
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input.astype(np.float32))
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t() if waveform.shape[0] > waveform.shape[1] else waveform
            src_sr = src_sr or self.sample_rate
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            src_sr = src_sr or self.sample_rate
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # 1. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resample if necessary
        if src_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 3. Variable-length handling: pad or crop
        num_samples = waveform.shape[-1]
        if num_samples < self.target_samples:
            padding = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode="constant", value=0.0)
        elif num_samples > self.target_samples:
            # Center crop
            start = (num_samples - self.target_samples) // 2
            waveform = waveform[:, start : start + self.target_samples]

        return waveform

    def compute_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert standardized 1D waveform (1, T) into Log-Mel Spectrogram (1, n_mels, time_frames).
        """
        mel = self.mel_spectrogram(waveform)
        log_mel = self.amplitude_to_db(mel)

        # Standardization: zero mean, unit variance per sample
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        normalized_log_mel = (log_mel - mean) / std

        return normalized_log_mel

    def process(self, audio_input: str | Path | torch.Tensor | np.ndarray, src_sr: int | None = None) -> torch.Tensor:
        """Full end-to-end preprocessing pipeline returning normalized log-mel tensor."""
        waveform = self.load_and_standardize_waveform(audio_input, src_sr=src_sr)
        return self.compute_log_mel_spectrogram(waveform)


class SpecAugment(nn.Module):
    """
    Time and Frequency masking augmentation for industrial spectrograms (SpecAugment).

    Applies localized time and frequency masking to force the CNN to avoid overfitting to
    spurious single-frequency tonal artifacts or narrow transient noise bursts.
    """

    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 16, p: float = 0.5):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.p:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        return spec
