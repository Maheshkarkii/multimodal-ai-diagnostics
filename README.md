# AI Field Engineer ? Multimodal Autonomous Troubleshooting & Diagnosis System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, diagnosis, and troubleshooting across images, audio, sensor telemetry, and technical documentation.

---

## ?? Phase 3 ? Acoustic Intelligence for Machine Fault Detection

Phase 3 introduces PyTorch-native **acoustic signal processing and diagnostic sound classification** for industrial rotating equipment (motors, pumps, compressors, turbines).

Key capabilities introduced:
- **Physics-Informed Signal Preprocessing**: Converts raw 1D acoustic pressure waveforms into 2D **Log-Mel Spectrograms** (64 filterbanks, 1024-point FFT, 512 hop length) to capture acoustic frequency peaks, periodic impact spikes, and harmonic sidebands.
- **Variable-Length Handling**: Deterministic zero-padding for short clips and center-window cropping for extended continuous audio streams.
- **SpecAugment Regularization**: Time and frequency masking transforms to prevent overfitting to spurious ambient tones or narrow-band shop-floor background noise.
- **Acoustic CNN Architecture**: 4-stage hierarchical 2D convolutional network with adaptive pooling and intermediate **512-dimensional acoustic feature embedding extraction** for cross-modal fusion.
- **Leakage Prevention by Machine Entity**: Group-based splitting (`machine_id`) ensures audio captured from the same equipment unit never leaks across training and evaluation splits.

---

## ??? Acoustic Processing & Model Architecture

```
Raw Machine Acoustic Recording (.wav @ 16 kHz)
   ?
   ?
[Audio Signal Standardizer] ???? Mono conversion, 16kHz resampling, deterministic padding/cropping (3.0s)
   ?
   ?
[Log-Mel Filterbank Engine] ???? STFT (n_fft=1024, hop=512) -> Mel Scale (64 bands) -> dB normalization
   ?
   ?
[SpecAugment (Train-only)] ????? Frequency & Time masking
   ?
   ?
[Acoustic 2D CNN Backbone] ????? 4x [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d] (32..256 channels)
   ?
   ?
[Global Adaptive AvgPool] ?????? Spatial reduction to (B, 256)
   ?
   ????????????????????????????? [512-dim Acoustic Feature Embedding] ??? (Reserved for Multimodal Fusion)
   ?
   ?
[Linear Diagnostic Head] ??????? Dropout(0.3) + Linear(512 -> 5 Acoustic Anomaly Classes)
   ?
   ?
[Softmax & Confidence Metric] ??? Predicted Acoustic Anomaly, Confidence Score, Top-K Failure Candidates
```

---

## ?? Acoustic Diagnostic Benchmark Results

Evaluated on the 5-class machine sound dataset (`normal_operation`, `bearing_defect`, `loose_component`, `rotor_imbalance`, `cavitation_anomaly`):

- **Test Accuracy**: **96.00%**
- **Macro F1-Score**: **0.9596**
- **Weighted F1-Score**: **0.9596**
- **Confidence Calibration**:
  - Correct predictions mean confidence: **$0.782$**
  - Misclassified predictions mean confidence: **$0.394$**
- **Error Analysis**: Isolated error (`loose_component` $\rightarrow$ `normal_operation`, 1 sample) occurred when mechanical impacts were masked by higher baseline motor hum.

---

## ?? Scientific & Engineering Distinction

> [!WARNING]
> **Acoustic Anomaly Detection $\neq$ Complete Machine Root-Cause Diagnosis**
>
> An acoustic classifier detects anomalous auditory frequency signatures (such as ultrasonic cavitation hiss or bearing impact ringing). However, machine sound is sensitive to acoustic reflection, surrounding shop-floor background noise, and microphone positioning. Definitive autonomous diagnosis requires correlating acoustic embeddings with visual surface wear and time-series sensor telemetry (Phase 4 & 5).

---

## ?? CLI Execution Commands

### 1. Run Unit & Integration Tests
```bash
pytest -v
```

### 2. Train Acoustic CNN
```bash
python scripts/train_audio.py --config configs/audio.yaml
```

### 3. Evaluate & Error Analysis
```bash
python scripts/evaluate_audio.py --config configs/audio.yaml --checkpoint checkpoints/acoustic_fault_baseline_best.pt
```

### 4. Run Audio Inference with 512-dim Embedding Extraction
```bash
python scripts/inference_audio.py --audio data/audio/raw/bearing_defect/pump01_bearing_defect_000.wav --checkpoint checkpoints/acoustic_fault_baseline_best.pt --extract-embedding
```

---

## ?? Roadmap: Next Phases

- **Phase 4**: Sensor Intelligence ? Time-series telemetry representations (Vibration, Temperature, RPM, Current, Pressure).
- **Phase 5**: Multimodal Fusion Engine ? Joint cross-attention module fusing Vision (1280-dim), Audio (512-dim), and Sensor embeddings.
- **Phase 6**: Technical-Document RAG & Agentic Diagnostic Reasoning.
- **Phase 7**: FastAPI Backend & Next.js UI.
