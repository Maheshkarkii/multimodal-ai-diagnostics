# AI Field Engineer ? Multimodal Autonomous Troubleshooting & Diagnosis System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, diagnosis, and troubleshooting across images, audio, sensor telemetry, and technical documentation.

---

## ?? Phase 2 ? Real Equipment Vision Intelligence

Phase 2 transitions the computer vision subsystem from pipeline validation to **real equipment & component visual fault diagnostics**. It introduces:
- **Group-Aware Data Splitting**: Prevents optimistic data leakage by ensuring all images/angles from a given physical equipment unit (`equipment_id`) reside exclusively in either train, validation, or test sets.
- **Class-Imbalance Handling**: Computes inverse-frequency class weights for `CrossEntropyLoss` to handle severe industrial fault rarity.
- **Transfer Learning vs. Fine-Tuning Benchmarking**: Compares frozen-backbone feature extraction with progressive deeper-layer fine-tuning using discriminative learning rates.
- **Diagnostic Error & Confidence Analysis**: Analyzes failure patterns, confused class pairs, and checks confidence calibration.
- **Multimodal Embedding Extraction**: Exposes 1280-dimensional feature representations from MobileNetV2 for future cross-modal fusion.

---

## ??? Phase 2 Vision Architecture

```
Equipment / Component Image (RGB or Grayscale Sensor)
   ?
   ?
[Preprocessing & Augmentations] ??? 3-channel RGB conversion, Resize (224x224),
                                    Physical-valid horizontal flips, Lighting ColorJitter (train)
   ?
   ?
[Group-Aware PyTorch DataLoader] ?? Machine-isolated batches with inverse class weights
   ?
   ?
[MobileNetV2 Backbone] ???????????? Pretrained Feature Extractor (Inverted Residual Blocks 0..18)
   ?                                 ? Exp A: Fully frozen backbone
   ?                                 ? Exp B: Deeper blocks unfrozen with discriminative LR
   ?
   ???????????????????????????????? [1280-dim Intermediate Feature Embedding] ??? (Reserved for Multimodal Fusion)
   ?
   ?
[Adaptive AvgPool & Head] ????????? Dropout(0.2..0.3) + Linear(1280 -> 5 Fault Classes)
   ?
   ?
[Softmax & Calibration Check] ????? Predicted Fault, Confidence Estimate, Ranked Diagnostic Candidates
```

---

## ?? Experimental Comparison: Transfer Learning vs Fine-Tuning

Evaluated on the 5-class industrial inspection benchmark (`normal`, `bearing_fault`, `corrosion`, `surface_crack`, `damaged_component`):

| Experiment Configuration | Training Strategy | Test Accuracy | Macro F1-Score | Weighted F1 | Primary Confusion Pattern |
|---|---|---|---|---|---|
| **Exp A: Frozen Baseline** (`exp_frozen_baseline`) | Frozen backbone, head-only training | **83.33%** | **0.8541** | **0.8189** | `normal` misclassified as `damaged_component` (3) |
| **Exp B: Fine-Tuned** (`exp_fine_tuned`) | Unfrozen top 4 blocks + discriminative LR ($5 \times 10^{-5}$) | **93.33%** | **0.9444** | **0.9315** | `normal` misclassified as `damaged_component` (2) |

---

## ?? Important Scientific & Engineering Distinction

> [!WARNING]
> **Model Prediction $\neq$ True Physical Machine Diagnosis**
>
> A computer vision model alone detects visual surface anomalies and structural texture changes. It **cannot** definitively diagnose internal thermodynamic faults, subsurface bearing fatigue, or sensor drift without correlating audio harmonics, vibration frequencies, and operating documentation. This single-modality system serves as an essential feature provider for upcoming multimodal fusion.

---

## ?? Quick Start & CLI Execution

### 1. Run Unit & Pipeline Tests
```bash
pytest -v
```

### 2. Run Training Experiments

**Experiment A (Frozen Backbone):**
```bash
python scripts/train.py --config configs/experiments/frozen_baseline.yaml
```

**Experiment B (Fine-Tuning Deeper Layers):**
```bash
python scripts/train.py --config configs/experiments/fine_tuned.yaml
```

### 3. Evaluate & Run Error Analysis
```bash
python scripts/evaluate.py --config configs/experiments/fine_tuned.yaml --checkpoint checkpoints/exp_fine_tuned_best.pt
```

### 4. Run Single-Image Inference with Feature Embedding Extraction
```bash
python scripts/inference.py --image data/industrial_inspection/surface_crack/machine_01_surface_crack_000.png --checkpoint checkpoints/exp_fine_tuned_best.pt --extract-embedding
```

---

## ?? Roadmap: Next Phases

- **Phase 3**: Audio Intelligence ? Abnormal machine sound detection and acoustic spectrogram feature extraction using PyTorch.
- **Phase 4**: Sensor & Telemetry Models ? Vibration, temperature, and pressure time-series modeling.
- **Phase 5**: Multimodal Fusion Engine ? Joint cross-attention representation combining Vision, Audio, and Sensor vectors.
- **Phase 6**: Technical-Document RAG & Agentic Diagnostic Reasoning Workflow.
- **Phase 7**: FastAPI Backend & Next.js Interactive Field Engineer UI.
