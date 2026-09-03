# AI Field Engineer ? Multimodal Autonomous Troubleshooting & Diagnosis System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, diagnosis, and troubleshooting across images, audio, sensor telemetry, and technical documentation.

It combines PyTorch-based computer vision, audio analysis, sensor data, technical-document RAG, and AI agents to generate explainable diagnoses, confidence scores, and recommended corrective actions.

---

## ?? Phase 1 ? Professional PyTorch Vision Foundation

Phase 1 establishes the core computer-vision baseline pipeline, configuration management, transfer-learning workflow, checkpointing, and evaluation metrics.

> **Dataset Note**: In Phase 1, **Fashion-MNIST** is utilized strictly as a fast, reproducible **pipeline-validation dataset** to verify tensor contracts, preprocessing transforms, transfer-learning mechanics, checkpointing, and evaluation metrics. Real industrial equipment and component fault images will be integrated in subsequent phases.

---

## ??? Vision Pipeline Architecture

```
Raw Image (1x28x28 or 3xHxW)
   ?
   ?
[Preprocessing & Transforms] ??? Convert to 3 channels, Resize (224x224), Data Augmentation (train), ImageNet Normalization
   ?
   ?
[PyTorch DataLoader] ??????????? Batched, shuffled, multi-worker tensor streaming
   ?
   ?
[MobileNetV2 Backbone] ????????? Pretrained Feature Extractor (frozen or fine-tuned)
   ?
   ?
[Global Adaptive AvgPool] ?????? Spatial reduction to (B, 1280)
   ?
   ?
[Classifier Head] ?????????????? Dropout(0.2) + Linear(1280 -> 10 classes)
   ?
   ?
[Softmax & Top-K Ranking] ?????? Fault/Class Prediction, Confidence Estimate, Ranked Causes
```

---

## ?? Why MobileNetV2?

1. **Pretrained Feature Representation**: Leverages rich visual representations learned on ImageNet-1k, reducing required training time and data volume for downstream classification.
2. **Transfer Learning Flexibility**: Allows freezing the convolutional feature extractor to rapidly train custom diagnostic heads before full fine-tuning.
3. **Computational Efficiency & Edge Deployment**: Inverted residual blocks with depthwise separable convolutions make MobileNetV2 ideal for edge execution on field laptops, handheld diagnostic tablets, and embedded industrial gateways.

---

## ?? Repository Structure

```
multimodal-ai-diagnostics/
??? checkpoints/              # Saved model checkpoints (.pt)
??? configs/                  # Modular YAML experiment configs
?   ??? base.yaml             # Base settings
?   ??? vision.yaml           # Vision pipeline configuration
??? data/
?   ??? raw/                  # Downloaded raw datasets
?   ??? processed/            # Processed artifacts
?   ??? metadata/             # Schema definitions and class mappings
??? reports/                  # Evaluation summaries and confusion matrices
??? scripts/
?   ??? train.py              # Standalone training script
?   ??? evaluate.py           # Standalone test evaluation script
?   ??? inference.py          # Single-image / batch inference CLI
??? src/
?   ??? data/                 # Dataset loaders, splitters & wrappers
?   ??? preprocessing/        # TorchVision transforms & normalizations
?   ??? vision/               # MobileNetV2 architecture & head definitions
?   ??? training/             # Trainer loop, AMP scaler, schedulers, checkpointing
?   ??? evaluation/           # Multiclass metrics (Acc, Precision, Recall, F1, Confusion Matrix)
?   ??? inference/            # VisionPredictor for production inference
?   ??? utils/                # Typed configuration, device resolution, logging, seeding
??? tests/                    # Comprehensive unit and integration test suite
??? pyproject.toml            # PEP 621 / PEP 517 build configuration
??? requirements.txt          # Production dependencies
??? requirements-dev.txt      # Development & testing dependencies
??? README.md                 # Project documentation
```

---

## ?? Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Run Tests

```bash
pytest -v
```

### 3. Train the Vision Model

```bash
python scripts/train.py --config configs/vision.yaml
```

To resume training from an existing checkpoint:
```bash
python scripts/train.py --config configs/vision.yaml --resume checkpoints/latest_model.pt
```

### 4. Evaluate on Test Set

```bash
python scripts/evaluate.py --config configs/vision.yaml --checkpoint checkpoints/best_model.pt
```

### 5. Run Single-Image Inference

```bash
python scripts/inference.py --image path/to/sample_image.png --checkpoint checkpoints/best_model.pt --top-k 3
```

---

## ?? Roadmap: Future Phases

- **Phase 2**: Industrial Equipment Visual Fault Dataset & Custom Domain Feature Fine-Tuning.
- **Phase 3**: Audio Acoustic Diagnostics (PyTorch Spectrograms / Audio CNNs for motor/bearing anomalies).
- **Phase 4**: Time-Series Sensor & Telemetry Models (Vibration, Temperature, Pressure).
- **Phase 5**: Multimodal Fusion Engine (Cross-Attention & Joint Representation).
- **Phase 6**: Technical-Document RAG & Diagnostic Reasoning Agent Workflow.
- **Phase 7**: FastAPI Backend & Next.js Interactive Field Engineer UI.
