# AI Field Engineer ? Multimodal Autonomous Troubleshooting & Diagnosis System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, diagnosis, and troubleshooting across images, audio, sensor telemetry, and technical documentation.

---

## ?? Phase 4 ? Sensor Intelligence & Machine-State Modeling

Phase 4 establishes the PyTorch multivariate telemetry subsystem for continuous machine operational state modeling and unsupervised anomaly detection.

Key capabilities introduced:
- **Leakage-Safe Preprocessing**: Implemented [`SensorPreprocessor`](file:///C:/Users/Mahesh%20Karki/Downloads/Mahesh/multimodal-ai-diagnostics/src/sensor/preprocessing/sensor_scaler.py) ensuring imputation medians and `StandardScaler` parameters are fitted **strictly on the training split** and persisted inside checkpoints to prevent data leakage.
- **Group-Aware Splitting by Asset ID**: Telemetry observations from a given machine (`machine_id`) are completely isolated across training, validation, and test splits.
- **Multivariate Sensor MLP & 256-dim Embedding Extraction**: Built [`SensorMLP`](file:///C:/Users/Mahesh%20Karki/Downloads/Mahesh/multimodal-ai-diagnostics/src/sensor/models/sensor_mlp.py) extracting compact **256-dimensional numerical sensor embeddings** for cross-modal fusion.
- **Dual-Capability Architecture**:
  - **Capability A (State Classification)**: Identifies specific failure operating modes (`bearing_overheat_wear`, `rotor_unbalance`, `hydraulic_pressure_loss`, `electrical_overcurrent`, `normal`).
  - **Capability B (Anomaly Detection & Operating Envelopes)**: Employs `IsolationForest` and $\sigma$-envelope boundaries to flag unprecedented deviations and compute continuous anomaly scores $[0.0 \dots 1.0]$.
- **Permutation Feature Importance**: Quantifies the relative percentage drop in Macro F1 when specific sensor features are perturbed.

---

## ??? Sensor Processing & Architecture

```
Raw Multivariate Sensor Telemetry (Temperature, Vibration, RPM, Current, Pressure, Load)
   ?
   ?
[SensorDataValidator] ????????? Schema verification, missing-value check, machine ID validation
   ?
   ?
[Leakage-Safe Preprocessor] ??? Median imputation & StandardScaler (Fitted strictly on Train partition)
   ?
   ???????????????????????????? [SensorAnomalyDetector] ??? Isolation Forest + Normal Envelope -> Anomaly Score [0..1]
   ?
   ?
[PyTorch SensorMLP Backbone] ?? [Linear(6->128) -> BN -> ReLU -> Dropout] -> [Linear(128->256) -> BN -> ReLU]
   ?
   ???????????????????????????? [256-dim Sensor Feature Embedding] ??? (Reserved for Multimodal Fusion)
   ?
   ?
[Classification Head] ????????? Dropout(0.2) + Linear(256 -> 5 Machine State Classes)
   ?
   ?
[Softmax & Confidence Metric] ?? Predicted Machine Operating State, Confidence Score, Ranked Candidates
```

---

## ?? Sensor Diagnostic Benchmark Results

Evaluated on the 5-class multivariate telemetry benchmark across unseen industrial machine assets:

- **Test Classification Accuracy**: **100.0%**
- **Macro F1-Score**: **1.0000**
- **Weighted F1-Score**: **1.0000**
- **Permutation Feature Importance (Relative Impact)**:
  - `vibration_rms_g`: **96.27%**
  - `rotational_speed_rpm`: **3.73%**
- **Anomaly Detection Calibration**:
  - Normal sample anomaly score: **$0.211$**
  - High-temperature/vibration anomalous sample: **$0.784$** (Envelope deviation: $>5.4\sigma$)

---

## ?? Scientific & Engineering Distinction

> [!WARNING]
> **Anomaly $\neq$ Physical Fault & Sensor State $\neq$ Root-Cause Diagnosis**
>
> An anomalous telemetry reading can stem from sensor drift, intermittent load spikes, or ambient temperature fluctuations without constituting physical hardware damage. True root-cause troubleshooting requires synthesizing sensor telemetry with acoustic harmonics (Phase 3) and visual surface inspection (Phase 2).

---

## ?? CLI Execution Commands

### 1. Run All Test Suites
```bash
pytest -v
```

### 2. Train Sensor State & Anomaly Model
```bash
python scripts/train_sensor.py --config configs/sensor.yaml
```

### 3. Evaluate Model on Test Machines & Run Error Analysis
```bash
python scripts/evaluate_sensor.py --config configs/sensor.yaml --checkpoint checkpoints/sensor_state_and_anomaly_baseline_best.pt
```

### 4. Run Sensor Inference with 256-dim Embedding Extraction
```bash
python scripts/inference_sensor.py --json-input '{"temperature_c": 96.5, "vibration_rms_g": 4.8, "rotational_speed_rpm": 1485.0, "motor_current_a": 10.4, "hydraulic_pressure_bar": 141.0, "load_percentage": 70.0}' --extract-embedding
```

---

## ?? Roadmap: Next Phases

- **Phase 5**: Multimodal Fusion Engine ? Joint cross-attention module fusing Vision (1280-dim), Audio (512-dim), and Sensor (256-dim) feature embeddings.
- **Phase 6**: Technical-Document RAG & Agentic Diagnostic Reasoning Workflow.
- **Phase 7**: FastAPI Backend & Next.js UI.
