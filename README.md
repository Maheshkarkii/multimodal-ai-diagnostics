# AI Field Engineer - Multimodal Autonomous Troubleshooting & Diagnosis System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, diagnosis, and troubleshooting across images, audio, sensor telemetry, and technical documentation.

---

## 🔍 Phase 8 — Explainability, Evidence & Auditable Diagnostic Reports

Phase 8 elevates the system to full **industrial auditability and explainability**. Every diagnostic decision, severity classification, and recommended action can be traced directly to physical sensor telemetry, model saliency heatmaps, acoustic harmonics, and verified OEM technical documentation citations.

### 🎯 The "Why" Layer of Field Diagnosis
> [!IMPORTANT]
> **Observation $\neq$ Interpretation $\neq$ Diagnosis $\neq$ Causal Proof**
>
> - **Observation**: Measured RMS vibration is $6.8\text{ mm/s}$; acoustic signal exhibits BPFI harmonic peaks at $3.2\text{ kHz}$.
> - **Interpretation**: High vibration and BPFI harmonics are consistent with rolling element bearing degradation per ISO 10816-3.
> - **Hypothesis**: Bearing degradation is the leading operational hypothesis ($86.0\%$ confidence).
> - **Auditable Trace**: Evidence items receive stable identifiers (`SEN-001`, `AUD-001`, `DOC-001`) linking claims directly to primary sources.

---

## 🏗️ Auditable Diagnostic Architecture

```
[Raw Observations (Vision, Audio, Sensors, Notes, RAG)]
                         │
                         ▼
        [Auditable Evidence Normalization Engine]
       (Assigns stable IDs: VIS-xxx, AUD-xxx, SEN-xxx, DOC-xxx)
                         │
                         ▼
       [Claim-to-Evidence Bidirectional Mapper]
   (Links primary diagnosis & severity to justifying evidence)
                         │
                         ▼
       [Confidence Decomposition & Rationale Engine]
   (Decomposes multimodal agreement, sensor margins & penalties)
                         │
                         ▼
       [Traceable Action & Requirement Planner]
      (Assigns REQUIRED/RECOMMENDED levels & citations)
                         │
                         ▼
       [Immutable Audit Trail & Reproducibility Logger]
   (Persists input hashes, model checkpoints, runtime latency)
                         │
                         ▼
      [Professional Auditable Markdown Report & JSON]
```

---

## ⚙️ Core Capabilities

1. **Standardized Evidence Taxonomy & Stable IDs**:
   - `VIS-xxx`: Visual defect and component inspection features.
   - `AUD-xxx`: Acoustic spectrum features and BPFI/BPFO harmonic signatures.
   - `SEN-xxx`: Physical telemetry readings with exact units and ISO threshold margins.
   - `TXT-xxx`: Field technician symptom descriptions.
   - `DOC-xxx`: Verified OEM maintenance manual citations with exact page numbers.

---

## 🚀 Phase 9 & 10 — Production API, Docker & CI/CD Deployment

The AI Field Engineer system is fully containerized, reproducible, and deployment-ready via FastAPI, multi-stage Docker builds, Docker Compose, and automated GitHub Actions CI/CD pipelines.

### 🏛️ Complete Deployment Architecture

```
                       Internet / Client (Field Engineer / Mobile / Web)
                                              │
                                              ▼
                                 [FastAPI Production Service]
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                │                             │                             │
                ▼                             ▼                             ▼
       [Vision Service]                [Audio Service]              [Sensor Service]
       (ResNet Backbone)              (1D-CNN / STFT)               (MLP / Telemetry)
                │                             │                             │
                └─────────────────────────────┼─────────────────────────────┘
                                              ▼
                                 [Multimodal Fusion Layer]
                               (Cross-Attention & Gating)
                                              │
                                              ▼
                                [Technical Knowledge RAG]
                                (Dense + BM25 Hybrid)
                                              │
                                              ▼
                                 [Diagnostic Reasoning Agent]
                                (Multi-Stage Autonomous Loop)
                                              │
                                              ▼
                                [Explainability & Audit Layer]
                               (Grad-CAM, FFT & Audit JSON)
                                              │
                                              ▼
                                 [Structured JSON & Reports]
```

---

## 🛠️ Production Setup & Deployment Guide

### Prerequisites
- **Python**: `3.10` to `3.13` (Production Docker image uses `Python 3.11-slim`)
- **Docker**: `20.10+` with Buildx support
- **Docker Compose**: `2.0+` (optional for local multi-volume stack)

### 1. Environment Configuration
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Key environment variables:
| Variable | Description | Default |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Runtime mode (`development`, `staging`, `production`) | `production` |
| `API_HOST` | Host address to bind inside container | `0.0.0.0` |
| `API_PORT` | Listening port for FastAPI | `8000` |
| `API_WORKERS` | Uvicorn worker processes | `1` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:3000,http://127.0.0.1:3000` |
| `TEMP_UPLOAD_DIR` | Sandboxed temporary directory for uploads | `/tmp/ai-field-engineer/uploads` |
| `VECTOR_STORE_DIR`| Persistent vector database directory | `/app/data/rag/vector_store` |
| `MODEL_DIR` | Trained model weights directory | `/app/models` |
| `LLM_PROVIDER` | Reasoning engine provider (`mock`, `gemini`, `openai`) | `mock` |

### 2. Local Docker Workflow

#### Build the Docker Image
```bash
docker build -t ai-field-engineer-api:latest .
```

#### Run the Production Container
```bash
docker run -d \
  --name ai_field_engineer_api \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8000" \
  -v $(pwd)/data/rag/vector_store:/app/data/rag/vector_store \
  -v $(pwd)/reports:/app/reports \
  ai-field-engineer-api:latest
```

#### Using Docker Compose
```bash
docker compose up -d
```

### 3. Verify Health & Diagnose
Check liveness & readiness:
```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
```

Submit a diagnostic case:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/diagnose \
  -F 'technician_description=High pitch squeal and casing temperature elevated to 88C' \
  -F 'sensor_data={"temperature": 88.0, "vibration": 7.4, "rpm": 1490.0}' \
  -F 'equipment_metadata={"equipment_type": "motor", "manufacturer": "Siemens"}'
```

---

## 🧪 CI/CD & Automated Quality Gates

Every Pull Request and Push to `main` executes the `.github/workflows/ci.yml` pipeline:

```
Push / PR
   │
   ▼
[1. Code Quality] ──> Ruff Linter & Formatter Check + mypy Static Type Checking
   │
   ▼
[2. Automated Tests] ──> Pytest Full Suite (64 Unit/Integration Tests across Phases 1-10)
   │
   ▼
[3. Docker Build & Smoke Test] ──> Multi-stage Docker Build + Container Startup + /ready Probe + Diagnostic Test
```

### Run Local Checks
```bash
# Linting & Formatting
ruff check src/ tests/
ruff format --check src/ tests/

# Type Checking
mypy src/api/ src/rag/config.py src/explainability/core/schema.py

# Test Suite
python -m pytest tests/ -v
```

---

## 🔒 Security Baseline

- **Non-Root Runtime**: Application executes under dedicated user `appuser` (`UID: 10001`).
- **Zero Secrets in Images**: Configuration strictly supplied via environment variables; `.env` is gitignored.
- **Sandboxed Uploads**: Multipart uploads stream through `FileValidationService` with strict MIME and size limits.
- **Production CORS**: Configurable allowed origins; prevents wildcard origins in production environments.

2. **Claim-to-Evidence Bidirectional Audit Trace**:
   - Explicitly links diagnostic claims to supporting and contradicting evidence IDs with verification status (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`).
3. **Multifactorial Confidence Decomposition**:
   - Replaces ungrounded single numbers with decomposed factors: *Multimodal Agreement*, *Sensor Margin*, *Acoustic Evidence*, *Visual Evidence*, and *Contradiction Penalty*.
4. **Action Traceability & Requirement Levels**:
   - Classifies maintenance steps into `REQUIRED` (Safety-critical), `RECOMMENDED` (High yield), and `OPTIONAL`.
5. **Immutable Audit Trail**:
   - Records input cryptographic SHA-256 hashes, model versions, RAG chunk IDs, and execution latency to `reports/audit/audit_<CASE_ID>.json`.

---

## 📊 Benchmark Evaluation Results

Evaluated on the Explainability & Audit benchmark suite (`scripts/evaluate_explainability.py`):

| Evaluation Metric | Value | Meaning |
| :--- | :---: | :--- |
| **Evidence Attribution Rate** | **100.00%** | All observations assigned stable audit identifiers |
| **Claim-to-Evidence Grounding** | **100.00%** | Diagnostic claims anchored in multi-channel observations |
| **Audit Trail Integrity** | **100.00%** | Complete cryptographic input hash & version logging |
| **Prompt Injection Immunity** | **100.00%** | Adversarial override attempts safely treated as passive data |

---

## 🛠️ CLI Execution Commands

### 1. Generate Auditable Diagnostic Report
```bash
python scripts/generate_auditable_report.py --equipment motor --model M-4500 --description "High pitch acoustic squealing from bearing" --vibration 6.8 --temp 84.0 --audio-pred "bearing_defect_wear"
```

### 2. Run Explainability Benchmark Suite
```bash
python scripts/evaluate_explainability.py
```

### 3. Run All Test Suites (17 unit & integration tests)
```bash
pytest tests/test_rag.py tests/test_agent.py tests/test_explainability.py -v
```

---

## 💻 Phase 12 — Field Engineer Web Application & Full System Integration

The complete system includes a modern, responsive Next.js web application designed specifically for field technicians and reliability engineers:

### Key UI Features
- **Intuitive Case Creation**: Input equipment metadata, physical telemetry thresholds, and technician notes.
- **Multimodal File Pickers**: Drag-and-drop support for high-resolution inspection photos and acoustic audio clips.
- **Auditable Diagnostic Dashboards**: View primary diagnoses, decomposed confidence metrics, and evidence trails.
- **Human Review & Feedback**: Integrated loop for recording domain-expert feedback to curate future benchmark datasets.

### Running the Full-Stack Application Locally

#### 1. Start with Docker Compose
```bash
docker compose up --build
```
- **Web UI**: http://localhost:3000
- **FastAPI Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### 2. Local Manual Startup
```bash
# Terminal 1: FastAPI Backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Next.js Frontend
cd frontend
npm install
npm run dev
```

---

## 🏆 Complete 12-Phase System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │       Field Engineer / Technician       │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Next.js 14 Frontend Application    │
                    │   (Responsive UI, Evidence & Feedback)  │
                    └────────────────────┬────────────────────┘
                                         │  (HTTP / Multipart)
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │       FastAPI Production Inference      │
                    │  (Validation, Stream Cleanup, Security) │
                    └────────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
      │   Vision Subsystem  │ │   Audio Subsystem   │ │   Sensor Subsystem  │
      │  (ResNet Backbone)  │ │  (1D-CNN / Mel-STFT)│ │ (MLP & Anomaly Det) │
      └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │        Multimodal Cross-Attention       │
                    │         Feature Fusion Pipeline         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │         Technical Knowledge RAG         │
                    │  (Dense + BM25 Hybrid Retrieval Index)  │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │        Diagnostic Reasoning Agent       │
                    │   (Autonomous Loop & Safety Bounds)     │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Explainability & Evidence Layer    │
                    │   (Grad-CAM, Saliency & Audit Trails)   │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │     Telemetry, Monitoring & Feedback    │
                    │      (Drift Analysis & CI Gates)        │
                    └─────────────────────────────────────────┘
```
