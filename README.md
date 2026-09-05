# AI Field Engineer - Multimodal Autonomous Diagnostics System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, root-cause diagnosis, and troubleshooting across **visual images, acoustic audio, sensor telemetry, and technical OEM documentation**.

---

## 🌟 Overview & Key Capabilities

The **AI Field Engineer** operates as an autonomous diagnostic assistant for reliability engineers and maintenance technicians:

- 👁️ **Visual Inspection**: Identifies surface defects, cracks, corrosion, and wear patterns using deep computer vision models with Grad-CAM spatial heatmaps.
- 🔊 **Acoustic Analysis**: Detects bearing defects, cavitation, and gear mesh faults from audio recordings via Fast Fourier Transform (FFT) spectrograms and harmonic peak detection.
- 📊 **Telemetry & Sensor Monitoring**: Evaluates real-time vibration, temperature, and speed against ISO 10816 standards with dynamic margin analysis.
- 📚 **OEM Knowledge Retrieval (RAG)**: Retrieves authoritative maintenance guidelines and repair procedures using hybrid Dense + BM25 search over technical manuals.
- 🧠 **Autonomous Reasoning Agent**: Synthesizes cross-modal observations into ranked diagnostic hypotheses and safety-prioritized maintenance actions.
- 🛡️ **Auditable Evidence Layer**: Assigns immutable evidence IDs (`VIS-xxx`, `AUD-xxx`, `SEN-xxx`, `DOC-xxx`) to every observation, ensuring full traceability and zero ungrounded claims.

---

## 🏛️ System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │       Field Engineer / Technician       │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Next.js 14 Web Application         │
                    │  (Interactive UI, Evidence & Feedback)  │
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
      │  (ResNet Backbone)  │ │ (1D-CNN / Mel-STFT) │ │ (MLP & Anomaly Det) │
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
                    │      Structured Auditable Reports       │
                    │      (JSON Data + Markdown Export)      │
                    └─────────────────────────────────────────┘
```

---

## 🚀 How to Clone and Run on Any Device

### 📋 Prerequisites

Before starting, ensure you have the following installed on your machine:

- **Git**: [Download Git](https://git-scm.com/)
- **Python**: `3.10` to `3.13` — [Download Python](https://www.python.org/)
- **Node.js**: `18.x` or higher — [Download Node.js](https://nodejs.org/)
- *(Optional)* **Docker & Docker Compose**: For containerized deployment — [Download Docker](https://www.docker.com/)

---

### Step 1: Clone the Repository

Open your terminal (PowerShell, Command Prompt, or Bash) and run:

```bash
git clone https://github.com/Maheshkarkii/multimodal-ai-diagnostics.git
cd multimodal-ai-diagnostics
```

---

### Step 2: Configure Environment Variables

Copy the example environment file:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On Linux / macOS:**
```bash
cp .env.example .env
```

---

### Step 3: Set Up and Run the Backend (FastAPI)

1. **Create and activate a virtual environment:**

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux / macOS:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend server:**
   ```bash
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

   - **Backend API**: [http://localhost:8000](http://localhost:8000)
   - **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

### Step 4: Set Up and Run the Frontend (Next.js)

Open a **new terminal window** in the project directory:

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the Next.js development server:**
   ```bash
   npm run dev
   ```

4. **Access the Web Dashboard:**
   Open your browser and navigate to **[http://localhost:3000](http://localhost:3000)**.

---

### 🐳 Alternative: 1-Step Run with Docker Compose

If you have Docker installed, you can launch both backend and frontend with a single command:

```bash
docker compose up --build -d
```

- **Frontend UI**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Backend**: [http://localhost:8000](http://localhost:8000)
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To stop the containers:
```bash
docker compose down
```

---

## 💻 Operating the System

### 1. Using the Web Dashboard
1. Open [http://localhost:3000](http://localhost:3000) in your browser.
2. Fill in the **Equipment Details** (Equipment Type, Manufacturer, Model).
3. Enter **Sensor Telemetry** (Temperature, Vibration RMS, Speed/RPM).
4. *(Optional)* Upload an inspection image (.png/.jpg) and acoustic audio sample (.wav).
5. Add technician notes and click **Run Diagnosis**.
6. Inspect the generated **Confidence Breakdown**, **Evidence Trail**, and **Recommended Actions**.

### 2. Using the Command Line (CLI)

Generate an auditable diagnostic report directly from the CLI:

```bash
python scripts/generate_auditable_report.py \
  --equipment motor \
  --model M-4500 \
  --description "High pitch acoustic squealing from bearing" \
  --vibration 6.8 \
  --temp 84.0 \
  --audio-pred "bearing_defect_wear"
```

Output reports are saved to `reports/diagnostics/` and audit logs to `reports/audit/`.

---

## 🧪 Testing & Verification

Run the comprehensive test suite and benchmarks:

```bash
# Run all unit and integration tests
pytest tests/ -v

# Run the complete diagnostic evaluation suite
python scripts/evaluate_all_phases.py

# Run the explainability & audit benchmark
python scripts/evaluate_explainability.py
```

---

## 📊 Benchmark Results

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Evidence Attribution Rate** | **100.0%** | Every observation receives a unique, stable audit ID |
| **Claim-to-Evidence Grounding** | **100.0%** | All diagnostic claims directly supported by physical evidence |
| **Audit Trail Integrity** | **100.0%** | Cryptographic input hashing and model provenance logging |
| **Prompt Injection Immunity** | **100.0%** | Technician inputs sanitized to prevent adversarial prompt overrides |

---

## 📁 Repository Structure

```
multimodal-ai-diagnostics/
├── configs/               # Model and pipeline configuration YAMLs
├── data/
│   ├── raw/               # Raw sample datasets (images, audio, telemetry)
│   └── rag/               # Technical OEM documentation manuals & vector store
├── frontend/              # Next.js 14 Web Application
│   ├── src/app/           # React dashboard & diagnosis components
│   └── package.json       # Frontend dependencies
├── reports/               # Generated audit JSONs, markdown reports & saliency maps
├── scripts/               # CLI evaluation, inference, and training scripts
├── src/
│   ├── agent/             # Autonomous diagnostic reasoning agent loop
│   ├── api/               # FastAPI REST service and endpoints
│   ├── explainability/    # Grad-CAM, FFT spectrograms, and evidence loggers
│   ├── models/            # Vision, Audio, Sensor & Multimodal Fusion models
│   └── rag/               # Hybrid Dense + BM25 document retrieval engine
├── tests/                 # Unit, integration, and security test suites
├── Dockerfile             # Multi-stage production container build
├── docker-compose.yml     # Multi-container orchestration
└── requirements.txt       # Python backend dependencies
```

---

## 🔒 Security Baseline

- **Non-Root Execution**: Container runs under a dedicated unprivileged user (`appuser`).
- **Input Validation**: Strict schema checking and sandboxed file upload validation.
- **Zero Secrets in Code**: Environment configuration managed through `.env` files.
- **Auditable Provenance**: Complete cryptographic SHA-256 traceability for all inputs.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

