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

## 🔮 Roadmap: Next Phases

- **Phase 9**: Production API & Inference Service — FastAPI backend with structured Pydantic schemas, asynchronous job processing, and health monitoring endpoints.
- **Phase 10**: Next.js Production Web UI & Field Deployment Dashboard.
