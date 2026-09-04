# AI Field Engineer - Multimodal Autonomous Troubleshooting & Diagnosis System

An industrial-grade, multimodal AI system for automated equipment inspection, fault detection, diagnosis, and troubleshooting across images, audio, sensor telemetry, and technical documentation.

---

## 🧠 Phase 7 — Diagnostic Reasoning Agent

Phase 7 introduces the **Autonomous Diagnostic Reasoning Agent** orchestration layer. The reasoning agent fuses empirical perception models (Vision, Acoustic Audio, Sensor Telemetry) with external OEM engineering manuals (Phase 6 RAG) to formulate structured, transparent, and grounded root-cause diagnoses.

### 🎯 Critical Architectural Separation
> [!IMPORTANT]
> - **PyTorch Perception (Phases 1–5)**: Answers *"What patterns are present?"* (e.g. 1X vibration peak, acoustic squeal, bearing surface defect).
> - **Technical RAG (Phase 6)**: Answers *"What does the technical manual say?"* (e.g. ISO 10816-3 limits, bearing lubrication SOP).
> - **Diagnostic Agent (Phase 7)**: Answers *"Given the multimodal observations and technical manual evidence, what failure hypothesis best explains the machine condition?"*

---

## 🏗️ Diagnostic Reasoning Architecture

```
                FIELD OBSERVATIONS
                       │
      ┌────────────────┼────────────────┐
      │                │                │
    Vision           Audio           Sensors
      │                │                │
      └────────────────┼────────────────┘
                       │
                Multimodal State
                       │
                  Technician Text
                       │
                       ▼
              ┌─────────────────┐
              │ Diagnostic      │
              │ Context Builder │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Reasoning Agent │
              └───────┬─────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
      RAG Retrieval        Evidence Analysis
            │                   │
            └─────────┬─────────┘
                      ▼
              Hypothesis Ranking
                      │
                      ▼
             Contradiction Check
                      │
                      ▼
              Diagnostic Report
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    Diagnosis      Evidence       Actions
```

---

## ⚙️ Core Agent Capabilities

1. **Multi-Stage Bounded Reasoning Pipeline**:
   - **Stage 1**: Sensor Telemetry & ISO 10816-3 evaluation.
   - **Stage 2**: Targeted Technical RAG query formulation.
   - **Stage 3**: Cross-modality contradiction and investigation gap detection.
   - **Stage 4**: Structured LLM hypothesis synthesis and alternative evaluation.
   - **Stage 5**: Groundedness verification against source evidence pool.
   - **Stage 6**: Auditable Diagnostic Report generation with Markdown export.
2. **Strict Evidence Hierarchy & Provenance**:
   - Explicit separation between *Observed Measurements*, *Model Inferences*, and *Retrieved OEM Manual Knowledge*.
   - Zero-fabrication enforcement on page numbers, citations, and sensor readings.
3. **Cross-Modality Contradiction Detection**:
   - Flags discrepancies (e.g. Normal camera image vs. Acoustic BPFI defect harmonic) and automatically penalizes overconfident scores.
4. **Safety-Grounded Action Planning**:
   - Distinguishes safety-critical instructions (e.g. Immediate emergency shutdown) from informational maintenance steps.

---

## 📊 Benchmark Evaluation Results

Evaluated across industrial benchmark test cases in [`scripts/evaluate_agent.py`](file:///C:/Users/Mahesh%20Karki/Downloads/Mahesh/multimodal-ai-diagnostics/scripts/evaluate_agent.py):

| Evaluation Metric | Value | Diagnostic Significance |
| :--- | :---: | :--- |
| **Diagnostic Accuracy** | **100.00%** | Primary root-cause diagnosis matched ground truth across all test cases |
| **Severity Classification** | **100.00%** | Exact alignment with ISO 10816-3 and manual severity zones |
| **Average Evidence Grounding** | **79.17%** | Diagnostic statements rigorously anchored in retrieved manual citations |
| **Contradiction Detection Rate** | **100.00%** | Identified cross-modality conflicts (Vision Normal vs Audio Defect) |

---

## 🛠️ CLI Execution Commands

### 1. Run Diagnostic Agent Benchmark Suite
```bash
python scripts/evaluate_agent.py
```

### 2. Run Autonomous Diagnostic Case Workflow
```bash
python scripts/run_diagnosis.py --equipment motor --model M-4500 --description "Motor emits loud periodic acoustic squealing and housing vibration is severe." --vibration 6.8 --temp 84.0 --audio-pred "bearing_defect_wear"
```

### 3. Run Unit and Integration Tests (13 tests)
```bash
pytest tests/test_rag.py tests/test_agent.py -v
```

---

## 🔮 Roadmap: Next Phases

- **Phase 8**: Explainability, Evidence & Diagnostic Report System — Auditable tracing connecting model activation heatmaps, acoustic spectrograms, and manual citations.
- **Phase 9**: FastAPI Production Backend & Next.js UI.
