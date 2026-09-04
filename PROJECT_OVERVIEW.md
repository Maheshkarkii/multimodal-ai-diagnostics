# AI Field Engineer — Architectural Design & Engineering Overview

## 1. Problem Statement & Motivation
Industrial rotating equipment (motors, pumps, gearboxes, compressors) failure accounts for billions of dollars annually in unplanned downtime, secondary damage, and worker safety risks. 

Traditional troubleshooting relies either on:
1. **Isolated single-modality models**: A vibration threshold model or an acoustic anomaly detector that operates in a silo and cannot explain root causes.
2. **Generic Large Language Models (LLMs)**: Chat-based models that hallucinate maintenance tolerances, lack physical sensory perception, and have no direct access to verified OEM documentation.

**AI Field Engineer** solves this challenge through a multi-tier autonomous diagnostic pipeline combining real-time PyTorch sensory perception, hybrid dense-sparse RAG knowledge retrieval, and evidence-grounded reasoning.

---

## 2. Core Architectural Pillars

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

---

## 3. Engineering & ML Challenges Solved

### A. Missing-Modality Robustness
Field diagnostics rarely provide complete input streams simultaneously. The Cross-Attention fusion layer incorporates structured modality dropout masks and modality-presence gating, enabling inference on arbitrary combinations of available inputs without degradation.

### B. Scientific Grounding vs. Hallucination Prevention
Diagnostic claims are validated by a `GroundednessChecker` before report synthesis. Every factual claim must be backed by a verified `DOC-xxx` (OEM manual citation) or `SEN-xxx`/`AUD-xxx`/`VIS-xxx` (observed telemetry) identifier. Unsupported assertions trigger safe abstention states (`INSUFFICIENT_EVIDENCE` / `REQUIRES_HUMAN_INSPECTION`).

### C. Industrial Auditability
Every diagnostic execution computes SHA-256 hashes of input streams, logs exact model checkpoint versions, records RAG retrieval chunk IDs, and persists immutable audit JSON trails in `reports/audit/`.

---

## 4. Operational Boundaries
- **Assisted Decision Support**: AI Field Engineer is an autonomous decision-support system designed to assist qualified engineers; it does not replace mandated safety procedures or on-site manual lock-out/tag-out (LOTO) protocols.
- **Offline & Edge Capability**: The core pipeline operates entirely locally on CPU/GPU hardware with zero external cloud dependencies.
