# Technical Interview Guide — AI Field Engineer

This document provides concise technical explanations and architectural rationales for engineering discussions and technical interviews.

---

### Q1: Why did you choose Cross-Attention Multimodal Fusion over simple feature concatenation?
**Answer**:
Direct concatenation assumes a fixed static relationship between modalities and fails when one or more inputs are missing or noisy. Cross-attention with modality presence gating allows dynamic weighting: if acoustic harmonic signatures clearly exhibit bearing defect peaks (BPFI/BPFO) while the visual inspection is ambiguous, the attention mechanism dynamically upweights the acoustic embedding relative to other modalities.

---

### Q2: How is Data Leakage prevented across Train, Validation, and Test splits?
**Answer**:
Correlated sensor telemetry, audio recordings, and video frames from the same physical machine or duty session are strictly grouped. In `src/evaluation/schemas.py` and `src/core/data_split.py`, splitting is enforced by `machine_id` and `session_id`, ensuring no operational noise or machine-specific signatures from the test machine are leaked into training folds.

---

### Q3: How do you prevent LLM Hallucinations in Maintenance Recommendations?
**Answer**:
1. **Tool-Mediated Context**: The LLM is never allowed to guess engineering tolerances; it must retrieve chunks from verified OEM manuals via hybrid Dense + BM25 search.
2. **Schema-Constrained Outputs**: The reasoning agent emits structured Pydantic models validated at runtime.
3. **Groundedness Checking**: Generated claims are audited against retrieved chunks. Unanchored claims are rejected or flagged, reverting the system to an explicit `INSUFFICIENT_EVIDENCE` or `REQUIRES_HUMAN_INSPECTION` state.

---

### Q4: How is Model Confidence distinguished from Diagnostic Certainty?
**Answer**:
- **Model Confidence**: Softmax output probability or anomaly score from individual neural networks.
- **Diagnostic Certainty**: Multifactorial decomposition incorporating *Multimodal Agreement*, *Sensor Threshold Margin (ISO 10816-3)*, *Acoustic Harmonics*, *Evidence Completeness*, and *Contradiction Penalties*.

---

### Q5: How is Latency managed for Real-Time Edge or Field Inspection?
**Answer**:
- Pre-indexed flat Numpy/vector stores load in $< 15\text{ ms}$.
- Feature extractors use lightweight 1D-CNNs and optimized ResNet backbones.
- Full end-to-end multimodal inference executes on standard multi-core CPUs in $< 90\text{ ms}$ without requiring active internet connectivity.
