# AI Field Engineer — Industrial Risk Register & Mitigation Strategy

This document outlines the technical, operational, and safety risks associated with autonomous multimodal diagnostics and their implemented software mitigations.

---

## 1. Risk Matrix

| Risk Category | Identified Risk | Severity | Likelihood | Implemented Mitigation | Remaining Limitation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Safety** | Erroneous diagnosis leading to premature return-to-service | HIGH | LOW | High vibration (>7.1 mm/s ISO limit) unconditionally triggers `CRITICAL` state requiring manual lockout/tagout. | Edge cases with sensor drift require periodic physical calibration. |
| **ML / Inference** | Hallucination of unobserved machine defect | HIGH | LOW | `GroundednessChecker` validates claims against input sensor IDs (`SEN-xxx`) and manual citations (`DOC-xxx`). | Requires comprehensive initial knowledge base ingestion. |
| **Data / Telemetry** | Corrupted or noisy sensor telemetry | MEDIUM | MEDIUM | Multi-modality cross-attention downweights uncorroborated single-channel anomalies; detects data conflict. | Extreme broadband noise may reduce overall confidence. |
| **RAG / Knowledge** | Outdated OEM maintenance manual index | MEDIUM | LOW | Vector store maintains document version hashes and chunk provenance tracking. | Requires engineering re-index upon manual revision. |
| **Security** | Prompt injection attack via technician text notes | HIGH | LOW | Input sanitization treats technician notes strictly as passive symptom data; system prompt instructions cannot be overridden. | Complex obfuscated text requires continuous pattern defense. |
| **Operational** | Network outage in remote industrial plant | MEDIUM | LOW | Standalone local Docker container executes all inference on local CPU/GPU without cloud roundtrips. | Model updates require local container redeployment. |
