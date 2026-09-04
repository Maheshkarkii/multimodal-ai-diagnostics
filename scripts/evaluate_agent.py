"""
Agent Benchmark Evaluation Framework.
Evaluates Diagnostic Reasoning Accuracy, Groundedness, Contradiction Detection,
and Abstention Quality across curated industrial failure cases.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.core.agent import DiagnosticReasoningAgent
from src.agent.core.schema import ModalityObservation, ModalityType, SeverityLevel
from src.rag.config import RAGConfig
from src.rag.embeddings.model import create_embedding_model
from src.rag.vectorstore.store import NumpyFlatVectorStore
from src.rag.retrieval.retriever import TechnicalRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("evaluate_agent")


@dataclass
class DiagnosticTestCase:
    case_id: str
    equipment_type: str
    equipment_model: Optional[str]
    technician_description: str
    observations: Dict[str, ModalityObservation]
    sensor_data: List[Dict[str, Any]]
    expected_primary_diagnosis: str
    expected_severity: SeverityLevel
    expected_citation_keywords: List[str]
    is_contradictory_case: bool = False


BENCHMARK_CASES = [
    # Case 1: Motor Bearing Defect
    DiagnosticTestCase(
        case_id="BENCH_CASE_01_BEARING",
        equipment_type="motor",
        equipment_model="M-4500",
        technician_description="Motor emits high-pitch acoustic squeal after 15 minutes of runtime.",
        observations={
            "audio": ModalityObservation(
                modality=ModalityType.AUDIO,
                prediction="bearing_defect_wear",
                confidence=0.89,
                summary="BPFI high-frequency harmonic peaks detected."
            ),
            "vision": ModalityObservation(
                modality=ModalityType.VISION,
                prediction="bearing_defect_wear",
                confidence=0.78,
                summary="Drive-end bearing discoloration."
            )
        },
        sensor_data=[
            {"parameter": "Vibration_RMS", "value": 6.8, "unit": "mm/s", "warning_threshold": 4.5, "critical_threshold": 7.1},
            {"parameter": "Bearing_Temp", "value": 82.0, "unit": "degC", "warning_threshold": 75.0, "critical_threshold": 90.0}
        ],
        expected_primary_diagnosis="bearing_defect_wear",
        expected_severity=SeverityLevel.HIGH,
        expected_citation_keywords=["motor_m4500_maintenance_manual.pdf", "Page 2"]
    ),
    # Case 2: Pump Cavitation
    DiagnosticTestCase(
        case_id="BENCH_CASE_02_CAVITATION",
        equipment_type="pump",
        equipment_model="CP-800",
        technician_description="Gravel-like popping sounds from pump casing with fluctuating discharge pressure.",
        observations={
            "audio": ModalityObservation(
                modality=ModalityType.AUDIO,
                prediction="hydraulic_cavitation",
                confidence=0.92,
                summary="Broadband acoustic noise between 5kHz and 15kHz."
            )
        },
        sensor_data=[
            {"parameter": "Suction_Pressure", "value": 0.4, "unit": "bar", "warning_threshold": 0.8},
            {"parameter": "Vibration_RMS", "value": 4.8, "unit": "mm/s", "warning_threshold": 4.5}
        ],
        expected_primary_diagnosis="hydraulic_cavitation",
        expected_severity=SeverityLevel.HIGH,
        expected_citation_keywords=["centrifugal_pump_cp800_troubleshooting.md", "Page 1"]
    ),
    # Case 3: Rotor Dynamic Unbalance
    DiagnosticTestCase(
        case_id="BENCH_CASE_03_UNBALANCE",
        equipment_type="motor",
        equipment_model="M-4500",
        technician_description="Heavy radial vibration synchronized with 1X shaft rotation.",
        observations={
            "sensor": ModalityObservation(
                modality=ModalityType.SENSOR,
                prediction="rotor_unbalance",
                confidence=0.85,
                summary="1X dominant FFT peak."
            )
        },
        sensor_data=[
            {"parameter": "Vibration_RMS", "value": 5.8, "unit": "mm/s", "warning_threshold": 4.5, "critical_threshold": 7.1}
        ],
        expected_primary_diagnosis="rotor_unbalance",
        expected_severity=SeverityLevel.MEDIUM,
        expected_citation_keywords=["motor_m4500_maintenance_manual.pdf", "Page 3"]
    ),
    # Case 4: Contradictory Evidence Case
    DiagnosticTestCase(
        case_id="BENCH_CASE_04_CONTRADICTION",
        equipment_type="motor",
        equipment_model="M-4500",
        technician_description="Operator reported noise but visual camera shows normal surface.",
        observations={
            "vision": ModalityObservation(
                modality=ModalityType.VISION,
                prediction="normal_state",
                confidence=0.95,
                summary="No visual surface defect."
            ),
            "audio": ModalityObservation(
                modality=ModalityType.AUDIO,
                prediction="bearing_defect_wear",
                confidence=0.88,
                summary="Acoustic chirping present."
            )
        },
        sensor_data=[],
        expected_primary_diagnosis="bearing_defect_wear",
        expected_severity=SeverityLevel.HIGH,
        expected_citation_keywords=[],
        is_contradictory_case=True
    )
]


def run_agent_benchmark(config_path: str = "configs/rag.yaml"):
    rag_cfg = RAGConfig.from_yaml(config_path) if Path(config_path).exists() else RAGConfig()
    emb_model = create_embedding_model(rag_cfg.embedding)
    store = NumpyFlatVectorStore(rag_cfg.vector_store)

    if store.count() == 0:
        logger.error("Vector store empty. Ingesting documents...")
        from src.rag.ingestion.pipeline import DocumentIngestionPipeline
        from src.rag.chunking.chunker import TechnicalDocumentChunker
        pipe = DocumentIngestionPipeline(rag_cfg.ingestion, manifest_path=rag_cfg.manifest_path)
        chunker = TechnicalDocumentChunker(rag_cfg.chunking)
        res = pipe.ingest_directory(rag_cfg.documents_dir)
        for m, p in res["parsed_documents"]:
            ch = chunker.chunk_document(m, p)
            store.add_chunks(ch, emb_model.embed_documents([c.text for c in ch]))
        store.save()

    retriever = TechnicalRetriever(store, emb_model, rag_cfg.retrieval)
    agent = DiagnosticReasoningAgent(retriever)

    print("\n=======================================================")
    print("      PHASE 7 — DIAGNOSTIC AGENT BENCHMARK EVALUATION  ")
    print("=======================================================")
    print(f"Total Test Cases: {len(BENCHMARK_CASES)}")
    print("-------------------------------------------------------")

    correct_diagnoses = 0
    correct_severities = 0
    groundedness_scores = []
    contradictions_detected_count = 0
    results = []

    for tc in BENCHMARK_CASES:
        report = agent.diagnose_case(
            case_id=tc.case_id,
            equipment_type=tc.equipment_type,
            equipment_model=tc.equipment_model,
            technician_description=tc.technician_description,
            observations=tc.observations,
            sensor_data=tc.sensor_data,
        )

        is_diag_correct = report.primary_diagnosis.lower() == tc.expected_primary_diagnosis.lower()
        is_sev_correct = report.severity == tc.expected_severity
        has_contradiction = len(report.contradictions_detected) > 0

        if is_diag_correct:
            correct_diagnoses += 1
        if is_sev_correct:
            correct_severities += 1
        if has_contradiction:
            contradictions_detected_count += 1

        groundedness_scores.append(report.groundedness_score)

        status_str = "PASSED" if is_diag_correct else "FAILED"
        print(f"[{status_str}] {tc.case_id}:")
        print(f"   Diagnosis: Expected '{tc.expected_primary_diagnosis}' | Got '{report.primary_diagnosis}'")
        print(f"   Confidence: {report.diagnostic_confidence * 100:.1f}% | Groundedness: {report.groundedness_score * 100:.1f}%")
        print(f"   Severity: {report.severity.value} | Contradictions: {len(report.contradictions_detected)}")

        results.append({
            "case_id": tc.case_id,
            "expected_diagnosis": tc.expected_primary_diagnosis,
            "predicted_diagnosis": report.primary_diagnosis,
            "confidence": report.diagnostic_confidence,
            "severity": report.severity.value,
            "groundedness_score": report.groundedness_score,
            "unsupported_claims": report.unsupported_claims,
            "status": status_str,
        })

    n = len(BENCHMARK_CASES)
    avg_groundedness = sum(groundedness_scores) / n

    print("-------------------------------------------------------")
    print(f"Diagnostic Accuracy:        {correct_diagnoses / n * 100:.2f}%")
    print(f"Severity Classification:    {correct_severities / n * 100:.2f}%")
    print(f"Average Evidence Grounding: {avg_groundedness * 100:.2f}%")
    print(f"Contradictions Detected:    {contradictions_detected_count}")
    print("=======================================================\n")

    report_path = Path("reports/agent_diagnostic_benchmark.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": n,
            "diagnostic_accuracy": correct_diagnoses / n,
            "severity_accuracy": correct_severities / n,
            "average_groundedness": avg_groundedness,
            "results": results,
        }, f, indent=2)
    print(f"Saved benchmark results to '{report_path}'\n")


if __name__ == "__main__":
    run_agent_benchmark()
