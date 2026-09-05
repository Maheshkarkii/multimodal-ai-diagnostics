"""
Unit and Integration Tests for Phase 7 Diagnostic Reasoning Agent.
"""

import tempfile

import pytest

from src.agent.core.agent import DiagnosticReasoningAgent
from src.agent.core.schema import (
    ModalityObservation,
    ModalityType,
    SeverityLevel,
)
from src.agent.tools.tools import (
    ISOVibrationStandardTool,
    SensorStateInspectionTool,
)
from src.agent.validation.groundedness import GroundednessChecker
from src.rag.config import RetrievalConfig, VectorStoreConfig
from src.rag.embeddings.model import DeterministicDenseEmbeddingModel
from src.rag.retrieval.retriever import TechnicalRetriever
from src.rag.schema import DocumentChunk
from src.rag.vectorstore.store import NumpyFlatVectorStore


@pytest.fixture
def mock_retriever():
    d = tempfile.mkdtemp()
    store = NumpyFlatVectorStore(VectorStoreConfig(persist_directory=d))
    emb = DeterministicDenseEmbeddingModel(embedding_dim=64)

    chunk = DocumentChunk.create(
        document_id="doc_m4500",
        document_name="motor_m4500_maintenance_manual.pdf",
        source_path="motor_m4500_maintenance_manual.pdf",
        page_number=2,
        text="SECTION 2: BEARING INSPECTION. Perform ultrasound check and grease discoloration test.",
        chunk_index=0,
        section="BEARING INSPECTION",
        equipment_type="motor",
    )
    store.add_chunks([chunk], emb.embed_documents([chunk.text]))
    return TechnicalRetriever(store, emb, RetrievalConfig(similarity_threshold=0.01))


def test_sensor_state_inspection_and_iso_tool():
    sensor_tool = SensorStateInspectionTool()
    iso_tool = ISOVibrationStandardTool()

    readings = [
        {
            "parameter": "Vibration_RMS",
            "value": 6.8,
            "unit": "mm/s",
            "warning_threshold": 4.5,
            "critical_threshold": 7.1,
        },
        {"parameter": "Bearing_Temp", "value": 85.0, "unit": "degC", "normal_max": 70.0, "critical_threshold": 90.0},
    ]

    results = sensor_tool.execute(readings)
    assert len(results) == 2
    assert results[0].status == "WARNING"
    assert results[0].is_anomaly is True

    iso_res = iso_tool.execute(6.8)
    assert "Zone C" in iso_res["iso_zone"]
    assert iso_res["severity"] == "HIGH"


def test_groundedness_checker():
    checker = GroundednessChecker(groundedness_threshold=0.50)

    from src.agent.core.schema import DiagnosticEvidenceItem, EvidenceType

    evidence = [
        DiagnosticEvidenceItem(
            evidence_id="ev1",
            evidence_type=EvidenceType.RETRIEVED_KNOWLEDGE,
            source="motor_manual.pdf (Page 2)",
            statement="Bearing defect causes high BPFI squeal and elevated vibration.",
        )
    ]

    score, unsupported = checker.evaluate_groundedness(
        primary_diagnosis="bearing_defect_wear",
        supporting_statements=["Bearing defect causes high BPFI squeal and elevated vibration."],
        recommended_actions=[{"action_text": "Inspect bearing BPFI squeal", "source_reference": "motor_manual.pdf"}],
        cited_references=["motor_manual.pdf"],
        available_evidence=evidence,
    )
    assert score >= 0.80
    assert len(unsupported) == 0


def test_cross_modality_contradiction_detection(mock_retriever):
    agent = DiagnosticReasoningAgent(mock_retriever)

    obs = {
        "vision": ModalityObservation(
            modality=ModalityType.VISION,
            prediction="normal_state",
            confidence=0.92,
        ),
        "audio": ModalityObservation(
            modality=ModalityType.AUDIO,
            prediction="bearing_defect_wear",
            confidence=0.88,
        ),
    }

    report = agent.diagnose_case(
        equipment_type="motor",
        technician_description="Motor vibrating",
        observations=obs,
    )

    assert len(report.contradictions_detected) >= 1
    assert "conflict_description" in report.contradictions_detected[0]


def test_end_to_end_diagnostic_workflow(mock_retriever):
    agent = DiagnosticReasoningAgent(mock_retriever)

    obs = {
        "audio": ModalityObservation(
            modality=ModalityType.AUDIO,
            prediction="bearing_defect_wear",
            confidence=0.91,
        )
    }
    sensors = [
        {
            "parameter": "Vibration_RMS",
            "value": 6.8,
            "unit": "mm/s",
            "warning_threshold": 4.5,
            "critical_threshold": 7.1,
        }
    ]

    report = agent.diagnose_case(
        case_id="TEST_CASE_001",
        equipment_type="motor",
        equipment_model="M-4500",
        technician_description="High-frequency squealing noise from drive end bearing.",
        observations=obs,
        sensor_data=sensors,
    )

    assert report.case_id == "TEST_CASE_001"
    assert report.primary_diagnosis == "bearing_defect_wear"
    assert report.severity == SeverityLevel.HIGH
    assert report.diagnostic_confidence > 0.70
    assert len(report.recommended_actions) >= 1
    assert len(report.technical_references) >= 1

    # Verify Markdown rendering
    md = report.to_markdown()
    assert "# AI FIELD ENGINEER -- DIAGNOSTIC ASSESSMENT REPORT" in md
    assert "BEARING_DEFECT_WEAR" in md
    assert "SAFETY CRITICAL" in md
