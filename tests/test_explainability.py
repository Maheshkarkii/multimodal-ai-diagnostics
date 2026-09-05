"""
Unit and Integration Tests for Phase 8 Explainability & Auditable Reporting Layer.
"""

import tempfile
from pathlib import Path

import pytest

from src.agent.core.schema import (
    DiagnosticReport,
    SeverityLevel,
)
from src.explainability.core.config import AuditConfig, ExplainabilityConfig
from src.explainability.core.schema import (
    ActionRequirement,
    AuditableEvidenceItem,
    DiagnosticSystemStatus,
    EvidenceCategory,
    EvidenceQuality,
)
from src.explainability.core.service import ExplainabilityService


@pytest.fixture
def sample_diagnostic_report():
    return DiagnosticReport(
        case_id="TEST_AUDIT_001",
        timestamp="2026-09-04 20:00:00",
        equipment={"type": "motor", "model": "M-4500"},
        problem_summary="Periodic bearing squeal and high RMS vibration",
        available_modalities=["audio", "sensor"],
        primary_diagnosis="bearing_defect_wear",
        diagnostic_confidence=0.86,
        severity=SeverityLevel.HIGH,
        alternative_hypotheses=[{"failure_mode": "rotor_unbalance", "likelihood_score": 0.35}],
        supporting_evidence=[
            {"evidence_type": "SUPPORTING", "statement": "Acoustic squeal matches BPFI.", "source": "Acoustic CNN"}
        ],
        contradicting_evidence=[],
        contradictions_detected=[],
        missing_information=["Visual inspection image missing."],
        recommended_actions=[
            {
                "action_text": "Perform acoustic ultrasound check.",
                "rationale": "Verify BPFI impact spikes.",
                "is_safety_critical": True,
                "source_reference": "motor_manual.pdf (Page 2)",
            }
        ],
        technical_references=["motor_m4500_maintenance_manual.pdf (Page 2)"],
        groundedness_score=1.0,
        unsupported_claims=[],
        status="completed",
    )


def test_evidence_item_creation_and_serialization():
    item = AuditableEvidenceItem(
        evidence_id="SEN-001",
        category=EvidenceCategory.SENSOR,
        source="Vibration Accelerometer",
        description="Measured RMS vibration",
        quality=EvidenceQuality.HIGH,
        raw_value=6.8,
        unit="mm/s",
    )
    assert item.evidence_id == "SEN-001"
    assert item.raw_value == 6.8
    d = item.to_dict()
    assert d["category"] == "SENSOR"
    assert d["quality"] == "HIGH"


def test_explainability_service_report_generation(sample_diagnostic_report):
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = ExplainabilityConfig(
            reports_output_dir=temp_dir,
            audit=AuditConfig(audit_storage_dir=temp_dir),
        )
        service = ExplainabilityService(cfg)
        auditable_report = service.generate_auditable_report(sample_diagnostic_report)

        assert auditable_report.case_id == "TEST_AUDIT_001"
        assert auditable_report.system_status == DiagnosticSystemStatus.DIAGNOSIS_SUPPORTED
        assert len(auditable_report.evidence_inventory) >= 1
        assert len(auditable_report.claim_mappings) >= 1

        ev_ids = [e.evidence_id for e in auditable_report.evidence_inventory]
        assert any(e_id.startswith("TXT-") or e_id.startswith("DOC-") for e_id in ev_ids)

        assert len(auditable_report.recommended_actions) == 1
        act = auditable_report.recommended_actions[0]
        assert act.requirement == ActionRequirement.REQUIRED
        assert act.is_safety_critical is True

        md = auditable_report.to_markdown()
        assert "# AI FIELD ENGINEER -- AUDITABLE DIAGNOSTIC ASSESSMENT REPORT" in md
        assert "TEST_AUDIT_001" in md
        assert "BEARING_DEFECT_WEAR" in md


def test_confidence_decomposition():
    service = ExplainabilityService()
    rep = DiagnosticReport(
        case_id="C1",
        timestamp="2026-09-04",
        equipment={},
        problem_summary="Noise",
        available_modalities=["audio", "sensor"],
        primary_diagnosis="bearing_defect_wear",
        diagnostic_confidence=0.88,
        severity=SeverityLevel.HIGH,
        alternative_hypotheses=[],
        supporting_evidence=[],
        contradicting_evidence=[],
        contradictions_detected=[],
        missing_information=[],
        recommended_actions=[],
        technical_references=["manual.pdf"],
        groundedness_score=1.0,
        unsupported_claims=[],
        status="completed",
    )
    aud_rep = service.generate_auditable_report(rep)
    decomp = aud_rep.confidence_decomposition
    assert decomp.overall_confidence == 0.88
    assert decomp.contradiction_penalty == 0.0


def test_audit_trail_file_persisted():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = ExplainabilityConfig(
            reports_output_dir=temp_dir,
            audit=AuditConfig(audit_storage_dir=temp_dir),
        )
        service = ExplainabilityService(cfg)
        rep = DiagnosticReport(
            case_id="AUDIT_TEST_999",
            timestamp="2026-09-04",
            equipment={},
            problem_summary="Shaft wobble",
            available_modalities=["sensor"],
            primary_diagnosis="rotor_unbalance",
            diagnostic_confidence=0.80,
            severity=SeverityLevel.MEDIUM,
            alternative_hypotheses=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            contradictions_detected=[],
            missing_information=[],
            recommended_actions=[],
            technical_references=[],
            groundedness_score=1.0,
            unsupported_claims=[],
            status="completed",
        )
        service.generate_auditable_report(rep)

        audit_file = Path(temp_dir) / "audit_AUDIT_TEST_999.json"
        assert audit_file.exists()
