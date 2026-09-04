"""
Frontend Integration Test for Phase 12.
Simulates client-side form submissions and tests compatibility with the FastAPI backend.
"""

from fastapi.testclient import TestClient
from src.api.main import create_app


def test_frontend_case_submission_and_report_schema_alignment():
    """Verify that form payloads sent by the frontend match the FastAPI API schemas."""
    app = create_app()
    client = TestClient(app)

    # 1. Simulate multi-field form submission matching Phase 9 route expectations
    form_data = {
        "technician_description": "High pitch squeal and casing temperature elevated to 88C",
        "sensor_json": '{"temperature": 88.0, "vibration": 7.4, "rpm": 1490.0}',
        "equipment_json": '{"equipment_type": "motor", "manufacturer": "Siemens", "model": "M-4500"}',
    }

    response = client.post("/api/v1/diagnose", data=form_data)
    assert response.status_code == 200

    data = response.json()
    assert "case_id" in data
    assert "diagnosis" in data
    assert "primary_diagnosis" in data["diagnosis"]
    assert "diagnostic_confidence" in data["diagnosis"]
    assert "severity" in data["diagnosis"]
    assert "confidence_decomposition" in data["diagnosis"]
    assert "evidence_inventory" in data
    assert len(data["evidence_inventory"]) > 0
    assert "recommended_actions" in data
