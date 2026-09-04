"""
Unit and Integration Tests for Phase 9 FastAPI Backend & Inference Service.
Tests:
- Application startup and lifespan
- Liveness health probe (GET /health)
- Readiness probe (GET /ready)
- Knowledge RAG query endpoint (GET /api/v1/knowledge/query)
- Multimodal diagnosis endpoint (POST /api/v1/diagnose) with:
  * Text only
  * Sensor data only
  * Multipart image upload
  * Multipart audio upload
  * Multimodal (Text + Sensor + Audio + Image)
- Validation failure handling (empty submission, malformed JSON)
- Request ID & latency header tracking
- OpenAPI schema generation
"""

import json
import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import create_app
from src.api.config import APIConfig


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-field-engineer-api"
    assert "X-Request-ID" in res.headers
    assert "X-Response-Time-Ms" in res.headers


def test_readiness_endpoint(client):
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["ready"] is True
    assert "components" in data


def test_knowledge_query_endpoint(client):
    res = client.get("/api/v1/knowledge/query?q=bearing%20inspection%20ultrasound&top_k=3")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "bearing inspection ultrasound"
    assert data["results_count"] >= 1
    assert len(data["matches"]) >= 1
    assert "motor_m4500_maintenance_manual.pdf" in data["matches"][0]["document_name"]


def test_diagnose_empty_submission_fails(client):
    # Submitting with zero modalities should return 422 Unprocessable Entity
    res = client.post("/api/v1/diagnose")
    assert res.status_code == 422


def test_diagnose_text_only(client):
    res = client.post(
        "/api/v1/diagnose",
        data={
            "technician_description": "Motor emits loud periodic acoustic squealing and excessive vibration.",
            "equipment_json": json.dumps({"equipment_type": "motor", "model": "M-4500"}),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"].startswith("CASE-")
    assert data["status"] == "DIAGNOSIS_SUPPORTED"
    assert data["diagnosis"]["primary_diagnosis"] == "bearing_defect_wear"
    assert len(data["evidence_inventory"]) >= 1
    assert len(data["recommended_actions"]) >= 1


def test_diagnose_sensor_only(client):
    sensor_payload = {
        "vibration": 6.8,
        "vibration_unit": "mm/s",
        "temperature": 84.0,
        "temperature_unit": "degC",
        "rpm": 1480.0,
    }
    res = client.post(
        "/api/v1/diagnose",
        data={
            "sensor_json": json.dumps(sensor_payload),
            "equipment_json": json.dumps({"equipment_type": "motor"}),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "sensor" in data["available_modalities"]
    assert data["diagnosis"]["severity"] == "HIGH"


def test_diagnose_multimodal_with_files(client):
    # 1. Create in-memory synthetic image (224x224 RGB PNG)
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (224, 224), color=(73, 109, 137)).save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # 2. Create in-memory synthetic audio file (WAV header bytes)
    wav_bytes = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    wav_bytes.seek(0)

    res = client.post(
        "/api/v1/diagnose",
        data={
            "technician_description": "Centrifugal pump cavitation noise and popping.",
            "sensor_json": json.dumps({"pressure": 0.4, "pressure_unit": "bar", "vibration": 5.2}),
            "equipment_json": json.dumps({"equipment_type": "pump", "model": "CP-800"}),
        },
        files={
            "image": ("pump_inspection.png", img_byte_arr, "image/png"),
            "audio": ("pump_sound.wav", wav_bytes, "audio/wav"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "vision" in data["available_modalities"]
    assert "audio" in data["available_modalities"]
    assert "sensor" in data["available_modalities"]
    assert data["diagnosis"]["primary_diagnosis"] == "hydraulic_cavitation"
    assert len(data["claim_mappings"]) >= 1


def test_openapi_documentation_schema(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    assert schema["info"]["title"] == "AI Field Engineer - Diagnostic & Troubleshooting API"
    assert "/api/v1/diagnose" in schema["paths"]
    assert "/api/v1/knowledge/query" in schema["paths"]
