"""
Deployment, Configuration, and Container Health Tests (Phase 10).
Validates environment variable parsing, missing settings, paths, CORS, and version metadata.
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.config import APIConfig, ServerConfig
from src.api.main import create_app


def test_api_config_environment_variable_overrides():
    """Verify that environment variables properly override default API configuration."""
    os.environ["API_HOST"] = "127.0.0.1"
    os.environ["API_PORT"] = "9000"
    os.environ["ALLOWED_ORIGINS"] = "https://app.industrialai.com,https://monitoring.internal"
    os.environ["TEMP_UPLOAD_DIR"] = "/tmp/custom_uploads"
    os.environ["ENVIRONMENT"] = "production"
    os.environ["GIT_SHA"] = "commit-abc1234"

    try:
        config = APIConfig.from_dict({})
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.allowed_origins == ["https://app.industrialai.com", "https://monitoring.internal"]
        assert config.server.temp_upload_dir == "/tmp/custom_uploads"
        assert config.server.environment == "production"
        assert config.server.git_sha == "commit-abc1234"
    finally:
        for k in ["API_HOST", "API_PORT", "ALLOWED_ORIGINS", "TEMP_UPLOAD_DIR", "ENVIRONMENT", "GIT_SHA"]:
            os.environ.pop(k, None)


def test_health_endpoint_metadata_and_versioning():
    """Verify that /health exposes non-sensitive build and environment metadata."""
    os.environ["ENVIRONMENT"] = "staging"
    os.environ["GIT_SHA"] = "deploy-sha-8899"

    try:
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-field-engineer-api"
        assert data["version"] == "1.0.0"
        assert data["environment"] == "staging"
        assert data["git_sha"] == "deploy-sha-8899"
        assert "timestamp" in data
    finally:
        os.environ.pop("ENVIRONMENT", None)
        os.environ.pop("GIT_SHA", None)


def test_cors_headers_production_restrictions():
    """Verify that CORS correctly reflects configured origins."""
    custom_config = APIConfig(
        server=ServerConfig(
            allowed_origins=["https://secure-dashboard.company.com"]
        )
    )
    app = create_app(custom_config)
    client = TestClient(app)

    # Allowed origin request
    res = client.get("/health", headers={"Origin": "https://secure-dashboard.company.com"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://secure-dashboard.company.com"


def test_temporary_upload_directory_lifecycle(tmp_path):
    """Verify temporary directory creation and file clearance."""
    from src.api.services.file_service import FileValidationService
    
    custom_temp = tmp_path / "test_uploads"
    service = FileValidationService(temp_dir=str(custom_temp))
    
    assert custom_temp.exists()
    
    test_file = custom_temp / "dummy.txt"
    test_file.write_text("sample content")
    assert test_file.exists()
    
    service.cleanup_file(str(test_file))
    assert not test_file.exists()
