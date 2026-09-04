"""
API Configuration Settings for Phase 9/10 FastAPI Backend.
Supports YAML loading and environment variable overrides for production deployment.
"""

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ServerConfig:
    """FastAPI Server configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    reload: bool = field(default_factory=lambda: os.getenv("API_RELOAD", "false").lower() in ("true", "1", "yes"))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    api_prefix: str = field(default_factory=lambda: os.getenv("API_PREFIX", "/api/v1"))
    title: str = "AI Field Engineer - Diagnostic & Troubleshooting API"
    version: str = "1.0.0"
    description: str = "Industrial multimodal diagnostic reasoning, RAG knowledge retrieval, and auditable reporting service."
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000").split(",")
            if o.strip()
        ]
    )
    temp_upload_dir: str = field(default_factory=lambda: os.getenv("TEMP_UPLOAD_DIR", "data/temp_uploads"))
    max_image_size_mb: float = field(default_factory=lambda: float(os.getenv("MAX_IMAGE_SIZE_MB", "15.0")))
    max_audio_size_mb: float = field(default_factory=lambda: float(os.getenv("MAX_AUDIO_SIZE_MB", "25.0")))
    request_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60.0")))
    git_sha: str = field(default_factory=lambda: os.getenv("GIT_SHA", "unknown"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))


@dataclass
class APIConfig:
    """Master API Service Configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "APIConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APIConfig":
        server_dict = d.get("server", {})
        # Environment variables take precedence if present
        if "API_HOST" in os.environ:
            server_dict["host"] = os.environ["API_HOST"]
        if "API_PORT" in os.environ:
            server_dict["port"] = int(os.environ["API_PORT"])
        if "ALLOWED_ORIGINS" in os.environ:
            server_dict["allowed_origins"] = [o.strip() for o in os.environ["ALLOWED_ORIGINS"].split(",") if o.strip()]
        if "TEMP_UPLOAD_DIR" in os.environ:
            server_dict["temp_upload_dir"] = os.environ["TEMP_UPLOAD_DIR"]
        if "ENVIRONMENT" in os.environ:
            server_dict["environment"] = os.environ["ENVIRONMENT"]
        if "GIT_SHA" in os.environ:
            server_dict["git_sha"] = os.environ["GIT_SHA"]

        return cls(server=ServerConfig(**server_dict))

    def to_yaml(self, save_path: Union[str, Path]) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
