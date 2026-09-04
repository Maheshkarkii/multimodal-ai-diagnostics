"""
API Configuration Settings for Phase 9 FastAPI Backend.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ServerConfig:
    """FastAPI Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    api_prefix: str = "/api/v1"
    title: str = "AI Field Engineer - Diagnostic & Troubleshooting API"
    version: str = "1.0.0"
    description: str = "Industrial multimodal diagnostic reasoning, RAG knowledge retrieval, and auditable reporting service."
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])
    temp_upload_dir: str = "data/temp_uploads"
    max_image_size_mb: float = 15.0
    max_audio_size_mb: float = 25.0
    request_timeout_seconds: float = 60.0


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
        return cls(server=ServerConfig(**d.get("server", {})))

    def to_yaml(self, save_path: Union[str, Path]) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
