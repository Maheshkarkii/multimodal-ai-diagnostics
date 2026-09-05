"""
Agent Configuration Dataclasses for Phase 7 Diagnostic Reasoning.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """LLM Provider and generation parameters."""

    provider: str = "mock"  # "mock", "openai", "anthropic", "gemini", "ollama"
    model_name: str = "gpt-4o-mini"
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class ReasoningConfig:
    """Diagnostic reasoning orchestration parameters."""

    max_reasoning_steps: int = 5
    max_hypotheses: int = 4
    enable_contradiction_detection: bool = True
    enable_groundedness_check: bool = True
    groundedness_threshold: float = 0.60
    allow_active_investigation: bool = True
    max_tool_calls: int = 5


@dataclass
class RetrievalToolConfig:
    """RAG Tool parameters for the agent."""

    default_top_k: int = 3
    min_similarity_threshold: float = 0.15
    max_context_chars: int = 2500


@dataclass
class AgentConfig:
    """Master Diagnostic Reasoning Agent configuration."""

    agent_name: str = "ai_field_engineer_reasoning_agent"
    version: str = "1.0.0"
    llm: LLMConfig = field(default_factory=LLMConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    retrieval_tool: RetrievalToolConfig = field(default_factory=RetrievalToolConfig)
    enabled_tools: list[str] = field(
        default_factory=lambda: [
            "retrieve_technical_evidence",
            "inspect_sensor_state",
            "check_iso_vibration_limits",
        ]
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "AgentConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentConfig":
        return cls(
            agent_name=d.get("agent_name", "ai_field_engineer_reasoning_agent"),
            version=d.get("version", "1.0.0"),
            llm=LLMConfig(**d.get("llm", {})),
            reasoning=ReasoningConfig(**d.get("reasoning", {})),
            retrieval_tool=RetrievalToolConfig(**d.get("retrieval_tool", {})),
            enabled_tools=d.get(
                "enabled_tools",
                [
                    "retrieve_technical_evidence",
                    "inspect_sensor_state",
                    "check_iso_vibration_limits",
                ],
            ),
        )

    def to_yaml(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
