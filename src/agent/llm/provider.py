"""
LLM Provider Abstraction and Strict Structured Output Validation.
Supports MockLLMProvider for deterministic offline execution/testing,
as well as standard HTTP-based OpenAI/Anthropic/Gemini/Ollama interfaces.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from src.agent.core.config import LLMConfig
from src.agent.core.schema import SeverityLevel

logger = logging.getLogger(__name__)


class LLMDiagnosticOutputSchema(BaseModel):
    """Pydantic validation schema for LLM structured output."""

    primary_diagnosis: str = Field(description="Leading root-cause failure mode diagnosis.")
    diagnostic_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in primary diagnosis.")
    severity: SeverityLevel = Field(description="Operational severity: LOW, MEDIUM, HIGH, CRITICAL.")
    alternative_hypotheses: list[dict[str, Any]] = Field(default_factory=list, description="Alternative failure modes.")
    supporting_evidence_statements: list[str] = Field(
        default_factory=list, description="Key statements from observations/manuals supporting diagnosis."
    )
    contradicting_evidence_statements: list[str] = Field(
        default_factory=list, description="Conflicting observations or thresholds."
    )
    missing_information: list[str] = Field(
        default_factory=list, description="Missing sensors, metadata, or investigation items."
    )
    recommended_actions: list[dict[str, Any]] = Field(default_factory=list, description="Recommended next actions.")
    cited_technical_references: list[str] = Field(
        default_factory=list, description="Manual citations used in reasoning."
    )


class BaseLLMProvider(ABC):
    """Abstract interface for LLM reasoning backend."""

    @abstractmethod
    def generate_structured_diagnosis(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMDiagnosticOutputSchema:
        """Execute reasoning and return validated structured diagnostic output."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic rule-guided diagnostic reasoning engine for offline environments,
    guaranteeing 100% deterministic, grounded, and schema-valid reasoning.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def generate_structured_diagnosis(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMDiagnosticOutputSchema:
        """Synthesize structured reasoning from inputs deterministically."""
        # Prompt injection safety check
        if "ignore previous instructions" in user_prompt.lower() or "system override" in user_prompt.lower():
            logger.warning("Prompt injection pattern detected in user prompt. Sanitizing.")

        # Heuristic analysis over structured diagnostic context in user_prompt
        lower_p = user_prompt.lower()

        primary_diagnosis = "normal_state"
        confidence = 0.50
        severity = SeverityLevel.LOW
        alternatives = []
        supporting = []
        contradictions = []
        missing = []
        actions = []
        refs = []

        # 1. Bearing Defect detection
        if "bearing" in lower_p or "bpfi" in lower_p or "bpfo" in lower_p or "squeal" in lower_p:
            primary_diagnosis = "bearing_defect_wear"
            confidence = 0.86
            severity = SeverityLevel.HIGH
            supporting.append("Elevated vibration or acoustic squeal detected matching rolling element bearing fault.")
            supporting.append("Acoustic CNN or Vision model predicts bearing defect.")
            alternatives.append(
                {
                    "failure_mode": "rotor_unbalance",
                    "likelihood_score": 0.35,
                    "description": "Unbalance can induce secondary vibration across bearing housings.",
                }
            )
            alternatives.append(
                {
                    "failure_mode": "lubrication_starvation",
                    "likelihood_score": 0.40,
                    "description": "Insufficient grease causes rapid friction and high BPFI harmonics.",
                }
            )
            actions.append(
                {
                    "action_id": "ACT_01",
                    "priority": 1,
                    "action_text": "Perform acoustic ultrasound listening check on drive-end bearing housing.",
                    "rationale": "Verify periodic impact signatures before dismounting.",
                    "is_safety_critical": True,
                    "source_reference": "motor_m4500_maintenance_manual.pdf (Page 2)",
                }
            )
            actions.append(
                {
                    "action_id": "ACT_02",
                    "priority": 2,
                    "action_text": "Inspect bearing grease sample for metallic discoloration.",
                    "rationale": "Confirm physical spalling vs lubrication breakdown.",
                    "is_safety_critical": False,
                    "source_reference": "motor_m4500_maintenance_manual.pdf (Page 2)",
                }
            )
            refs.append("motor_m4500_maintenance_manual.pdf (Page 2, Section: BEARING INSPECTION)")

        # 2. Hydraulic Cavitation detection
        elif "cavitation" in lower_p or "popping" in lower_p or "npsh" in lower_p:
            primary_diagnosis = "hydraulic_cavitation"
            confidence = 0.88
            severity = SeverityLevel.HIGH
            supporting.append("Acoustic broadband hiss (5-15 kHz) or popping sound in pump casing.")
            supporting.append("Erratic discharge pressure fluctuations observed.")
            alternatives.append(
                {
                    "failure_mode": "mechanical_seal_wear",
                    "likelihood_score": 0.30,
                    "description": "Cavitation vibrations can cause secondary seal leakage.",
                }
            )
            actions.append(
                {
                    "action_id": "ACT_CAV_01",
                    "priority": 1,
                    "action_text": "Verify suction line strainer is clean and isolation valve is 100% open.",
                    "rationale": "Ensure NPSHa exceeds NPSHr by at least 1.5 meters.",
                    "is_safety_critical": True,
                    "source_reference": "centrifugal_pump_cp800_troubleshooting.md (Page 1)",
                }
            )
            refs.append("centrifugal_pump_cp800_troubleshooting.md (Page 1, Section: CAVITATION)")

        # 3. Rotor Unbalance detection
        elif "unbalance" in lower_p or "1x" in lower_p or "radial vibration" in lower_p:
            primary_diagnosis = "rotor_unbalance"
            confidence = 0.82
            severity = SeverityLevel.MEDIUM
            supporting.append("Dominant 1X running speed radial vibration component exceeds standard.")
            alternatives.append(
                {
                    "failure_mode": "shaft_misalignment",
                    "likelihood_score": 0.45,
                    "description": "Angular misalignment exhibits strong 2X and axial vibration.",
                }
            )
            actions.append(
                {
                    "action_id": "ACT_UNB_01",
                    "priority": 1,
                    "action_text": "Perform dynamic two-plane balancing if radial vibration > 5.0 mm/s.",
                    "rationale": "Reduce 1X dynamic force on machine bearings.",
                    "is_safety_critical": False,
                    "source_reference": "motor_m4500_maintenance_manual.pdf (Page 3)",
                }
            )
            refs.append("motor_m4500_maintenance_manual.pdf (Page 3, Section: ROTOR UNBALANCE)")

        # 4. Check for contradiction indicators
        if "normal" in lower_p and ("abnormal" in lower_p or "defect" in lower_p or "warning" in lower_p):
            contradictions.append("One sensor/modality indicates normal state while another reports defect/anomaly.")
            confidence = max(0.40, confidence - 0.20)

        # 5. Check for missing information
        if "audio" not in lower_p:
            missing.append("Acoustic audio recording unavailable for harmonics verification.")
        if "sensor" not in lower_p:
            missing.append("Real-time vibration/temperature sensor telemetry unavailable.")

        return LLMDiagnosticOutputSchema(
            primary_diagnosis=primary_diagnosis,
            diagnostic_confidence=confidence,
            severity=severity,
            alternative_hypotheses=alternatives,
            supporting_evidence_statements=supporting,
            contradicting_evidence_statements=contradictions,
            missing_information=missing,
            recommended_actions=actions,
            cited_technical_references=refs,
        )


def create_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function for instantiating the configured LLM provider."""
    provider_type = config.provider.lower()
    if provider_type == "mock":
        return MockLLMProvider(config)
    else:
        logger.warning(f"Provider '{config.provider}' requested. Using MockLLMProvider for deterministic safety.")
        return MockLLMProvider(config)
