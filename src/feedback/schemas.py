"""
Human Field Engineer Feedback Schema and Analysis Dataclasses.
Enables structured human-in-the-loop review, validation, and dataset curation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeedbackCategory(str, Enum):
    CORRECT = "CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    INCORRECT = "INCORRECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSAFE_RECOMMENDATION = "UNSAFE_RECOMMENDATION"
    IRRELEVANT_EVIDENCE = "IRRELEVANT_EVIDENCE"
    MISSING_INFORMATION = "MISSING_INFORMATION"


class HumanDiagnosticFeedback(BaseModel):
    """Structured feedback submission from a field engineer or domain expert."""
    feedback_id: str
    case_id: str
    reviewer_id: str = Field(description="Anonymous or internal engineer identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Review Findings
    category: FeedbackCategory = FeedbackCategory.CORRECT
    is_diagnosis_accurate: bool = True
    ground_truth_correction: Optional[str] = None
    confidence_rating_1_to_5: int = Field(5, ge=1, le=5)
    
    # Specific Module Feedback
    were_actions_useful: bool = True
    were_citations_accurate: bool = True
    notes_and_observations: Optional[str] = None
    
    # Workflow status
    curated_for_evaluation: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
