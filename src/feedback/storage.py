"""
Feedback Storage and Aggregation Engine.
Stores human engineer feedback and provides disagreement and module error queries.
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from src.feedback.schemas import HumanDiagnosticFeedback

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Thread-safe persistent JSONL store for field engineer reviews."""

    def __init__(self, log_path: str = "reports/feedback/feedback.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def submit_feedback(self, feedback: HumanDiagnosticFeedback) -> None:
        """Persist structured human review record."""
        line = json.dumps(feedback.model_dump()) + "\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def load_feedback(self) -> list[HumanDiagnosticFeedback]:
        """Load all human feedback entries."""
        if not self.log_path.exists():
            return []
        records = []
        with self._lock:
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(HumanDiagnosticFeedback(**json.loads(line)))
                        except Exception as e:
                            logger.warning(f"Error loading feedback entry: {e}")
        return records

    def analyze_feedback(self) -> dict[str, Any]:
        """Analyze disagreement rates and frequently corrected diagnoses."""
        records = self.load_feedback()
        if not records:
            return {"total_feedback_count": 0, "accuracy_rate": 1.0}

        total = len(records)
        correct = sum(1 for r in records if r.is_diagnosis_accurate)
        category_counts: dict[str, int] = {}
        corrections: dict[str, int] = {}

        for r in records:
            cat = str(r.category.value)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if r.ground_truth_correction:
                corrections[r.ground_truth_correction] = corrections.get(r.ground_truth_correction, 0) + 1

        return {
            "total_feedback_count": total,
            "accuracy_rate": round(correct / total, 4),
            "category_distribution": category_counts,
            "frequent_corrections": corrections,
        }
