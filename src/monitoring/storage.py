"""
Thread-safe Local Storage Backend for Monitoring Telemetry Events.
Persists sanitized JSON Lines records without exposing raw user files or keys.
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import List, Optional
from src.monitoring.events import DiagnosticMonitoringEvent

logger = logging.getLogger(__name__)


class MonitoringStore:
    """Append-only thread-safe JSONL storage for system telemetry."""

    def __init__(self, log_path: str = "reports/monitoring/events.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record_event(self, event: DiagnosticMonitoringEvent) -> None:
        """Persist a single sanitized monitoring event."""
        line = json.dumps(event.model_dump()) + "\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def load_events(self, limit: int = 1000) -> List[DiagnosticMonitoringEvent]:
        """Load recent monitoring events for aggregation and drift analysis."""
        if not self.log_path.exists():
            return []
        
        events = []
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            events.append(DiagnosticMonitoringEvent(**data))
                        except Exception as e:
                            logger.warning(f"Skipping malformed event line: {e}")
        return events[-limit:]
