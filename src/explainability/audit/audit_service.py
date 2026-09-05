"""
Audit Trail and Immutable Run Logger.
Records complete, reproducible operational execution parameters without leaking secrets or private CoT.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.explainability.core.schema import AuditTrailRecord

logger = logging.getLogger(__name__)


def compute_string_hash(text: str) -> str:
    """Deterministic SHA-256 hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class AuditService:
    """Manages audit logging and persistent storage for compliance and diagnostic verification."""

    def __init__(self, audit_storage_dir: str = "reports/audit"):
        self.storage_dir = Path(audit_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def log_diagnostic_run(
        self,
        case_id: str,
        system_version: str,
        report_version: str,
        model_versions: dict[str, str],
        knowledge_base_version: str,
        input_data_summary: dict[str, Any],
        available_modalities: list[str],
        retrieval_queries: list[str],
        retrieved_chunk_ids: list[str],
        final_diagnosis: str,
        diagnostic_confidence: float,
        status: str,
        execution_duration_ms: float,
    ) -> AuditTrailRecord:
        """Create and persist an immutable audit record."""
        # Compute input fingerprints
        input_hashes = {}
        for k, v in input_data_summary.items():
            input_hashes[k] = compute_string_hash(str(v))

        record = AuditTrailRecord(
            case_id=case_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            system_version=system_version,
            report_version=report_version,
            model_versions=model_versions,
            knowledge_base_version=knowledge_base_version,
            input_hashes=input_hashes,
            available_modalities=available_modalities,
            retrieval_queries=retrieval_queries,
            retrieved_chunk_ids=retrieved_chunk_ids,
            final_diagnosis=final_diagnosis,
            diagnostic_confidence=diagnostic_confidence,
            status=status,
            execution_duration_ms=execution_duration_ms,
        )

        # Persist audit record to disk
        file_path = self.storage_dir / f"audit_{case_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        logger.info(f"Audit record saved for case '{case_id}' to '{file_path}'")
        return record
