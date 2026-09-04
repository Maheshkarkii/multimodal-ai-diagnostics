"""
Document Ingestion Pipeline Manager.
Handles batch document discovery, incremental indexing change detection, and validation reporting.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.rag.config import DocumentIngestionConfig
from src.rag.ingestion.parser import DocumentParser, compute_file_hash
from src.rag.schema import DocumentMetadata, RawDocumentPage

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Orchestrates document discovery, hash checking, parsing, and provenance tracking."""

    def __init__(
        self,
        config: Optional[DocumentIngestionConfig] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ):
        self.config = config or DocumentIngestionConfig()
        self.parser = DocumentParser(self.config)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest: Dict[str, Dict] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        """Load persistent record of previously ingested document hashes."""
        if self.manifest_path and self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest from {self.manifest_path}: {e}")
        return {}

    def save_manifest(self) -> None:
        """Persist document hashes manifest."""
        if self.manifest_path:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)

    def scan_directory(self, dir_path: Union[str, Path]) -> List[Path]:
        """Recursively scan a directory for supported technical document files."""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return []

        found_files: List[Path] = []
        for ext in self.config.supported_extensions:
            found_files.extend(dir_path.rglob(f"*{ext}"))
            found_files.extend(dir_path.rglob(f"*{ext.upper()}"))

        # Deduplicate and sort for deterministic ordering
        unique_paths = sorted(list({p.resolve() for p in found_files}))
        return unique_paths

    def ingest_document(
        self,
        file_path: Union[str, Path],
        force_reindex: bool = False,
        extra_metadata: Optional[Dict] = None,
    ) -> Tuple[Optional[DocumentMetadata], List[RawDocumentPage], bool]:
        """
        Ingest a single document file.
        Returns:
            Tuple of (DocumentMetadata, List[RawDocumentPage], was_skipped: bool)
        """
        path = Path(file_path)
        file_hash = compute_file_hash(path)
        str_path = str(path.resolve())

        # Check incremental hash
        if not force_reindex and str_path in self.manifest:
            prev_entry = self.manifest[str_path]
            if prev_entry.get("file_hash") == file_hash:
                logger.info(f"Skipping '{path.name}' (unaltered file hash {file_hash[:8]}).")
                return None, [], True

        # Parse document
        metadata, pages = self.parser.parse_document(path, extra_metadata=extra_metadata)

        # Update manifest record
        self.manifest[str_path] = {
            "document_id": metadata.document_id,
            "document_name": metadata.document_name,
            "file_hash": file_hash,
            "file_size_bytes": metadata.file_size_bytes,
            "num_pages": len(pages),
            "equipment_type": metadata.equipment_type,
            "manufacturer": metadata.manufacturer,
            "model": metadata.model,
        }
        self.save_manifest()

        return metadata, pages, False

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        force_reindex: bool = False,
        metadata_map: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Batch ingest all documents found in a directory.
        Returns detailed summary statistics.
        """
        dir_path = Path(dir_path)
        doc_paths = self.scan_directory(dir_path)
        meta_map = metadata_map or {}

        results: List[Tuple[DocumentMetadata, List[RawDocumentPage]]] = []
        skipped_count = 0
        failed_files: List[Tuple[str, str]] = []

        total_pages = 0

        for path in doc_paths:
            try:
                extra_meta = meta_map.get(path.name, {})
                meta, pages, was_skipped = self.ingest_document(
                    path, force_reindex=force_reindex, extra_metadata=extra_meta
                )
                if was_skipped:
                    skipped_count += 1
                elif meta is not None:
                    results.append((meta, pages))
                    total_pages += len(pages)
            except Exception as e:
                logger.error(f"Failed to ingest document '{path.name}': {e}")
                failed_files.append((str(path), str(e)))

        return {
            "documents_found": len(doc_paths),
            "documents_parsed": len(results),
            "documents_skipped": skipped_count,
            "total_pages_extracted": total_pages,
            "failed_files": failed_files,
            "parsed_documents": results,
        }
