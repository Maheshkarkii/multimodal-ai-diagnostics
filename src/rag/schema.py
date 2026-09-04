"""
Data models and typed representations for RAG Documents, Chunks, and Retrieved Evidence.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import hashlib
import json


@dataclass
class DocumentMetadata:
    """Provenance and origin metadata for an ingested technical manual."""
    document_id: str
    document_name: str
    source_path: str
    file_type: str
    file_hash: str
    file_size_bytes: int
    num_pages: int
    equipment_type: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    manual_version: Optional[str] = None
    publication_date: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawDocumentPage:
    """Extracted text and structural information from a single document page."""
    page_number: int  # 1-indexed
    text: str
    section_title: Optional[str] = None
    has_tables: bool = False
    is_scanned_suspicious: bool = False
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Atomic text chunk with exact provenance and contextual metadata."""
    chunk_id: str
    document_id: str
    document_name: str
    source_path: str
    page_number: int
    text: str
    section: Optional[str] = None
    chunk_index_in_doc: int = 0
    document_type: str = "technical_manual"
    equipment_type: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        document_id: str,
        document_name: str,
        source_path: str,
        page_number: int,
        text: str,
        chunk_index: int,
        section: Optional[str] = None,
        equipment_type: Optional[str] = None,
        manufacturer: Optional[str] = None,
        model: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> "DocumentChunk":
        # Deterministic chunk ID computed from doc ID, page, index, and content hash
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"{document_id}_p{page_number}_c{chunk_index}_{content_hash}"
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            document_name=document_name,
            source_path=source_path,
            page_number=page_number,
            text=text,
            section=section,
            chunk_index_in_doc=chunk_index,
            equipment_type=equipment_type,
            manufacturer=manufacturer,
            model=model,
            metadata=extra_metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


@dataclass
class RetrievedEvidence:
    """Typed internal representation for evidence retrieved from the knowledge base."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    section: Optional[str]
    text: str
    score: float
    retrieval_mode: str  # "dense", "sparse", "hybrid"
    source_path: str
    equipment_type: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def formatted_citation(self) -> str:
        """Human-readable citation format."""
        sec_str = f" | Section: {self.section}" if self.section else ""
        return f"{self.document_name} (Page {self.page_number}{sec_str})"


@dataclass
class StructuredEvidenceContext:
    """Context assembled from multiple evidence chunks, formatted for reasoning models."""
    query: str
    evidence_items: List[RetrievedEvidence]
    total_chunks: int
    total_characters: int
    truncated: bool = False

    def to_structured_text(self) -> str:
        """Format as a clear, standardized document citation block."""
        if not self.evidence_items:
            return "No sufficiently relevant technical evidence found in indexed manuals."

        lines = [
            "=== TECHNICAL KNOWLEDGE RETRIEVAL EVIDENCE ===",
            f"Query: {self.query}",
            f"Retrieved Chunks: {len(self.evidence_items)} (Total Indexed Matches: {self.total_chunks})",
            "------------------------------------------------"
        ]

        for idx, ev in enumerate(self.evidence_items, start=1):
            lines.append(f"\n[EVIDENCE {idx}]")
            lines.append(f"DOCUMENT: {ev.document_name}")
            lines.append(f"PAGE: {ev.page_number}")
            if ev.section:
                lines.append(f"SECTION: {ev.section}")
            if ev.equipment_type:
                lines.append(f"EQUIPMENT: {ev.equipment_type}")
            lines.append(f"RELEVANCE SCORE: {ev.score:.4f} ({ev.retrieval_mode})")
            lines.append("CONTENT:")
            lines.append(ev.text.strip())
            lines.append("------------------------------------------------")

        return "\n".join(lines)
