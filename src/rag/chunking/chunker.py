"""
Deliberate Technical Document Chunker.
Splits documents by sections, paragraphs, bullet points, and tables while preserving page provenance.
"""

import logging
import re

from src.rag.config import ChunkingConfig
from src.rag.schema import DocumentChunk, DocumentMetadata, RawDocumentPage

logger = logging.getLogger(__name__)


class TechnicalDocumentChunker:
    """
    Structure-aware chunker tailored for engineering manuals, maintenance SOPs,
    and specification sheets.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def chunk_document(self, metadata: DocumentMetadata, pages: list[RawDocumentPage]) -> list[DocumentChunk]:
        """
        Split document pages into coherent, structure-preserving chunks.
        """
        chunks: list[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            current_section = page.section_title

            # Break page into structured semantic blocks (headings, tables, paragraphs)
            blocks = self._extract_semantic_blocks(page_text)

            # Group blocks into target chunk sizes with overlap
            page_chunks = self._group_blocks_into_chunks(
                blocks=blocks,
                metadata=metadata,
                page_number=page.page_number,
                default_section=current_section,
                start_index=global_chunk_idx,
            )

            chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return chunks

    def _extract_semantic_blocks(self, text: str) -> list[str]:
        """
        Split raw text into logical semantic units (sections, paragraphs, tables).
        """
        # Split by double newline first
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        blocks: list[str] = []

        for p in raw_paragraphs:
            # Check if block is a table or contains structured procedures
            if "|" in p and "\n" in p:
                # Keep table intact as a single block where possible
                blocks.append(p)
            elif len(p) > self.config.chunk_size:
                # If paragraph exceeds chunk size, split by sentence or numbered items
                sub_blocks = self._split_large_paragraph(p)
                blocks.extend(sub_blocks)
            else:
                blocks.append(p)

        return blocks

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        """Split oversized paragraph using sentence boundaries or numbered lists."""
        # Split on numbered list items or sentences
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\n)(?=\d+[\.\)])", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [paragraph]

        sub_blocks: list[str] = []
        curr = ""

        for s in sentences:
            if not curr:
                curr = s
            elif len(curr) + len(s) + 1 <= self.config.chunk_size:
                curr += " " + s
            else:
                sub_blocks.append(curr)
                curr = s

        if curr:
            sub_blocks.append(curr)

        return sub_blocks

    def _group_blocks_into_chunks(
        self,
        blocks: list[str],
        metadata: DocumentMetadata,
        page_number: int,
        default_section: str | None,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Combine blocks into chunks with controlled size and overlap."""
        chunks: list[DocumentChunk] = []
        current_text_blocks: list[str] = []
        current_len = 0
        current_section = default_section
        chunk_idx = start_index

        for block in blocks:
            # Detect section heading inside block
            lines = block.split("\n")
            if lines and (lines[0].startswith("#") or lines[0].isupper() and len(lines[0]) < 80):
                current_section = lines[0].lstrip("#").strip()

            block_len = len(block)

            if current_len + block_len <= self.config.chunk_size or not current_text_blocks:
                current_text_blocks.append(block)
                current_len += block_len + 1
            else:
                # Emit current accumulated chunk
                chunk_text = "\n\n".join(current_text_blocks).strip()
                chunks.append(
                    DocumentChunk.create(
                        document_id=metadata.document_id,
                        document_name=metadata.document_name,
                        source_path=metadata.source_path,
                        page_number=page_number,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        section=current_section,
                        equipment_type=metadata.equipment_type,
                        manufacturer=metadata.manufacturer,
                        model=metadata.model,
                    )
                )
                chunk_idx += 1

                # Calculate overlap: retain the last block if it fits within chunk_overlap
                if (
                    self.config.chunk_overlap > 0
                    and current_text_blocks
                    and len(current_text_blocks[-1]) <= self.config.chunk_overlap
                ):
                    current_text_blocks = [current_text_blocks[-1], block]
                    current_len = len(current_text_blocks[0]) + block_len + 1
                else:
                    current_text_blocks = [block]
                    current_len = block_len

        # Emit remaining blocks
        if current_text_blocks:
            chunk_text = "\n\n".join(current_text_blocks).strip()
            chunks.append(
                DocumentChunk.create(
                    document_id=metadata.document_id,
                    document_name=metadata.document_name,
                    source_path=metadata.source_path,
                    page_number=page_number,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    section=current_section,
                    equipment_type=metadata.equipment_type,
                    manufacturer=metadata.manufacturer,
                    model=metadata.model,
                )
            )

        return chunks
