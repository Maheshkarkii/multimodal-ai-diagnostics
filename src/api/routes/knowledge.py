"""
RAG Technical Knowledge Route Handlers.
Exposes direct semantic queries to the persistent vector database.
"""

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel

from src.api.dependencies.services import get_diagnostic_orchestrator
from src.api.services.orchestrator import DiagnosticOrchestrator

router = APIRouter(prefix="/api/v1/knowledge", tags=["Technical Manuals & RAG Knowledge"])


class RAGQueryResultItem(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    section: str | None
    score: float
    text: str
    citation: str


class RAGQueryResponse(BaseModel):
    query: str
    results_count: int
    matches: list[RAGQueryResultItem]


@router.get("/query", response_model=RAGQueryResponse, status_code=status.HTTP_200_OK)
async def query_technical_manuals(
    q: str = Query(
        ...,
        description="Technical question or fault symptom",
        json_schema_extra={"examples": ["vibration severity limits ISO"]},
    ),
    equipment_type: str | None = Query(None, description="Optional equipment filter (motor, pump, gearbox)"),
    top_k: int = Query(5, ge=1, le=20, description="Max candidate chunks to return"),
    orchestrator: DiagnosticOrchestrator = Depends(get_diagnostic_orchestrator),
):
    """Query OEM equipment manuals and maintenance SOPs directly via hybrid RAG retrieval."""
    filters = {"equipment_type": equipment_type} if equipment_type else None
    results = orchestrator.retriever.retrieve(
        query=q,
        top_k=top_k,
        filters=filters,
    )

    matches = [
        RAGQueryResultItem(
            chunk_id=r.chunk_id,
            document_name=r.document_name,
            page_number=r.page_number,
            section=r.section,
            score=r.score,
            text=r.text,
            citation=r.formatted_citation(),
        )
        for r in results
    ]

    return RAGQueryResponse(
        query=q,
        results_count=len(matches),
        matches=matches,
    )
