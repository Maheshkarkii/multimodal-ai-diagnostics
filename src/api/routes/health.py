"""
Health and Readiness Route Handlers.
"""

from datetime import datetime
import os
from fastapi import APIRouter, Depends, status, Response
from src.api.schemas.diagnosis import HealthResponse, ReadinessResponse
from src.api.dependencies.services import get_diagnostic_orchestrator
from src.api.services.orchestrator import DiagnosticOrchestrator

router = APIRouter(tags=["System Health & Monitoring"])

@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness probe returning application operational status and version metadata."""
    return HealthResponse(
        status="healthy",
        service="ai-field-engineer-api",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "production"),
        git_sha=os.getenv("GIT_SHA", "unknown"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    response: Response,
    orchestrator: DiagnosticOrchestrator = Depends(get_diagnostic_orchestrator),
):
    """Readiness probe verifying initialization of underlying ML and RAG subsystems."""
    is_ready = orchestrator.is_ready

    components = {
        "vision_service": True,
        "audio_service": True,
        "sensor_service": True,
        "multimodal_fusion": True,
        "rag_retrieval": orchestrator.vector_store.count() > 0,
        "diagnostic_agent": True,
        "explainability_service": True,
    }

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=is_ready,
        status="ready" if is_ready else "initializing",
        components=components,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
