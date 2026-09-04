"""
Main FastAPI Application Entrypoint.
Initializes lifespan, routers, CORS middleware, global exception handlers, and services.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import logging
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.config import APIConfig
from src.api.dependencies.services import init_app_services
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.routes.health import router as health_router
from src.api.routes.diagnosis import router as diagnosis_router
from src.api.routes.knowledge import router as knowledge_router
from src.api.schemas.diagnosis import ErrorResponse
from src.api.services.file_service import FileValidationService
from src.api.services.orchestrator import DiagnosticOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager handling application startup and resource cleanup."""
    logger.info("Initializing AI Field Engineer Backend Services on Startup...")
    config = getattr(app.state, "config", APIConfig())

    # Initialize singletons
    file_service = FileValidationService(
        temp_dir=config.server.temp_upload_dir,
        max_img_mb=config.server.max_image_size_mb,
        max_aud_mb=config.server.max_audio_size_mb,
    )
    orchestrator = DiagnosticOrchestrator()

    init_app_services(orchestrator, file_service)
    logger.info("AI Models, RAG Index, and Diagnostic Agent initialized successfully.")

    yield

    logger.info("Shutting down AI Field Engineer Backend. Releasing resources...")


def create_app(config: APIConfig = None) -> FastAPI:
    """Factory function for instantiating the FastAPI application."""
    cfg = config or APIConfig()

    app = FastAPI(
        title=cfg.server.title,
        version=cfg.server.version,
        description=cfg.server.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.config = cfg

    # 1. Add Middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Centralized Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = request.headers.get("X-Request-ID")
        logger.warning(f"[{req_id}] Validation error on '{request.url.path}': {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_code="INVALID_INPUT",
                message="Request payload failed validation.",
                details={"errors": exc.errors()},
                request_id=req_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = request.headers.get("X-Request-ID")
        logger.error(f"[{req_id}] Unhandled internal exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="An unexpected server error occurred during diagnostic processing.",
                details={"error_type": type(exc).__name__},
                request_id=req_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ).dict(),
        )

    # 3. Register Routers
    app.include_router(health_router)
    app.include_router(diagnosis_router)
    app.include_router(knowledge_router)

    return app


# Default ASGI Application Instance for Uvicorn
app = create_app()
