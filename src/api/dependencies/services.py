"""
Dependency Injection Providers for FastAPI Routes.
"""

from src.api.services.file_service import FileValidationService
from src.api.services.orchestrator import DiagnosticOrchestrator

# Global singletons initialized during application lifecycle
_orchestrator: DiagnosticOrchestrator | None = None
_file_service: FileValidationService | None = None


def init_app_services(orchestrator: DiagnosticOrchestrator, file_service: FileValidationService) -> None:
    global _orchestrator, _file_service
    _orchestrator = orchestrator
    _file_service = file_service


def get_diagnostic_orchestrator() -> DiagnosticOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DiagnosticOrchestrator()
    return _orchestrator


def get_file_service() -> FileValidationService:
    global _file_service
    if _file_service is None:
        _file_service = FileValidationService()
    return _file_service
