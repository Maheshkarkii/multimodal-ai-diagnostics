"""
Multimodal Diagnosis Route Handlers.
Supports multipart form uploads with optional images, audio files, JSON sensor data, and notes.
"""

import json
import logging
from pathlib import Path
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status

from src.api.schemas.diagnosis import DiagnosisResponse, EquipmentMetadataInput, SensorTelemetryInput
from src.api.dependencies.services import get_diagnostic_orchestrator, get_file_service
from src.api.services.file_service import FileValidationService
from src.api.services.orchestrator import DiagnosticOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Diagnostic Reasoning & Troubleshooting"])


@router.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Multimodal Autonomous Equipment Diagnosis",
    description=(
        "Submit an industrial diagnostic case with arbitrary combinations of optional modalities: "
        "inspection image, acoustic audio, sensor telemetry, technician description, and metadata."
    ),
)
async def submit_diagnostic_case(
    request: Request,
    technician_description: Optional[str] = Form(None, description="Field technician symptom observation"),
    sensor_json: Optional[str] = Form(None, description="JSON string encoded SensorTelemetryInput"),
    equipment_json: Optional[str] = Form(None, description="JSON string encoded EquipmentMetadataInput"),
    image: Optional[UploadFile] = File(None, description="Optional equipment image file (JPEG, PNG, WebP)"),
    audio: Optional[UploadFile] = File(None, description="Optional acoustic recording file (WAV, MP3)"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    orchestrator: DiagnosticOrchestrator = Depends(get_diagnostic_orchestrator),
    file_service: FileValidationService = Depends(get_file_service),
):
    req_id = x_request_id or f"REQ-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"[{req_id}] Ingesting incoming diagnostic submission...")

    # 1. Validate that at least ONE diagnostic modality is provided
    if not any([technician_description, sensor_json, image, audio]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one diagnostic modality (image, audio, sensor data, or technician notes) must be provided.",
        )

    # 2. Parse JSON inputs safely
    sensor_input: Optional[SensorTelemetryInput] = None
    if sensor_json:
        try:
            s_dict = json.loads(sensor_json)
            sensor_input = SensorTelemetryInput(**s_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed sensor_json payload: {str(e)}",
            )

    equip_input: Optional[EquipmentMetadataInput] = None
    if equipment_json:
        try:
            e_dict = json.loads(equipment_json)
            equip_input = EquipmentMetadataInput(**e_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed equipment_json payload: {str(e)}",
            )

    saved_img_path: Optional[Path] = None
    saved_aud_path: Optional[Path] = None

    try:
        # 3. Securely validate & stage files
        if image:
            saved_img_path = await file_service.save_and_validate_image(image)
        if audio:
            saved_aud_path = await file_service.save_and_validate_audio(audio)

        # 4. Invoke Central Diagnostic Orchestrator
        response = await orchestrator.execute_diagnosis(
            request_id=req_id,
            technician_description=technician_description,
            sensor_data=sensor_input,
            equipment_meta=equip_input,
            image_path=saved_img_path,
            audio_path=saved_aud_path,
        )

        return response

    finally:
        # 5. Deterministic temporary file cleanup
        if saved_img_path:
            file_service.cleanup_file(saved_img_path)
        if saved_aud_path:
            file_service.cleanup_file(saved_aud_path)
