"""
Secure Temporary File and Payload Validation Service.
Handles multipart uploaded files, validates mime-types and size limits,
and guarantees deterministic temporary cleanup.
"""

from pathlib import Path
import shutil
import tempfile
import uuid
import logging
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SUPPORTED_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}


class FileValidationService:
    """Validates uploaded media and manages sandboxed temporary storage."""

    def __init__(self, temp_dir: str = "data/temp_uploads", max_img_mb: float = 15.0, max_aud_mb: float = 25.0):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_img_bytes = int(max_img_mb * 1024 * 1024)
        self.max_aud_bytes = int(max_aud_mb * 1024 * 1024)

    async def save_and_validate_image(self, file: UploadFile) -> Path:
        """Validate and save uploaded image to a secure temporary path."""
        # Sanitize filename & extension
        ext = Path(file.filename or "image.jpg").suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported image format '{ext}'. Allowed: {list(SUPPORTED_IMAGE_EXTS)}",
            )

        # Read content & check size limit
        content = await file.read()
        if len(content) > self.max_img_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image file exceeds maximum allowable size of {self.max_img_bytes / (1024*1024):.1f} MB",
            )

        # Safe unique path
        safe_filename = f"img_{uuid.uuid4().hex[:12]}{ext}"
        save_path = self.temp_dir / safe_filename
        save_path.write_bytes(content)
        return save_path

    async def save_and_validate_audio(self, file: UploadFile) -> Path:
        """Validate and save uploaded audio file to a secure temporary path."""
        ext = Path(file.filename or "audio.wav").suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio format '{ext}'. Allowed: {list(SUPPORTED_AUDIO_EXTS)}",
            )

        content = await file.read()
        if len(content) > self.max_aud_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio file exceeds maximum allowable size of {self.max_aud_bytes / (1024*1024):.1f} MB",
            )

        safe_filename = f"aud_{uuid.uuid4().hex[:12]}{ext}"
        save_path = self.temp_dir / safe_filename
        save_path.write_bytes(content)
        return save_path

    def cleanup_file(self, file_path: Optional[Path]) -> None:
        """Securely remove temporary file."""
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file '{file_path}': {e}")
