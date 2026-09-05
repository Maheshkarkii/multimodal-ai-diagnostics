"""
Request ID and Latency Middleware for FastAPI.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID and measures request latency."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or f"REQ-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.time()

        # Log request start
        logger.info(f"[{req_id}] Started {request.method} '{request.url.path}'")

        try:
            response: Response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000.0
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
            logger.info(
                f"[{req_id}] Completed {request.method} '{request.url.path}' with status {response.status_code} in {duration_ms:.2f}ms"
            )
            return response
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000.0
            logger.error(
                f"[{req_id}] Unhandled error on {request.method} '{request.url.path}': {e} (after {duration_ms:.2f}ms)"
            )
            raise
