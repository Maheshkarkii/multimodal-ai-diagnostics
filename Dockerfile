# ==============================================================================
# AI Field Engineer - Production Multi-Stage Dockerfile (Phase 10)
# Base: Python 3.11-slim (Deterministic, lightweight CPU inference)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Resolution
# ------------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies into a separate wheels directory or virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim as runner

LABEL maintainer="AI Field Engineer Team <eng@aifieldengineer.org>"
LABEL description="AI Field Engineer - Multimodal Autonomous Troubleshooting & Diagnosis System"
LABEL version="1.0.0"

# Set environment variables for Python runtime & UTF-8
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/home/appuser/.local/bin: \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    ENVIRONMENT=production \
    TEMP_UPLOAD_DIR=/tmp/ai-field-engineer/uploads \
    VECTOR_STORE_DIR=/app/data/rag/vector_store \
    MODEL_DIR=/app/models

# Install minimal runtime libraries (libglib/libgomp for OpenCV & PyTorch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 10001 -m -d /home/appuser -s /bin/bash appuser

WORKDIR /app

# Copy installed python wheels/site-packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source code, configuration and default schemas
COPY configs/ /app/configs/
COPY data/rag/ /app/data/rag/
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

# Create persistent storage directories with proper ownership
RUN mkdir -p /tmp/ai-field-engineer/uploads \
             /app/reports/audit \
             /app/reports/explainability \
             /app/reports/diagnostics \
             /app/data/rag/vector_store \
             /app/models \
    && chown -R appuser:appgroup /app /tmp/ai-field-engineer /home/appuser

# Switch to non-root execution user
USER appuser

# Expose production port
EXPOSE 8000

# Health check calling standard liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Production startup command using Uvicorn without --reload
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
