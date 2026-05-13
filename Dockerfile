# ============================================================================
# ModelServe — FastAPI Inference Service Dockerfile
# ============================================================================
# TODO: Implement a multi-stage Docker build.
#
# Requirements:
#   - Multi-stage build (at least two FROM statements)
#   - Final image must be under 800 MB
#   - Must run as a non-root user
#   - Must use a production WSGI/ASGI server (gunicorn with uvicorn workers)
#   - Must include a HEALTHCHECK directive
#   - Must copy only what's needed (use .dockerignore too)
#
# Suggested stages:
#   Stage 1 (builder):
#     - Start from python:3.10-slim
#     - Install build dependencies (gcc, etc.)
#     - Copy requirements.txt and install Python packages
#
#   Stage 2 (runtime):
#     - Start from python:3.10-slim (clean)
#     - Copy installed packages from builder stage
#     - Copy application code
#     - Create a non-root user and switch to it
#     - Expose the service port
#     - Set the healthcheck
#     - Define the CMD with gunicorn/uvicorn
# ============================================================================

# ============================================================================
# ModelServe — FastAPI Inference Service Dockerfile
# ============================================================================
# Multi-stage production Docker build
# Python version: 3.12
# ============================================================================


# ----------------------------------------------------------------------------
# STAGE 1 — BUILDER
# ----------------------------------------------------------------------------
# Installs Python dependencies in an isolated layer.
# Build tools stay ONLY in this stage to keep runtime image small.
# ----------------------------------------------------------------------------

FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs appear immediately
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# System dependencies required for compiling Python packages
# ----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Virtual environment
# ----------------------------------------------------------------------------

ENV VENV_PATH=/opt/venv

RUN python -m venv ${VENV_PATH}

ENV PATH="${VENV_PATH}/bin:$PATH"

# ----------------------------------------------------------------------------
# Working directory
# ----------------------------------------------------------------------------

WORKDIR /build

# ----------------------------------------------------------------------------
# Install dependencies separately for better Docker layer caching
# ----------------------------------------------------------------------------

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt


# ----------------------------------------------------------------------------
# STAGE 2 — RUNTIME
# ----------------------------------------------------------------------------
# Clean lightweight runtime image.
# ----------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# Runtime system packages only
# ----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Non-root user
# ----------------------------------------------------------------------------

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser

# ----------------------------------------------------------------------------
# Virtual environment from builder
# ----------------------------------------------------------------------------

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:$PATH"

COPY --from=builder ${VENV_PATH} ${VENV_PATH}

# ----------------------------------------------------------------------------
# Application directory
# ----------------------------------------------------------------------------

WORKDIR /app

# ----------------------------------------------------------------------------
# Copy ONLY required application files
# ----------------------------------------------------------------------------

COPY app ./app
COPY feast_repo ./feast_repo
COPY training/sample_request.json ./training/sample_request.json

# Optional:
# needed only if runtime references these
COPY .env.example ./

# ----------------------------------------------------------------------------
# Permissions
# ----------------------------------------------------------------------------

RUN chown -R appuser:appgroup /app

USER appuser

# ----------------------------------------------------------------------------
# Expose FastAPI port
# ----------------------------------------------------------------------------

EXPOSE 8000

# ----------------------------------------------------------------------------
# Healthcheck
# ----------------------------------------------------------------------------
# Docker uses this to determine container health.
# ----------------------------------------------------------------------------

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8000/health || exit 1

# ----------------------------------------------------------------------------
# Production ASGI server
# ----------------------------------------------------------------------------
# gunicorn manages worker processes
# uvicorn handles ASGI execution
# ----------------------------------------------------------------------------

CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]