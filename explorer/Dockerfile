# syntax=docker/dockerfile:1
FROM python:3.12-slim

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    ECG_RUNS_DIR=/app/models \
    ECG_DATA_DIR=/data \
    ECG_DATA_PATH=ptb-xl \
    ECG_DELIN_CACHE_DIR=/app/delineation_cache \
    ECG_DEFAULT_RUN=cem_softseg

WORKDIR /app

# Minimal system deps for numpy/scipy/torch + healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU by default; override TORCH_INDEX_URL for CUDA) + explorer deps
COPY explorer/requirements-explorer.txt /app/explorer/requirements-explorer.txt
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && pip install -r /app/explorer/requirements-explorer.txt

# Copy only source + explorer assets (models/ and data/ are mounted at runtime)
COPY src/ /app/src/
COPY explorer/ /app/explorer/
COPY config.yaml /app/

# Create mount points and run as non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/models /app/delineation_cache /data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8040

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8040/health || exit 1

CMD ["uvicorn", "app:app", "--app-dir", "/app/explorer", "--host", "0.0.0.0", "--port", "8040"]
