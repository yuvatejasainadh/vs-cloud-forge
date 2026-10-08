# syntax=docker/dockerfile:1
# CloudForge Production Container for VMC-LID (Open-Vakgyata)
# Multi-stage minimal non-root container

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/vmc-lid/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime audio libraries and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src:/app/src/gen:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create non-root user and group
RUN groupadd -g 65532 nonroot && \
    useradd -u 65532 -g nonroot -s /bin/false nonroot

# Copy application source, configs, and registry
COPY proto/ /app/proto/
COPY services/vmc-lid/configs/ /app/configs/
COPY services/vmc-lid/src/ /app/src/
COPY services/vmc-lid/cmd/ /app/cmd/
COPY services/vmc-lid/tools/ /app/tools/
COPY models/registry.json /app/models/registry.json

# Set ownership and permissions
RUN chown -R 65532:65532 /app

USER 65532:65532

EXPOSE 50051 9090

# Standard CloudForge Environment Variables
ENV CF_GRPC_PORT=50051 \
    CF_METRICS_PORT=9090 \
    LID_MODEL_ID=open-vakgyata \
    LID_MODELS_DIR=/app/models \
    LID_LANGUAGE_GROUPS_FILE=/app/configs/language_groups.yaml \
    LID_CONFIDENCE_THRESHOLD=0.40

# Health check probing Prometheus HTTP /healthz endpoint
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9090/healthz || exit 1

ENTRYPOINT ["python", "cmd/server.py"]
