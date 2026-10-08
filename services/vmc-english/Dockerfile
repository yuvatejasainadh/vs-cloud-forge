# Multi-stage secure build for CloudForge VM7 English Deepfake Detector (vmc-english)
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runtime image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user 65532:65532
RUN groupadd -g 65532 nonroot && \
    useradd -u 65532 -g nonroot -s /bin/false -m nonroot

COPY --from=builder /root/.local /home/nonroot/.local
ENV PATH=/home/nonroot/.local/bin:$PATH
ENV PYTHONPATH=/app/src:/app/src/gen:$PYTHONPATH

COPY src /app/src
COPY configs /app/configs

RUN chown -R nonroot:nonroot /app

USER 65532:65532

EXPOSE 50057 9090

ENTRYPOINT ["python", "-u", "src/server/grpc_server.py"]
