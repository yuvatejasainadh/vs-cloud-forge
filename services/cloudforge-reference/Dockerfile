# Multi-stage build for CloudForge Reference Service
# Stage 1: Build stage
FROM golang:alpine AS builder

WORKDIR /app

# Install certificates and timezone data
RUN apk add --no-cache ca-certificates tzdata

# Cache go modules
COPY go.mod go.sum ./
RUN go mod download

# Copy source tree
COPY . .

# Build statically compiled binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o /bin/cloudforge-reference \
    ./cmd/server

# Stage 2: Runtime stage using distroless nonroot
FROM gcr.io/distroless/static-debian12:nonroot

WORKDIR /app

# Copy CA certificates and timezone database
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy compiled binary
COPY --from=builder /bin/cloudforge-reference /app/cloudforge-reference

# Run as non-root user (65532:65532)
USER 65532:65532

# Default environment configuration
ENV PORT=50051 \
    METRICS_PORT=9090 \
    METRICS_ENABLED=true \
    GRPC_MAX_RECV_MSG_SIZE_MB=32 \
    GRPC_MAX_SEND_MSG_SIZE_MB=32 \
    SERVICE_NAME=cloudforge-reference \
    SERVICE_VERSION=v1.0.0 \
    PLATFORM_NAME=cloudforge \
    ENVIRONMENT=local

EXPOSE 50051 9090

ENTRYPOINT ["/app/cloudforge-reference"]
