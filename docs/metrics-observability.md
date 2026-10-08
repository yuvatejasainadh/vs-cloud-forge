# CloudForge Metrics & Observability Standard

## Overview

CloudForge establishes an out-of-the-box telemetry standard for all application and ML microservices.

Telemetry Model: **Prometheus-Native Exporters with OpenTelemetry Compatibility**.

---

## 1. Metrics Architecture

Each CloudForge service exposes a dedicated HTTP server on port `9090` (`/metrics`) running concurrently with the main gRPC server (port `50051`).

```
                    ┌─────────────────────────┐
                    │  Kubernetes Pod         │
                    │                         │
[gRPC Clients] ────►│ :50051 (gRPC Server)    │
                    │   ├─ Interceptors       │
                    │   │    └─ Metrics Hook  │
                    │                         │
[Prometheus / OTel] ►│ :9090  (Prometheus HTTP)│
                    │   └─ /metrics           │
                    │   └─ /healthz           │
                    └─────────────────────────┘
```

---

## 2. Standard Platform Metric Catalog

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `cloudforge_grpc_requests_total` | Counter | `service`, `method`, `code` | Total count of gRPC requests. |
| `cloudforge_grpc_request_duration_seconds` | Histogram | `service`, `method` | gRPC latency distribution. |
| `cloudforge_grpc_errors_total` | Counter | `service`, `method`, `code` | Total count of failed gRPC calls. |
| `cloudforge_service_info` | Gauge (1) | `service`, `version`, `platform`, `environment` | Static service build metadata. |
| `cloudforge_service_health_status` | Gauge (0/1) | `service` | Real-time health status (1 = SERVING). |
| `cloudforge_ml_inference_duration_seconds` | Histogram | `service`, `model`, `version` | Dedicated ML model inference latency for VMC. |

---

## 3. Kubernetes Scraping Standard

Deployments define standard Prometheus annotations:
```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    prometheus.io/path: "/metrics"
```
CoreDNS and Prometheus Operator service monitors automatically discover the service without external agents.
