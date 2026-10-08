# CloudForge Service Golden Template Guide

## Purpose

The **CloudForge Reference Service** (`services/cloudforge-reference`) serves as the production golden template for all subsequent microservices onboarded to CloudForge (e.g., `VMC-LID`, `VMC-ASR`, `VMC-RISK`, `VoiceShield Backend`).

Follow this lifecycle for creating and deploying new services.

---

## 1. Directory Layout Standard

```
services/<service-name>/
├── cmd/
│   ├── server/
│   │   └── main.go         # Application entrypoint, signal handling, graceful shutdown
│   └── client/
│       └── main.go         # Lightweight test/CLI client
├── internal/
│   ├── config/
│   │   ├── config.go       # Environment-driven configuration
│   │   └── config_test.go
│   ├── service/
│   │   ├── service.go      # Hand-written gRPC service business logic
│   │   └── service_test.go
│   └── server/
│       ├── server.go       # gRPC server setup, health server, reflection
│       └── server_test.go
├── proto/
│   └── <service>.proto     # Versioned Protobuf definition
├── gen/
│   └── <service>/v1/       # Generated protobuf & gRPC Go stubs (DO NOT EDIT manually)
├── Dockerfile              # Multi-stage minimal non-root build
├── .dockerignore
├── go.mod
└── go.sum
```

---

## 2. Protobuf Code Generation

On Windows systems with Device Guard / custom paths:
```powershell
protoc --plugin=protoc-gen-go="$env:USERPROFILE\go\bin\protoc-gen-go.exe" `
       --plugin=protoc-gen-go-grpc="$env:USERPROFILE\go\bin\protoc-gen-go-grpc.exe" `
       --proto_path=proto `
       --go_out=. `
       --go-grpc_out=. `
       proto/<service>.proto
```

---

## 3. Containerization Standard

All containers must:
- Use multi-stage builds (`golang:alpine` -> `gcr.io/distroless/static-debian12:nonroot`).
- Run as non-root user `65532:65532`.
- Include CA certificates and timezone database.
- Build statically (`CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s"`).

---

## 4. Kubernetes Manifest Standard

Store manifests under `kubernetes/workloads/<service-name>/`:
- `deployment.yaml`:
  - `nodeSelector`: enforce `workload: application` or `workload: ml`.
  - `securityContext`: `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `drop: [ALL]`.
  - `resources`: explicit `requests` and `limits`.
  - `readinessProbe` & `livenessProbe`: native Kubernetes `grpc` probes.
- `service.yaml`: `ClusterIP` with named ports.
- `kustomization.yaml`: declarative bundle.

---

## 5. Deployment Verification Checklist

1. Run unit/integration tests: `go test -v ./...`
2. Build container image: `docker build -t cloudforge/<service-name>:v1.0.0 .`
3. Load image into Kind: `kind load docker-image cloudforge/<service-name>:v1.0.0 --name cloudforge`
4. Apply manifests: `kubectl apply -k kubernetes/workloads/<service-name>`
5. Verify rollout: `kubectl rollout status deployment/<service-name> -n <namespace>`
6. Verify node placement: `kubectl get pods -n <namespace> -o wide`
7. Verify gRPC health & status endpoints via port-forward.
