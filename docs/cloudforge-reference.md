# CloudForge Reference Service

## Overview

The `cloudforge-reference` service is the foundational microservice for the CloudForge platform. It demonstrates the complete lifecycle of a high-performance Go gRPC service running on Kubernetes.

## Service Contract

**Proto**: `services/cloudforge-reference/proto/reference.proto`  
**Package**: `cloudforge.reference.v1`  
**Service**: `ReferenceService`  

### RPC: `GetStatus`
- **Request**: `StatusRequest` (`service: string`)
- **Response**: `StatusResponse`
  - `service`: Name of the service
  - `status`: Service status (`SERVING`)
  - `version`: Version string (`v1.0.0`)
  - `platform`: Platform name (`cloudforge`)

### Health Checks
- Implements standard `grpc.health.v1.Health`
- Serving statuses registered:
  - `""` (overall process)
  - `cloudforge-reference` (short name)
  - `cloudforge.reference.v1.ReferenceService` (fully-qualified service name)

## Local Execution

### Build & Run
```powershell
cd C:\CloudForge\services\cloudforge-reference
go test -v ./...
go build -o bin/server.exe ./cmd/server
.\bin\server.exe
```

### CLI Client Validation
```powershell
go build -o bin/client.exe ./cmd/client
.\bin\client.exe -addr="localhost:50051"
.\bin\client.exe -addr="localhost:50051" -health
```

## Container Build

```powershell
docker build -t cloudforge/cloudforge-reference:v1.0.0 .
```

## Kubernetes Deployment

```powershell
# Load image into Kind
kind load docker-image cloudforge/cloudforge-reference:v1.0.0 --name cloudforge

# Apply manifests
kubectl apply -k C:\CloudForge\kubernetes\workloads\cloudforge-reference

# Validate Deployment and Service
kubectl get pods -n cloudforge-system -o wide
kubectl get svc -n cloudforge-system
```
