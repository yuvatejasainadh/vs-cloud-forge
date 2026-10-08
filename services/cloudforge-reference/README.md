# CloudForge Reference Service

Production-ready reference microservice implementing the CloudForge gRPC standards.

## Project Structure

```
.
├── cmd/
│   ├── server/          # Main gRPC server daemon
│   └── client/          # Verification CLI tool
├── internal/
│   ├── config/          # Configuration loader
│   ├── service/         # ReferenceService RPC handler
│   └── server/          # gRPC lifecycle, health, reflection
├── proto/               # Protobuf definition
├── gen/                 # Generated protobuf & gRPC Go stubs
├── Dockerfile           # Distroless multi-stage Docker build
├── .dockerignore
├── go.mod
└── go.sum
```

## Quick Start

### Testing
```powershell
go test -v ./...
```

### Build & Run Locally
```powershell
go build -o bin/server.exe ./cmd/server
go build -o bin/client.exe ./cmd/client

# Start server
.\bin\server.exe

# In another terminal: test GetStatus and Health
.\bin\client.exe -addr="localhost:50051"
.\bin\client.exe -addr="localhost:50051" -health
```

### Build Container
```powershell
docker build -t cloudforge/cloudforge-reference:v1.0.0 .
```
