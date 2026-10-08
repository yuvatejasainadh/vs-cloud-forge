# CloudForge Protobuf Governance & API Strategy

## Overview

CloudForge microservices and ML inference services communicate across language boundaries using versioned **Protocol Buffers (v3)** over **gRPC**. To ensure zero-breaking-change backwards/forwards compatibility across VoiceShield and VMC, CloudForge mandates a structured schema repository model.

---

## Directory Hierarchy Standard

All protobuf definitions are organized by domain, service, and semantic version:

```
proto/
├── buf.yaml                      # Root Buf configuration (for linting & breaking checks)
├── cloudforge/                   # Core Platform Infrastructure Domain
│   └── reference/
│       └── v1/
│           └── reference.proto
├── vmc/                          # Voice Model Core Domain (ML Services)
│   ├── lid/
│   │   └── v1/
│   │       └── lid.proto         # Language Identification Service
│   ├── asr/
│   │   └── v1/
│   │       └── asr.proto         # Automated Speech Recognition Service
│   ├── risk/
│   │   └── v1/
│   │       └── risk.proto        # Risk Analysis & Fraud Engine
│   └── identity/
│       └── v1/
│           └── identity.proto    # Voice Biometrics & Identity Service
└── voiceshield/                  # VoiceShield Application Domain
    ├── backend/
    │   └── v1/
    │       └── backend.proto     # VoiceShield Core Backend API
    └── realtime/
        └── v1/
            └── realtime.proto    # Realtime Audio Gateway
```

---

## Semantic Versioning Rules

1. **Package Naming Convention**:
   - `package <domain>.<service>.<major_version>;`
   - Examples: `cloudforge.reference.v1`, `vmc.lid.v1`, `vmc.asr.v1`, `voiceshield.backend.v1`.

2. **No In-Place Breaking Changes**:
   - Field tags, once assigned, are immutable.
   - Do not delete fields or change field numbers/types. Use `reserved` for deprecated tags.
   - Breaking contract modifications MUST be introduced under a new version package (e.g. `v2`).

3. **Request/Response Wrapping**:
   - Every RPC MUST use explicit message wrappers (e.g. `GetStatus(StatusRequest) returns (StatusResponse)`), never primitive scalar types or `google.protobuf.Empty` as top-level parameters, to allow future non-breaking field expansion.

---

## Architecture Decision: Buf vs Standard Protoc

| Dimension | Option A: Buf Schema Registry / CLI | Option B: Repository Hierarchy + Protoc Standard | Selected Decision |
|---|---|---|---|
| **Local Tooling** | Requires `buf` binary installation | Standard `protoc` + official plugins | **Option B for Local Dev** (Zero friction) |
| **Breaking Checks** | `buf breaking` | Manual review | **Enabled via root `buf.yaml` for CI** |
| **Windows WDAC** | May trigger application control blocks | Explicit path invocation | **Explicit plugin path standard** |

### Decision Summary:
For the local development phase, developers generate code using standard `protoc` (via `scripts/gen-proto.ps1`). A root `buf.yaml` is maintained so CI/CD pipelines can run automated linting and breaking-change detection without imposing external tool dependencies on local developer workstations.
