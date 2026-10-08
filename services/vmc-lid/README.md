# CloudForge VMC-LID (Language Identification Service)

## Overview

The `vmc-lid` service is the Voice Model Core (VMC) Language Identification microservice for CloudForge (VM1_LID). It provides a decoupled, model-agnostic ML platform hosting both the benchmark **Open-Vakgyata** LID model and the foundational **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`) speech encoder backbone with pluggable 19-class classification heads.

CloudForge natively owns model acquisition, deterministic caching, cryptographic verification, runtime loading, inference execution, and observability with zero external cloud dependencies.

---

## Service Specification

- **Service Port (gRPC)**: `50051`
- **Metrics / Health Port (HTTP)**: `9090`
- **Task**: `LANGUAGE_IDENTIFICATION` (`VM1_LID`)
- **Proto Contract**: `proto/vmc/lid/v1/lid.proto` (`vmc.lid.v1.LIDService`)
- **Models Hosted**:
  1. **Open-Vakgyata** (`onecxi/open-vakgyata`):
     - Role: `END_TO_END`
     - Status: `BENCHMARK`
     - Parameters: 58.7M (~235 MB)
     - Supported Languages (10): `en-IN`, `hi-IN`, `or-IN`, `bn-IN`, `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `gu-IN`
  2. **Facebook Wav2Vec2 XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`):
     - Role: `BACKBONE`
     - Status: `BACKBONE`
     - Parameters: 317.4M (~1.2 GB)
     - Task: `SPEECH_ENCODER` (1024-dim representations)
     - Pinned Revision: `1a640f32ac3e39899438a2931f9924c02f080a54`
     - 19-Class Classification Head: Pluggable (`LinearClassificationHead`, `MLPClassificationHead`, `ONNXClassificationHead`, `NoClassificationHead`)

---

## Local Development & Usage

### 1. Run Complete Test Suite (66 Tests)
```powershell
python -m pytest services/vmc-lid/tests -v
```

### 2. Run Open-Vakgyata Benchmark
```powershell
python services/vmc-lid/tools/benchmark.py -n 20
```

### 3. Run XLS-R 300M Backbone Benchmark
```powershell
python services/vmc-lid/tools/benchmark_xlsr.py --iterations 10
```

### 4. Start Local Server
```powershell
python services/vmc-lid/cmd/server.py
```

### 5. Validate with CLI Client
```powershell
# Health check
python services/vmc-lid/tools/client.py --health

# Query active model metadata
python services/vmc-lid/tools/client.py --info

# Send inference request
python services/vmc-lid/tools/client.py --infer
```

---

## Endpoints

- `gRPC: /vmc.lid.v1.LIDService/IdentifyLanguage`: Accepts 16kHz PCM/WAV audio bytes and returns predicted ISO language code, confidence, and target 7-VM cluster.
- `gRPC: /vmc.lid.v1.LIDService/GetModelInfo`: Returns model metadata, framework, parameter count, and role.
- `gRPC: /vmc.lid.v1.LIDService/GetSupportedLanguages`: Returns all canonical 19 VoiceShield languages and active model supported languages.
- `gRPC: /vmc.lid.v1.LIDService/ResolveLanguageGroup`: Resolves ISO language code to canonical group and target VM.
- `gRPC: /grpc.health.v1.Health/Check`: Standard gRPC health check.
- `HTTP: http://localhost:9090/healthz`: Deep readiness probe (checks model artifact & runtime readiness).
- `HTTP: http://localhost:9090/metrics`: Prometheus metrics exporter.
