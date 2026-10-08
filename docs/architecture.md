# CloudForge Platform Architecture — V2 (Canonical 7-VM Topology)

## Overview

CloudForge is a production-grade ML inference and routing platform designed for real-time speech AI applications.
In VoiceShield V2, CloudForge provides the foundational infrastructure for language identification (LID) and deepfake audio spoofing detection across 19 canonical languages.

```
+-------------------------------------------------------------------------+
|                               CloudForge                                |
|                                                                         |
|  +-------------------+      +------------------+     +---------------+  |
|  |  Model Registry   | ---> |  Model Runtimes  | --> | Decision /    |  |
|  |  (models/reg.json)|      |  (ONNX/PyTorch)  |     | Routing Engine|  |
|  +-------------------+      +------------------+     +---------------+  |
|                                                              |          |
|                                                              v          |
|  +--------------------------------------------------------------------+  |
|  |                    Canonical 7-VM Topology                         |  |
|  |                                                                    |  |
|  |  VM1_LID (Port 50051) -> Central Language Identification & LACR   |  |
|  |  VM2_SOUTH (Port 50052) -> Telugu, Tamil, Malayalam, Kannada       |  |
|  |  VM3_NORTH (Port 50053) -> Hindi, Marathi, Sanskrit               |  |
|  |  VM4_EAST (Port 50054)  -> Bengali, Assamese, Odia, Bishnupriya   |  |
|  |  VM5_NORTH_WEST_FRONTIER (50055) -> Punjabi, Kashmiri, Sindhi, Urdu|  |
|  |  VM6_WEST (Port 50056)  -> Gujarati, Rajasthani, Kutchi            |  |
|  |  VM7_ENGLISH (Port 50057) -> English (Azure AASIST-L)             |  |
|  +--------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

## Core Standards
- **Protobuf / gRPC Contracts**: `proto/vmc/lid/v1/lid.proto` & `proto/vmc/detector/v1/detector.proto`.
- **Max Message Size**: 32 MB for full high-fidelity streaming audio buffers.
- **Port Standards**:
  - `50051`: VM1 LID Service
  - `50052–50057`: VM2–VM7 Deepfake Detector Clusters
  - `9090`: Prometheus Telemetry & Metrics
- **Security**: Root-forbidden runtime execution (UID/GID `65532`), all Linux capabilities dropped.
- **Resampling**: Polyphase sinc bandlimited resampling (`scipy.signal.resample_poly`).
