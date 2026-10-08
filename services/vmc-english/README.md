# VMC-ENGLISH — VM7 VoiceShield English AASIST-L Detector

## 1. Overview
`vmc-english` is the dedicated CloudForge-native deepfake voice spoofing detection microservice for **VM7 — English Detector** in the VoiceShield Canonical 7-VM Architecture.

All model inference, hydration, verification, and execution run entirely through **CloudForge ML infrastructure** without external cloud or third-party API dependencies.

---

## 2. Canonical Topology
- **VM Identity**: `VM7_ENGLISH`
- **Language Group**: `ENGLISH`
- **Supported Language**: `en-IN` (English - India)
- **Task**: Deepfake Voice Spoofing Detection
- **Model Architecture**: AASIST-L (Integrated Spectro-Temporal Graph Attention Networks - Lightweight variant)
- **Serving gRPC Port**: `50057`
- **Prometheus Metrics / HTTP Health Port**: `9090`

---

## 3. Ingestion & Routing Flow
```
               VM1 LID (vmc-lid :50051)
                         │
                      en-IN
                         │
                         ▼
             Canonical Group Resolver
                         │
                      ENGLISH
                         │
                         ▼
             VM7_ENGLISH (vmc-english :50057)
                         │
                         ▼
             CloudForge Model Registry
                         │
                         ▼
               AASIST-L Model Artifact
                         │
                         ▼
             CloudForge AASIST-L Runtime
                         │
                         ▼
              Deepfake Decision Engine
                         │
                         ▼
                REAL / SPOOF / UNCERTAIN
```

---

## 4. CloudForge Model Lifecycle
1. **REGISTER**: Model defined in `models/registry.json` as `voiceshield-aasist-english`.
2. **RESOLVE & HYDRATE**: `ArtifactResolver` resolves artifact from `model-cache/voiceshield-aasist-english/<version>/model.onnx`.
3. **CHECKSUM VERIFY**: Validates SHA256 integrity before loading.
4. **LOAD & SERVE**: `AASISTLRuntime` initializes ONNX Runtime session with configurable execution provider (`CPUExecutionProvider` / `CUDAExecutionProvider`).
5. **OBSERVE**: Exports Prometheus telemetry and health probes.

---

## 5. Decision & Threshold Logic
- Output: Softmax classification over bonafide vs spoof classes.
- Threshold: Configurable via `detector.threshold.value` (default: 0.50, `calibrated: false`).
- Language Gating: Rejects non-English languages with `DETECTION_STATUS_UNSUPPORTED_LANGUAGE`.

---

## 6. Security & Hardening
- Container runs as non-root user `65532:65532`.
- All Linux capabilities dropped (`capabilities.drop: ["ALL"]`).
- Read-only root filesystem compatible.
- Strict audio data redaction in JSON structured logs.

---

## 7. Historical Reference Note
- **Historical SRS Reference**: "Azure-hosted original AASIST-L" (legacy hosting note).
- **Current CloudForge Architecture**: CloudForge-native hosted AASIST-L runtime.
