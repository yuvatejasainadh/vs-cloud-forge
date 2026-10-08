# CloudForge — Open-Vakgyata LID Integration & Architecture

## Overview

**Open-Vakgyata** is integrated into CloudForge as the foundational Voice Model Core (VMC) Language Identification (LID) runtime. CloudForge acts as the generic ML platform, keeping the core platform decoupled from any single concrete model or framework.

```
                    CloudForge Platform
                             │
                     ┌───────▼────────┐
                     │ Model Registry │
                     └───────┬────────┘
                             │
                             ▼
                    Model Definition
                     (Metadata/Config)
                             │
                             ▼
                    ModelRuntimeFactory
                             │
                             ▼
                    OpenVakgyataRuntime
                      (ONNX Runtime)
                             │
                             ▼
                        VMC-LID
                    (gRPC :50051 / HTTP :9090)
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
             Language Code         Confidence
               (e.g., te-IN)         (e.g., 0.98)
                  │                     │
                  ▼                     ▼
         Language Group Resolver   Threshold Gate
                  │
                  ▼
              7VM Router
```

---

## 1. Model Specifications

- **Model ID**: `open-vakgyata`
- **Hugging Face Repository**: `onecxi/open-vakgyata`
- **Model Version / Commit**: `f2754058e485dfc65cc62589b8d0e21c1d328399`
- **Model SHA256**: `c4cc68450a7e571c21acb6928553f5cbba3e5f136d554210c07f5fbbe37d1d79`
- **Parameter Count**: 58,746,794 (~58.7M parameters)
- **Artifact Size**: 224.19 MB (235,086,936 bytes)
- **Architecture**: Wav2Vec2 Sequence Classification (`Wav2Vec2ForSequenceClassification`)
- **Execution Backend**: ONNX Runtime (`CPUExecutionProvider`, extensible to CUDA/TensorRT)
- **Input Format**: 16,000 Hz, 1-channel mono, 16-bit PCM or WAV

### Supported Indian Languages (10 ISO Codes)

| Index | ISO Code | Language Name | Default 7VM Group |
|---|---|---|---|
| 0 | `en-IN` | English (India) | `GROUP_05` |
| 1 | `hi-IN` | Hindi | `GROUP_01` |
| 2 | `or-IN` | Odia | `GROUP_02` |
| 3 | `bn-IN` | Bengali | `GROUP_02` |
| 4 | `ta-IN` | Tamil | `GROUP_04` |
| 5 | `te-IN` | Telugu | `GROUP_03` |
| 6 | `kn-IN` | Kannada | `GROUP_03` |
| 7 | `ml-IN` | Malayalam | `GROUP_04` |
| 8 | `mr-IN` | Marathi | `GROUP_01` |
| 9 | `gu-IN` | Gujarati | `GROUP_01` |

---

## 2. End-to-End Model Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│ Registry │ ──► │ Download │ ──► │ Validate │ ──► │  Cache   │ ──► │   Load   │ ──► │ Monitor │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘     └─────────┘
```

1. **Registry**: `models/registry.json` defines canonical model metadata, parameter counts, framework definitions, and remote artifact URLs.
2. **Download & Cache**: `services/vmc-lid/tools/download_model.py` downloads artifacts with 16-worker parallel HTTP Range chunks, caching locally in `models/<model_id>/<version>/`.
3. **Validation**: Model checksums (SHA256) are calculated and validated against remote ETags.
4. **Load**: `OpenVakgyataRuntime` initializes an optimized ONNX session with graph optimizations enabled.
5. **Serve**: The gRPC service serves `IdentifyLanguage` and `GetModelInfo` with 32MB payload support.
6. **Monitor**: Exposes Prometheus metrics on `:9090/metrics` and deep health verification on `:9090/healthz`.

---

## 3. Audio Preprocessing Pipeline

```
Raw Audio Stream (PCM/WAV)
       │
       ▼
1. Validation (Reject empty or malformed buffers)
       │
       ▼
2. Multi-channel to Mono Conversion (Channel averaging)
       │
       ▼
3. Linear Resampling to 16,000 Hz
       │
       ▼
4. Wav2Vec2 Normalization: (x - mean) / sqrt(variance + 1e-7)
       │
       ▼
5. Tensor Construction: input_values [1, N], attention_mask [1, N]
```

---

## 4. 7VM Integration & Routing Layer

The model runtime strictly outputs the raw `language` code and `confidence`. Decoupled from the runtime, the `LanguageGroupResolver` maps language codes to 7VM routing targets via `services/vmc-lid/configs/language_groups.yaml`:

```yaml
language_to_group:
  hi-IN: "GROUP_01"  # Indo-Aryan North & Central
  mr-IN: "GROUP_01"
  gu-IN: "GROUP_01"
  bn-IN: "GROUP_02"  # Indo-Aryan East
  or-IN: "GROUP_02"
  te-IN: "GROUP_03"  # Dravidian South-Central
  kn-IN: "GROUP_03"
  ta-IN: "GROUP_04"  # Dravidian South
  ml-IN: "GROUP_04"
  en-IN: "GROUP_05"  # Indian English
```

---

## 5. Confidence Handling & Operational Statuses

- **`LID_STATUS_SUCCESS`**: Inference succeeded and `confidence >= threshold` (default: `0.40`).
- **`LID_STATUS_LOW_CONFIDENCE`**: Inference completed but `confidence < threshold`.
- **`LID_STATUS_INVALID_AUDIO`**: Input audio is empty, truncated, or corrupt.
- **`LID_STATUS_INFERENCE_ERROR`**: Model execution failure or unexpected runtime error.
- **`LID_STATUS_UNKNOWN`**: Prediction unmapped to supported languages.

---

## 6. Telemetry & Metrics Catalog

Exposed on port `9090` (`/metrics`):

| Metric | Type | Description |
|---|---|---|
| `lid_inference_total` | Counter | Total inferences labeled by `model_id`, `model_version`, `status`, `language` |
| `cloudforge_ml_inference_duration_seconds` | Histogram | Latency distribution labeled by `service`, `model`, `version` |
| `lid_inference_latency` | Histogram | Dedicated LID latency histogram |
| `lid_inference_errors` | Counter | Errors labeled by `model_id`, `error_type` |
| `lid_low_confidence_total` | Counter | Count of inferences falling below the threshold |
| `lid_model_load_time_seconds` | Gauge | Model memory loading duration in seconds |
| `lid_model_load_failures_total` | Counter | Model load failures count |
| `cloudforge_service_info` | Gauge | Static service metadata (`service`, `version`, `platform`, `environment`) |
| `cloudforge_service_health_status` | Gauge | Real-time health gauge (1 = SERVING, 0 = NOT_SERVING) |

---

## 7. Performance Benchmarks (Measured Locally)

Measured on local AMD64 workstation using `services/vmc-lid/tools/benchmark.py`:

```
================================================================================
Model:                 onecxi/open-vakgyata (ONNX Quantized)
Parameters:            58.7M
Artifact Size:         224.19 MB
Runtime Backend:       ONNX Runtime (CPUExecutionProvider, 4 Threads)
--------------------------------------------------------------------------------
Model Load Time:       ~403 ms (0.40 s)
Model RAM Overhead:    ~233 MB
First Inference:       ~101 ms
Warm Latency (P50):    ~97.6 ms
Warm Latency (P95):    ~104.2 ms
Warm Latency (P99):    ~106.3 ms
Throughput:            ~10.2 QPS (Single process CPU)
Total RAM (RSS):       ~347 MB
================================================================================
```

---

## 8. Security & Governance

- **Zero Raw Audio in Logs**: All logging interceptors and formatters scrub and redact raw voice buffers (`<REDACTED_AUDIO_PAYLOAD>`).
- **Least Privilege**: Runs as non-root user `65532:65532` in Docker and Kubernetes.
- **Deterministic Checksums**: Model artifacts are verified against SHA256 checksums before loading.
- **Payload Bound**: gRPC limits capped at 32MB to prevent unbounded memory exhaustion while accommodating multi-channel audio bursts.
