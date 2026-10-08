# CloudForge V2 — VM7_ENGLISH AASIST-L Deepfake Dataset Evaluation Report

**Evaluation Date:** October 7, 2026  
**Service:** `vmc-english` (VM7_ENGLISH)  
**Target Architecture:** AASIST-L  
**Model Identifier:** `voiceshield-aasist-english`  
**Evaluation Dataset:** `C:\Users\yuvat\Music\Deepfake Dataset`  
**Dataset Manifest:** `reports/deepfake_dataset_manifest.json`  
**Summary Data:** `reports/deepfake_dataset_evaluation_summary.json`  
**Final System Status:** `ARCHITECTURE READY | SERVICE READY | CLOUDFORGE RUNTIME READY | MODEL ARTIFACT BLOCKED | REAL INFERENCE BLOCKED`

---

## Executive Summary

This report documents the rigorous evaluation of the **CloudForge V2 VM7_ENGLISH (AASIST-L) Deepfake Detection Service** against the real audio deepfake dataset located at `C:\Users\yuvat\Music\Deepfake Dataset`.

In strict adherence to CloudForge scientific governance:
1. **Dataset Discovery & Cataloging:** Comprehensive discovery of all 12,000 real WAV audio samples (~13.33 hours) across four primary conditions and three SNR acoustic subconditions.
2. **Zero Fabrication & Anti-Hallucination Policy:** Model predictions, spoof probabilities, and classification metrics were not fabricated, synthetic dummy weights were not loaded, and dataset folder labels were not forwarded as model outputs.
3. **Model Artifact Inspection:** The verified ONNX weights (`model.onnx`) for `voiceshield-aasist-english` are currently unhydrated in the local model cache (`model-cache/voiceshield-aasist-english/v1.0.0/`).
4. **Runtime Status:** The CloudForge-native ONNX runtime and service pipeline are fully ready, verified, and correctly handle unhydrated states with strict error isolation (`STATUS_MODEL_UNAVAILABLE`).
5. **Preprocessing Throughput:** The native 16 kHz mono resampling, zero-mean unit-variance normalization, and AASIST fixed windowing pipeline (64,600 samples) was benchmarked on the real audio dataset, achieving **1,617.1 samples/sec** throughput with a median latency of **0.54 ms**.

---

## A. Dataset Inventory & Discovery

The dataset at `C:\Users\yuvat\Music\Deepfake Dataset` contains 12,000 single-channel WAV audio files.

| Category Directory | Description | Subcondition | File Count | Format | Sample Rate | Window |
|:---|:---|:---|---:|:---:|---:|---:|
| `fake_voice_fake_bg_wf_4s` | Synthetic Voice + Synthetic Background | `balanced`, `bg_dominant`, `voice_dominant` | 3,000 | WAV | 16 kHz | ~4.00s |
| `fake_voice_real_bg_wf_4s` | Synthetic Voice + Genuine Background | `balanced`, `bg_dominant`, `voice_dominant` | 3,000 | WAV | 16 kHz | ~4.00s |
| `real_voice_fake_bg_wf_4s` | Genuine Voice + Synthetic Background | `balanced`, `bg_dominant`, `voice_dominant` | 3,000 | WAV | 16 kHz | ~4.00s |
| `real_voice_real_bg_wf_4s` | Genuine Voice + Genuine Background | `balanced`, `bg_dominant`, `voice_dominant` | 3,000 | WAV | 16 kHz | ~4.00s |
| **TOTAL** | **4 Categories** | **3 Acoustic Subconditions** | **12,000** | **WAV** | **16 kHz** | **~13.33 hrs** |

---

## B. Dataset Labels & Ground Truth

Labels are obtained strictly from the canonical directory hierarchies:

```mermaid
flowchart TD
    subgraph DatasetRoot ["Deepfake Dataset (12,000 files)"]
        direction TB
        C1["fake_voice_fake_bg_wf_4s (3,000)"]
        C2["fake_voice_real_bg_wf_4s (3,000)"]
        C3["real_voice_fake_bg_wf_4s (3,000)"]
        C4["real_voice_real_bg_wf_4s (3,000)"]
    end

    subgraph Subconditions ["Acoustic SNR Subconditions"]
        S1["balanced (4,000 files)"]
        S2["bg_dominant (4,000 files)"]
        S3["voice_dominant (4,000 files)"]
    end

    subgraph GroundTruthClasses ["Ground Truth Classification"]
        GT_SPOOF["SPOOF (Synthetic Voice) : 6,000 samples"]
        GT_ENV_SPOOF["BONAFIDE_ENV_SPOOF (Real Voice / Fake BG) : 3,000 samples"]
        GT_BONAFIDE["BONAFIDE (Real Voice / Real BG) : 3,000 samples"]
    end

    C1 --> S1 & S2 & S3
    C2 --> S1 & S2 & S3
    C3 --> S1 & S2 & S3
    C4 --> S1 & S2 & S3

    C1 --> GT_SPOOF
    C2 --> GT_SPOOF
    C3 --> GT_ENV_SPOOF
    C4 --> GT_BONAFIDE
```

- **Corrupted / Unreadable Audio:** 0 files (100% integrity across all 12,000 WAV headers).
- **Audio Characteristics:** Mono single-channel, 16 kHz, 16-bit PCM, constant 4.00s duration (64,000 raw samples).

---

## C. Model Artifact Verification

Model metadata from `models/registry.json`:

```json
{
  "model_id": "voiceshield-aasist-english",
  "name": "VoiceShield AASIST-L English Detector",
  "role": "DETECTOR",
  "target_vm": "VM7_ENGLISH",
  "group": "ENGLISH",
  "language": "en-IN",
  "architecture": "AASIST-L",
  "runtime": "onnx",
  "framework": "onnx",
  "provider": "cloudforge",
  "task": "DEEPFAKE_DETECTION",
  "sample_rate": 16000,
  "channels": 1,
  "status": "CANDIDATE",
  "supported_languages": ["en-IN"],
  "artifact": {
    "source": "local_filesystem",
    "path": "model-cache/voiceshield-aasist-english/v1.0.0/model.onnx",
    "sha256": null
  }
}
```

- **Artifact Path on Disk:** `C:\CloudForge\model-cache\voiceshield-aasist-english\v1.0.0\model.onnx`
- **Artifact Existence:** `False` (Directory/file not present on disk)
- **Artifact Verification Status:** `AASIST-L ARTIFACT: NOT AVAILABLE`
- **Real Inference Status:** `REAL INFERENCE: BLOCKED`

---

## D. CloudForge Runtime Readiness

```text
================================================================================
CLOUDFORGE AASIST-L RUNTIME INSPECTION
================================================================================
Platform:              CloudForge Native (Zero Azure Runtime Dependencies)
Service Name:          vmc-english (VM7_ENGLISH)
Execution Provider:    CPUExecutionProvider
Intra-Op Threads:      4
Graph Optimization:    ORT_ENABLE_ALL
Runtime Status:        AWAITING_ARTIFACT (Loaded: False, Ready: False)
Error Isolation:       STATUS_MODEL_UNAVAILABLE (Code 6)
================================================================================
```

The runtime layer successfully enforces integrity boundaries:
1. `ArtifactResolver` detects missing model weights without crashing.
2. `AASISTLRuntime.load()` gracefully sets `_loaded = False` and logs `MODEL_LOAD_SKIPPED`.
3. `AASISTLRuntime.predict()` raises `ModelUnavailableError` when called unhydrated.
4. `EnglishDeepfakeDetector.analyze()` intercepts `ModelUnavailableError` and returns a clean gRPC `STATUS_MODEL_UNAVAILABLE` response with zero latency overhead.

---

## E. Real Inference Results & Anti-Fabrication Guarantees

```text
EVALUATED INFERENCE SAMPLES: 0
FABRICATED PREDICTIONS:      0 (DISABLED)
SYNTHETIC SCORES:            0 (DISABLED)
REAL INFERENCE STATUS:       BLOCKED
```

In accordance with CloudForge anti-fabrication guidelines, no synthetic probabilities or random numbers were generated. Real inference remains cleanly blocked until genuine trained AASIST-L ONNX weights are placed into `model-cache`.

---

## F. Classification Metrics

```text
TRUE POSITIVES (TP):   N/A (Inference Blocked)
TRUE NEGATIVES (TN):   N/A (Inference Blocked)
FALSE POSITIVES (FP):  N/A (Inference Blocked)
FALSE NEGATIVES (FN):  N/A (Inference Blocked)
ACCURACY:              N/A (Inference Blocked)
PRECISION / RECALL:    N/A (Inference Blocked)
ROC-AUC / EER:         N/A (Inference Blocked)
```

Classification metrics are not reported to prevent false claims of model accuracy on unhydrated weights.

---

## G. Threshold Calibration Status

```yaml
detector:
  threshold:
    value: 0.50
    calibrated: false
```

- **Threshold Value:** `0.50` (uncalibrated baseline default)
- **Threshold Calibration Status:** `NOT AVAILABLE`
- **Semantics:** Because `calibrated = false` and model weights are unhydrated, no production classification accuracy claims are made.

---

## H. Performance & Preprocessing Benchmarks

Evaluated on 120 real dataset audio files across all four categories:

| Metric | Measured Value | Standard Target | Status |
|:---|---:|---:|:---:|
| **P50 Preprocessing Latency** | **0.539 ms** | < 10.0 ms | **PASS** |
| **P95 Preprocessing Latency** | **1.189 ms** | < 25.0 ms | **PASS** |
| **P99 Preprocessing Latency** | **1.516 ms** | < 50.0 ms | **PASS** |
| **Mean Preprocessing Latency** | **0.618 ms** | < 15.0 ms | **PASS** |
| **Min / Max Latency** | **0.336 ms / 3.327 ms** | — | **PASS** |
| **Preprocessing Throughput** | **1,617.1 samples/sec** | > 100 samples/sec | **PASS** |

---

## I. Memory Footprint

| Metric | Value |
|:---|---:|
| **Initial Process RSS** | **105.37 MB** |
| **Final Process RSS** | **116.95 MB** |
| **Memory Delta** | **+11.58 MB** |
| **Leakage Detected** | **None** |

---

## J. VM7 English Routing Contract & Language Gating

The VM7 routing contract was verified using gRPC and detector unit tests:

```mermaid
flowchart TD
    Req["Incoming Inference Request"] --> Gating{"Language Gating\n(Language == en-IN?)"}
    Gating -- "Yes (en-IN / ENGLISH)" --> ValidAudio{"Audio Validation\n& Preprocessing"}
    Gating -- "No (e.g. hi-IN / NORTH)" --> Reject["STATUS_UNSUPPORTED_LANGUAGE\n(Code 7, Rejected)"]
    ValidAudio --> ModelCheck{"AASIST-L ONNX\nHydrated?"}
    ModelCheck -- "Yes" --> Infer["AASIST-L Forward Pass\n-> SPOOF / GENUINE"]
    ModelCheck -- "No" --> Unavail["STATUS_MODEL_UNAVAILABLE\n(Code 6, Clean Fail)"]
```

- **English (`en-IN` / `ENGLISH`):** Accepted by language gate $\rightarrow$ Routed to VM7 detector core.
- **Non-English (`hi-IN`, `te-IN`, etc.):** Rejected by language gate $\rightarrow$ `STATUS_UNSUPPORTED_LANGUAGE`.
- **Target VM:** `VM7_ENGLISH`
- **Target Service:** `vmc-english:50057`

---

## K. Failures & Exceptions

- **Dataset Corruptions:** 0 corrupted audio files out of 12,000 files.
- **Pipeline Crashes:** 0 unhandled exceptions.
- **Unhandled Failures:** 0.

---

## L. Final Production Readiness & Certification Status

```text
================================================================================
ARCHITECTURE READY:          VERIFIED
SERVICE READY:               VERIFIED
CLOUDFORGE RUNTIME READY:    VERIFIED
MODEL ARTIFACT:              BLOCKED (Awaiting verified ONNX weights in model-cache)
REAL INFERENCE:              BLOCKED (No synthetic scores permitted)
ACCURACY VALIDATED:          PENDING ARTIFACT HYDRATION
PRODUCTION READY:            SERVICE ARCHITECTURE COMPLETE (Awaiting weights)
================================================================================
```

- **Automated Test Suite Status:** **29/29 Unit and Integration Tests Passing** (`pytest services/vmc-english/tests -v`)
- **Quality Gate:** Passed all platform verification checks (`scripts/verify-all.ps1 -Service vmc-english`)
