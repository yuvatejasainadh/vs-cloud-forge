# CloudForge V2 — Real Dataset E2E Validation Report

**Dataset**: WaveFake English Speech Synthesis / Deepfake Dataset (`C:\Users\yuvat\Downloads\sources_wavefake\_sources_wavefake`)  
**Evaluation Date**: `2026-10-06T17:43:28Z`  
**Scope**: `2007` samples evaluated in `1044.97s` (`1.92 QPS`)

---

## 1. Executive Summary

This report delivers the end-to-end empirical validation of the **CloudForge VoiceShield V2 English Detection Pipeline**:
```text
Real Audio (WaveFake) ➔ VM1 LID (Open-Vakgyata) ➔ Canonical Language Registry ➔ VM7_ENGLISH (vmc-english) ➔ AASIST-L
```

All routing decisions and latency metrics reflect **actual runtime execution** without mocked paths or fabricated data.

### Validation Matrix
| Component | Operational Role | Status | Notes |
|---|---|---|---|
| **VM1 LID** | Language Identification | **VALIDATED** | Evaluated on real WaveFake speech audio (100% English detection rate). |
| **Canonical Routing** | ISO -> Group -> VM7 | **VALIDATED** | `en-IN` strictly resolved to `ENGLISH` -> `VM7_ENGLISH`. Invariant 100% verified (2,007/2,007). |
| **VM7 AASIST-L Runtime** | English Deepfake Detector | **BLOCKED (Awaiting Artifact)** | Service & runtime architecture ready; weights unhydrated in model-cache. |
| **Real Dataset Pipeline** | End-to-End Execution | **PARTIALLY VALIDATED** | LID + Routing 100% operational; Detector awaits ONNX artifact hydration. |

---

## 2. Dataset Overview & Discovery

- **Dataset Path**: `C:\Users\yuvat\Downloads\sources_wavefake\_sources_wavefake`
- **Total Audio Files Discovered**: `2007`
- **Valid Audio Files**: `2007` (`100.0%`)
- **Invalid Audio Files**: `0`
- **Total Audio Duration**: `14402.49s` (`4.00 hours`)
- **Duration Statistics**: Min = `4.00s`, Max = `10.10s`, Mean = `7.18s`
- **Sample Rate Distribution**: `{16000: 2007}` (100% 16kHz)
- **Channel Distribution**: `{1: 2007}` (100% Mono)
- **Format Distribution**: `{'.wav': 2007}` (100% WAV PCM_16)
- **Ground Truth Distribution**:
  - `SPOOF` (`fake_voice` TTS synthesis): `1000` (49.8%)
  - `BONAFIDE` (`real_voice` LJSpeech human speech): `1007` (50.2%)

---

## 3. VM1 Language Identification (LID) Results

- **Model ID**: `open-vakgyata` (version: `f2754058e485dfc65cc62589b8d0e21c1d328399`)
- **Samples Evaluated**: `2007`
- **English (`en-IN`) Predictions**: `2007` (`100.00%`)
- **Non-English Predictions**: `0` (`0.00%`)
- **Language Prediction Breakdown**:
  - `en-IN`: 2007 (100.00%)

### LID Latency Profile
- **P50 (Median)**: `511.48 ms`
- **P95**: `762.82 ms`
- **P99**: `933.48 ms`
- **Mean**: `518.97 ms`

---

## 4. Canonical Routing Invariant Validation

```text
en-IN  ───>  ENGLISH  ───>  VM7_ENGLISH (Port 50057)
```

- **`en-IN` → `ENGLISH` Group Invariant**: `2007` / `2007` (**100.0%**)
- **`ENGLISH` → `VM7_ENGLISH` Target VM Invariant**: `2007` / `2007` (**100.0%**)
- **Routing Success Rate**: `100.00%`
- **Routing Invariant Violations**: `0`
- **Routing Latency**: P50 = `0.0014 ms`, P95 = `0.0020 ms`

---

## 5. VM7 English AASIST-L Detector Availability & Results

- **Target VM**: `VM7_ENGLISH` (`vmc-english`)
- **Model ID**: `voiceshield-aasist-english` (version: `v1.0.0`)
- **Artifact Availability Gate**: `BLOCKED_AWAITING_ARTIFACT`
- **Runtime Execution Status**: `UNHYDRATED`
- **Samples Routed to VM7**: `2007`
- **Inference Status**: As mandated by production governance rules, because the genuine AASIST-L ONNX model artifact is currently awaiting placement in `model-cache/voiceshield-aasist-english/v1.0.0/model.onnx`, **no fake or mocked deepfake predictions were generated**. The gate reported `MODEL_UNAVAILABLE` honestly.

---

## 6. End-to-End Latency Profile

| Stage | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
|---|---|---|---|---|
| **Audio Preprocessing & VM1 LID** | `511.48` | `762.82` | `933.48` | `518.97` |
| **Canonical Routing Resolution** | `0.0014` | `0.0020` | `0.0024` | `0.0015` |
| **End-to-End Pipeline** | `512.73` | `765.08` | `935.47` | `520.65` |

---

## 7. Error Classification & Non-English Handling

```json
{}
```

---

## 8. Final Status Assessment

```text
VM1 LID             = VALIDATED
Canonical Routing   = VALIDATED
VM7 AASIST-L        = BLOCKED (Awaiting Model Artifact)
Real Dataset        = PARTIALLY VALIDATED
E2E Pipeline        = PARTIALLY VALIDATED
```
