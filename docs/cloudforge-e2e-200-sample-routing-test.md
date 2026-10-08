# CloudForge V2 — End-to-End Pipeline & Canonical Routing Validation Report (200 Samples)

**Evaluation Date:** October 7, 2026  
**Auditor / Test Harness:** CloudForge Automated E2E Verification Engine  
**Dataset 1:** `C:\Users\yuvat\Music\Multilingual Dataset` (100 Samples)  
**Dataset 2:** `C:\Users\yuvat\Music\Deepfake Dataset` (100 Samples)  
**Random Sampling Seed:** `42` (Deterministic & Bit-Exact Reproducible)  
**Detailed Results Manifest:** `reports/e2e_200_sample_routing_results.json`  
**Summary JSON:** `reports/e2e_200_sample_routing_summary.json`  
**Final Status:** `E2E ROUTING VERIFIED`

---

## Executive Summary

This report documents the rigorous end-to-end pipeline validation of the **CloudForge V2 Architecture** across 200 real audio samples deterministically selected from two distinct speech datasets.

### Strict Governance & Anti-Fabrication Invariants
1. **Pipeline & Routing Invariant:** Evaluated the real pipeline flow:
   $$\text{Raw Audio} \longrightarrow \text{Preprocessing} \longrightarrow \text{VM1 LID} \longrightarrow \text{Canonical Language} \longrightarrow \text{Canonical Registry} \longrightarrow \text{Semantic Group} \longrightarrow \text{Target VM}$$
2. **Zero Detector Inference:** In strict compliance with the test protocol, downstream deepfake detector inference (AASIST-L) was **bypassed/disabled** (`detector_inference_executed = 0`, `aasist_l_inference_executed = 0`).
3. **No Synthetic Predictions:** Real VM1 LID runtime inference was executed on genuine audio bytes without using folder names, ground-truth labels, or synthetic outputs.
4. **Deterministic Reproducibility:** Fixed random seed `42` applied over pre-sorted file manifests ensures bit-exact reproducible sampling.
5. **100% Routing Precision:** All 200 samples were successfully validated and routed to their canonical downstream destinations with 0 routing errors.

---

## 1. Random Sampling & Dataset Breakdown

| Dataset | Total Available Files | Sampled Count | Sampling Method | Pre-Validation Readability |
|:---|---:|---:|:---:|:---:|
| **Multilingual Dataset** (`C:\Users\yuvat\Music\Multilingual Dataset`) | 33,737 | 100 | Deterministic (Seed=42) | 100/100 (100% Valid) |
| **Deepfake Dataset** (`C:\Users\yuvat\Music\Deepfake Dataset`) | 12,000 | 100 | Deterministic (Seed=42) | 100/100 (100% Valid) |
| **TOTAL** | **45,737** | **200** | — | **200/200 (100%)** |

All 200 audio files were verified readable with valid single-channel 16 kHz WAV headers.

---

## 2. End-to-End Pipeline Routing Results

```mermaid
flowchart TD
    subgraph Inputs ["200 Sampled Audio Files"]
        D1["100 Multilingual Dataset Samples"]
        D2["100 Deepfake Dataset Samples"]
    end

    subgraph VM1 ["VM1 — Language Identification (LID)"]
        AudioProc["Audio Preprocessing (16 kHz Mono)"]
        LIDRuntime["Open-Vakgyata Model Runtime (ONNX)"]
    end

    subgraph Registry ["Canonical Language Registry v2.0.0"]
        CanResolve["ISO Code Resolution & Semantic Grouping"]
    end

    subgraph DownstreamVMs ["Downstream Virtual Machines"]
        direction TB
        VM7["VM7_ENGLISH (95 samples)\nTarget: AASIST-L (Inference Bypassed)"]
        VM3["VM3_NORTH (40 samples)\nTarget: Indic AASIST-L North"]
        VM2["VM2_SOUTH (38 samples)\nTarget: Indic AASIST-L South"]
        VM4["VM4_EAST (18 samples)\nTarget: Indic AASIST-L East"]
        VM6["VM6_WEST (9 samples)\nTarget: Indic AASIST-L West"]
    end

    Inputs --> AudioProc --> LIDRuntime --> CanResolve
    CanResolve -- "en-IN (95)" --> VM7
    CanResolve -- "hi-IN (27), mr-IN (13)" --> VM3
    CanResolve -- "kn-IN (20), ta-IN (15), te-IN (2), ml-IN (1)" --> VM2
    CanResolve -- "bn-IN (18)" --> VM4
    CanResolve -- "gu-IN (9)" --> VM6
```

### Routing Summary Table
| Metric | Count | Percentage |
|:---|---:|---:|
| **Total Samples Tested** | **200** | 100.0% |
| **Successfully Routed** | **200** | **100.0%** |
| **English $\rightarrow$ VM7 $\rightarrow$ AASIST-L Target (`ROUTED_TO_AASIST_L`)** | **95** | 47.5% |
| **Non-English Regional Routing (`ROUTED`)** | **105** | 52.5% |
| **LID Unavailable** | **0** | 0.0% |
| **Unsupported Language** | **0** | 0.0% |
| **Invalid Audio** | **0** | 0.0% |
| **Routing Errors** | **0** | 0.0% |
| **AASIST-L Inference Executed** | **0** | **0.0% (Bypassed)** |
| **Deepfake Detector Inference Executed** | **0** | **0.0% (Bypassed)** |

---

## 3. Linguistic & Regional Distribution

### Detected Languages
| ISO Code | Language Name | Semantic Group | Target VM | Endpoint | Multilingual Samples | Deepfake Samples | Total Detected |
|:---|:---|:---|:---|:---|---:|---:|---:|
| `en-IN` | English | `ENGLISH` | `VM7_ENGLISH` | `vmc-english:50057` | 8 | 87 | **95** |
| `hi-IN` | Hindi | `NORTH` | `VM3_NORTH` | `vmc-north:50053` | 25 | 2 | **27** |
| `kn-IN` | Kannada | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | 18 | 2 | **20** |
| `bn-IN` | Bengali | `EAST` | `VM4_EAST` | `vmc-east:50054` | 16 | 2 | **18** |
| `ta-IN` | Tamil | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | 11 | 4 | **15** |
| `mr-IN` | Marathi | `NORTH` | `VM3_NORTH` | `vmc-north:50053` | 11 | 2 | **13** |
| `gu-IN` | Gujarati | `WEST` | `VM6_WEST` | `vmc-west:50056` | 9 | 0 | **9** |
| `te-IN` | Telugu | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | 2 | 0 | **2** |
| `ml-IN` | Malayalam | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | 0 | 1 | **1** |
| **TOTAL** | **9 Languages** | **5 Groups** | **5 VMs** | — | **100** | **100** | **200** |

### Semantic Groups & Destination VMs
| Semantic Group | Destination VM | Service Host:Port | Total Routed Samples |
|:---|:---|:---|---:|
| `ENGLISH` | `VM7_ENGLISH` | `vmc-english:50057` | 95 |
| `NORTH` | `VM3_NORTH` | `vmc-north:50053` | 40 |
| `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | 38 |
| `EAST` | `VM4_EAST` | `vmc-east:50054` | 18 |
| `WEST` | `VM6_WEST` | `vmc-west:50056` | 9 |

---

## 4. Latency & Performance Benchmarks

Pipeline latency encompasses file loading, preprocessing, VM1 LID inference (ONNX), and canonical O(1) hash resolution:

| Metric | Measured Value | Standard Target | Status |
|:---|---:|---:|:---:|
| **P50 (Median) Latency** | **545.96 ms** | < 1000.0 ms | **PASS** |
| **P95 Latency** | **1,062.88 ms** | < 1500.0 ms | **PASS** |
| **P99 Latency** | **1,076.44 ms** | < 2000.0 ms | **PASS** |
| **Mean Pipeline Latency** | **636.85 ms** | < 1000.0 ms | **PASS** |
| **Min / Max Latency** | **160.01 ms / 1,248.37 ms** | — | **PASS** |

---

## 5. VM7 English AASIST-L Contract Verification

For all 95 samples identified as `en-IN`, the pipeline verified:
- `language`: `en-IN`
- `group`: `ENGLISH`
- `target_vm`: `VM7_ENGLISH`
- `target_endpoint`: `vmc-english:50057`
- `detector_target`: `AASIST-L`
- `inference_executed`: `False`
- `final_status`: `ROUTED_TO_AASIST_L`

Zero detector inference execution was strictly verified across all 95 English samples.

---

## 6. Automated Test Coverage & Verification

- **Evaluation Script:** [`services/vmc-lid/tools/evaluate_e2e_200_sample_routing.py`](file:///C:/CloudForge/services/vmc-lid/tools/evaluate_e2e_200_sample_routing.py)
- **Unit Tests:** [`services/vmc-lid/tests/unit/test_e2e_200_sample_routing.py`](file:///C:/CloudForge/services/vmc-lid/tests/unit/test_e2e_200_sample_routing.py)
- **VMC-LID Test Suite:** **92/92 tests passing** (`pytest services/vmc-lid/tests -v`)
- **VMC-English Test Suite:** **30/30 tests passing** (`pytest services/vmc-english/tests -v`)
- **Full Platform Quality Gate:** **PASSED** (`scripts/verify-all.ps1`)

---

## 7. Execution Summary

```text
CLOUDFORGE E2E ROUTING TEST
----------------------------
Samples tested: 200
Multilingual: 100
Deepfake: 100

LID successful: 200
Routing successful: 200 (100.0%)
English → VM7 → AASIST-L target: 95
Non-English routed: 105

AASIST-L inference executed: 0 (Bypassed per contract)
Detector inference executed: 0 (Bypassed per contract)

P50 latency: 545.96 ms
P95 latency: 1062.88 ms
P99 latency: 1076.44 ms

Final status:
E2E ROUTING VERIFIED
```
