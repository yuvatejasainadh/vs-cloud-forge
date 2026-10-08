# CloudForge V2 — Canonical 7-VM VoiceShield Architecture

## 1. Executive Summary

VoiceShield V2 establishes the authoritative **Canonical 7-VM Topology** supporting **19 Indian and English languages** across 6 dedicated deepfake voice spoofing detector clusters and 1 central Language Identification (LID) orchestrator.

The architecture decouples model identity from infrastructure routing, establishing a model-agnostic inference platform where **CloudForge owns model acquisition, caching, verification, runtime loading, inference, observability, and deployment**. There is zero external Azure runtime dependency.

---

## 2. Canonical 7-VM Topology

```
                              ┌───────────────────────────┐
                              │  Client Audio Stream/File │
                              └─────────────┬─────────────┘
                                            │
                                            ▼
                              ┌───────────────────────────┐
                              │          VM1_LID          │
                              │ Language Identification   │
                              │ (vmc-lid :50051)          │
                              └─────────────┬─────────────┘
                                            │
           ┌───────────────┬────────────────┼───────────────┬───────────────┬───────────────┐
           │ (te, ta,      │ (hi, mr,       │ (bn, as,      │ (pa, ks,      │ (gu, raj,     │ (en-IN)
           │  ml, kn)      │  sa)           │  or, bpy)     │  sd, ur)      │  kut)         │
           ▼               ▼                ▼               ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  VM2_SOUTH  │ │  VM3_NORTH  │  │  VM4_EAST   │ │ VM5_FRONTIER│ │  VM6_WEST   │ │ VM7_ENGLISH │
    │ Dravidian   │ │ North Indic │  │ East Indic  │ │  Frontier   │ │ West Indic  │ │English AASIST│
    │  :50052     │ │  :50053     │  │  :50054     │ │  :50055     │ │  :50056     │ │  :50057     │
    └─────────────┘ └─────────────┘  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

---

## 3. VM1 Language Identification (LID) Architecture

VM1 operates as a multi-model, pluggable language identification engine with clear separation between backbones, classification heads, and benchmark models:

```
                                  VM1 LID
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
             Open-Vakgyata                      XLS-R 300M
               Benchmark                         Backbone
             (onecxi/open-             (facebook/wav2vec2-xls-r-300m)
              vakgyata)                              │
           Role: END_TO_END                   Role: BACKBONE
           Status: BENCHMARK                  Status: BACKBONE
             (10 Classes)                            │
                     │                        Future 19-Class
                     │                      Classification Head
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                            LID Decision Engine
                                     │
                          Canonical 19-Reg Registry
                                     │
                        Language Group Resolver
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
            VM2_SOUTH          VM5_FRONTIER         VM7_ENGLISH
```

### 3.1. Model Roles & Distinctions
- **Open-Vakgyata** (`onecxi/open-vakgyata`):
  - **Role**: `END_TO_END`
  - **Status**: `BENCHMARK`
  - **Task**: `LID` (10 Indian & English classes)
  - Preserved as the reference baseline model.
- **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`):
  - **Role**: `BACKBONE`
  - **Status**: `BACKBONE` (Must NOT be marked `ACTIVE` until trained 19-class head exists)
  - **Task**: `SPEECH_ENCODER` (300M parameters, 1024-dim hidden representations)
  - **Classification Head**: Pluggable interface (`LIDClassificationHead`). Returns `MODEL_HEAD_NOT_AVAILABLE` until a validated 19-language checkpoint is supplied.

---

## 4. Authoritative 19-Language Mapping

| Language | ISO Code | Language Family | Semantic Group | Target VM | Service Port | Model Coverage (Active LID) |
|---|---|---|---|---|---|---|
| **Telugu** | `te-IN` | Dravidian | `SOUTH` | `VM2_SOUTH` | `50052` | Supported |
| **Tamil** | `ta-IN` | Dravidian | `SOUTH` | `VM2_SOUTH` | `50052` | Supported |
| **Malayalam** | `ml-IN` | Dravidian | `SOUTH` | `VM2_SOUTH` | `50052` | Supported |
| **Kannada** | `kn-IN` | Dravidian | `SOUTH` | `VM2_SOUTH` | `50052` | Supported |
| **Hindi** | `hi-IN` | Indo-Aryan | `NORTH` | `VM3_NORTH` | `50053` | Supported |
| **Marathi** | `mr-IN` | Indo-Aryan | `NORTH` | `VM3_NORTH` | `50053` | Supported |
| **Sanskrit** | `sa-IN` | Indo-Aryan | `NORTH` | `VM3_NORTH` | `50053` | Unsupported (Candidate) |
| **Bengali** | `bn-IN` | Indo-Aryan | `EAST` | `VM4_EAST` | `50054` | Supported |
| **Assamese** | `as-IN` | Indo-Aryan | `EAST` | `VM4_EAST` | `50054` | Unsupported (Candidate) |
| **Odia** | `or-IN` | Indo-Aryan | `EAST` | `VM4_EAST` | `50054` | Supported |
| **Bishnupriya Manipuri** | `bpy-IN` | Indo-Aryan | `EAST` | `VM4_EAST` | `50054` | Unsupported (Candidate) |
| **Punjabi** | `pa-IN` | Indo-Aryan | `NORTH_WEST_FRONTIER` | `VM5_NORTH_WEST_FRONTIER` | `50055` | Unsupported (Candidate) |
| **Kashmiri** | `ks-IN` | Dardic / Indo-Aryan | `NORTH_WEST_FRONTIER` | `VM5_NORTH_WEST_FRONTIER` | `50055` | Unsupported (Candidate) |
| **Sindhi** | `sd-IN` | Indo-Aryan | `NORTH_WEST_FRONTIER` | `VM5_NORTH_WEST_FRONTIER` | `50055` | Unsupported (Candidate) |
| **Urdu** | `ur-IN` | Indo-Aryan | `NORTH_WEST_FRONTIER` | `VM5_NORTH_WEST_FRONTIER` | `50055` | Unsupported (Candidate) |
| **Gujarati** | `gu-IN` | Indo-Aryan | `WEST` | `VM6_WEST` | `50056` | Supported |
| **Rajasthani** | `raj-IN` | Indo-Aryan | `WEST` | `VM6_WEST` | `50056` | Unsupported (Candidate) |
| **Kutchi** | `kut-IN` | Indo-Aryan | `WEST` | `VM6_WEST` | `50056` | Unsupported (Candidate) |
| **English** | `en-IN` | Germanic | `ENGLISH` | `VM7_ENGLISH` | `50057` | Supported |

> **Note on Kutchi**: The canonical ISO code for Kutchi is `kut-IN`. `kfr-IN` is strictly deprecated and not used.

---

## 5. Key Architectural Subsystems

### 5.1. Central Model Registry (`models/registry.json`) & Artifact Resolver (`core/artifact_resolver.py`)
- Standardized metadata schema supporting roles (`BACKBONE`, `CLASSIFIER`, `DETECTOR`, `END_TO_END`, `BENCHMARK`).
- Multi-source artifact resolution: `LocalFilesystemSource`, `HuggingFaceArtifactSource` (with pinned commit revisions), and `ObjectStorageSource`.
- SHA-256 bit-exact integrity verification before runtime initialization.

### 5.2. Canonical Registry Resolver (`routing/canonical_registry.py`)
- Authoritative single source of truth for 19 languages, 6 detector clusters, and 7 VMs.
- Guaranteed startup validation: strictly enforces bidirectional bijection between languages, groups, and target VMs.
- Dynamic canonical ordering queryable via `canonical_registry.lid_labels()`.

### 5.3. Language-Aware Continuous Routing (LACR) (`routing/lacr.py`)
- Manages streaming call sessions across sequential chunks (`checkpoint_interval_chunks: 5`).
- Chunk 1: Evaluates LID -> locks target VM.
- Chunks 2–5: Routes directly to locked VM without redundant LID overhead.
- Chunk 6, 11, 16...: Checkpoint re-evaluation detects mid-call code-switching when confidence $\ge 0.80$.

### 5.4. Bandlimited Audio Preprocessor (`core/audio_processor.py`)
- Sinc polyphase resampling using `scipy.signal.resample_poly` with exact rational factor $\frac{\text{up}}{\text{down}}$.
- Strict zero-mean unit-variance normalization with zero-variance safeguards.
- Sanitizes `NaN` and `Inf` floating-point anomalies.

### 5.5. VM7 English Deepfake Detector (`services/vmc-english`)
- **CloudForge-Native Architecture**: Runs AASIST-L inference natively via ONNX Runtime without external cloud or third-party API dependencies.
- **Model Lifecycle**: Hydrates model from `model-cache/voiceshield-aasist-english/<version>/model.onnx`, validates SHA256 checksum, and loads onto configured execution provider (`CPUExecutionProvider` / `CUDAExecutionProvider`).
- **Historical SRS Reference Note**: The historical SRS referenced "Azure-hosted original AASIST-L" as a hosting location; current CloudForge architecture executes AASIST-L directly inside CloudForge native infrastructure.
