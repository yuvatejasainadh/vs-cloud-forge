# VoiceShield CloudForge

> Cloud-native speech intelligence and deepfake analysis infrastructure for VoiceShield.

CloudForge is the backend inference and infrastructure foundation of the **VoiceShield** platform. It provides language identification, language-aware routing, speech representation, deepfake detection, model governance, telemetry, gRPC service infrastructure, and the foundation for future risk intelligence and threat/call decisioning.

This repository currently represents the **pre-RIAS CloudForge baseline**. The existing implementation is intentionally separated into verified capabilities, partial implementations, scaffolds, blocked components, and planned components. This README describes repository reality rather than treating architecture diagrams or mock services as production functionality.

---

## 1. Current Status

**Current phase:** Pre-RIAS infrastructure baseline

| Area | Current State |
|---|---|
| Audio validation / preprocessing | ✅ Implemented + validated |
| VM1 Open-Vakgyata LID | ✅ Implemented + validated |
| LACR | ✅ Implemented + validated |
| Canonical 19-language resolver | ✅ Implemented + validated |
| DSR / XLS-R 300M | 🟡 Partial: real backbone, no 19-class head |
| VM2 SOUTH | 🟡 Scaffold |
| VM3 NORTH | 🟡 Scaffold |
| VM4 EAST | 🟡 Scaffold |
| VM5 NORTH-WEST / FRONTIER | 🟡 Scaffold |
| VM6 WEST | 🟡 Scaffold |
| VM7 ENGLISH / AASIST-L | 🟠 Real model inference, calibration incomplete |
| RIAS | 🔴 Not implemented yet |
| TCED | 🔴 Not implemented yet |
| Distributed fan-out/orchestration | 🔴 Not implemented yet |
| Android reporting contract | 🔴 Not implemented yet |
| Production cloud deployment | 🔴 Not deployed |

**Evidence baseline:** `docs/cloudforge-full-implementation-audit.md` and `reports/cloudforge-full-implementation-audit.json`.

---

## 2. CloudForge Mission

CloudForge is designed to turn incoming call/audio data into structured VoiceShield intelligence.

The long-term architecture is:

```text
                         VOICESHIELD ANDROID APP
                                  │
                                  │ audio / call stream
                                  ▼
                       ┌──────────────────────────┐
                       │   CLOUD / CLOUDFORGE    │
                       │ Audio Validation &       │
                       │ Preprocessing            │
                       └────────────┬─────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
               ┌───────┐        ┌───────┐        ┌───────┐
               │  LID  │        │  DSR  │        │ RIAS  │
               │ VM1   │        │ XLS-R │        │ Risk  │
               └───┬───┘        └───┬───┘        └───┬───┘
                   │                │                 │
                  LACR              │          Risk Assessment
                   │                │                 │
                   ▼                ▼                 │
              VM Routing        AASIST-L              │
                   │                │                 │
                   │         Deepfake Evidence        │
                   └───────────────┬──────────────────┘
                                   │
                                  TCED
                                   │
                                   ▼
                          Android App Report
```

### Architectural rule

**RIAS is a parallel intelligence pipeline.** It is not a mandatory sequential stage after AASIST-L. LID/LACR, DSR/deepfake analysis, and RIAS are independent intelligence branches whose evidence is intended to converge at the decision layer.

---

## 3. Canonical 7-VM Architecture

| VM | Identity | Responsibility | Status |
|---|---|---|---|
| VM1 | `VM1_LID` | Language identification + LACR | Implemented + validated |
| VM2 | `VM2_SOUTH` | Telugu/Tamil/Malayalam/Kannada detection | Scaffold |
| VM3 | `VM3_NORTH` | Hindi/Marathi/Sanskrit detection | Scaffold |
| VM4 | `VM4_EAST` | Bengali/Assamese/Odia/Bishnupriya Manipuri detection | Scaffold |
| VM5 | `VM5_NORTH_WEST_FRONTIER` | Punjabi/Kashmiri/Sindhi/Urdu detection | Scaffold |
| VM6 | `VM6_WEST` | Gujarati/Rajasthani/Kutchi detection | Scaffold |
| VM7 | `VM7_ENGLISH` | English AASIST-L detection | Partially implemented |

The seven VMs are the canonical VoiceShield routing architecture. Their existence in the repository does **not** imply that every VM currently has a real detector model or a production-ready service.

---

## 4. Canonical Language Registry

```text
SOUTH
  te-IN  Telugu
  ta-IN  Tamil
  ml-IN  Malayalam
  kn-IN  Kannada

NORTH
  hi-IN  Hindi
  mr-IN  Marathi
  sa-IN  Sanskrit

EAST
  bn-IN   Bengali
  as-IN   Assamese
  or-IN   Odia
  bpy-IN  Bishnupriya Manipuri

NORTH_WEST_FRONTIER
  pa-IN  Punjabi
  ks-IN  Kashmiri
  sd-IN  Sindhi
  ur-IN  Urdu

WEST
  gu-IN   Gujarati
  raj-IN  Rajasthani
  kut-IN  Kutchi

ENGLISH
  en-IN  English
```

The canonical registry defines **19 language codes**. Model availability is a separate concern: the current Open-Vakgyata LID baseline itself is a 10-class model.

---

## 5. Pipeline Responsibilities

| Pipeline | Responsibility | Status |
|---|---|---|
| Audio ingestion/validation | WAV/PCM validation, channel handling | ✅ |
| Audio preprocessing | Resampling, normalization, sanitization | ✅ |
| VM1 LID | Identify language | ✅ |
| LACR | Continuous language-aware routing/session state | ✅ |
| Canonical resolver | Language → group → VM | ✅ |
| DSR / XLS-R | Speech representation | 🟡 |
| Regional detectors | Deepfake detection by language group | 🟡 scaffolds |
| VM7 AASIST-L | English deepfake detection | 🟠 |
| RIAS | Risk identification and assessment | 🔴 |
| TCED | Threat/call evaluation and decision | 🔴 |
| Orchestrator | Concurrent fan-out and aggregation | 🔴 |
| Android report | Final client contract | 🔴 |
| Observability | Metrics/health/telemetry | ✅ |
| Model governance | Registry/artifact integrity | ✅ |

---

## 6. VM1 — Open-Vakgyata LID

**Service:** `vmc-lid`

**Model:** `onecxi/open-vakgyata`

**Runtime:** ONNX Runtime

Current validated 10-class language set:

```text
en-IN
hi-IN
or-IN
bn-IN
ta-IN
te-IN
kn-IN
ml-IN
mr-IN
gu-IN
```

The repository contains a real model artifact, runtime implementation, preprocessing path, gRPC service, tests, and evaluation tooling.

The audited baseline records approximately **97.6 ms P50** warm inference latency for the Open-Vakgyata benchmark path.

Open-Vakgyata is the current **LID model**. XLS-R is not treated as the LID classifier.

---

## 7. LACR — Language-Aware Continuous Routing

LACR maintains language-aware session state and connects language intelligence to canonical routing.

Current functionality includes:

- canonical language normalization
- semantic group resolution
- target VM resolution
- chunk-level session tracking
- temporal checkpointing
- language code-switch handling where the required confidence/state conditions are met

Conceptually:

```text
Detected Language
       ↓
Canonical Code
       ↓
Semantic Group
       ↓
Target VM
```

Example:

```text
te-IN → SOUTH → VM2_SOUTH
en-IN → ENGLISH → VM7_ENGLISH
```

---

## 8. DSR / XLS-R 300M

CloudForge integrates the real `facebook/wav2vec2-xls-r-300m` speech backbone.

Current role:

```text
DSR = Deepfake Speech/Spectrogram Representation component
```

Current capabilities:

- real XLS-R 300M weights
- Transformers/PyTorch runtime integration
- masked mean pooling
- 1024-dimensional utterance representation
- artifact/hash tracking

Current limitation:

```text
19-class XLS-R classification head = unavailable
```

Therefore the repository must not claim that XLS-R currently performs 19-language classification. The backbone provides representation; a trained classification head is a separate requirement.

---

## 9. VM2–VM6 Regional Detectors

The regional services are currently architectural scaffolds.

### VM2 SOUTH

```text
te-IN / ta-IN / ml-IN / kn-IN
```

### VM3 NORTH

```text
hi-IN / mr-IN / sa-IN
```

### VM4 EAST

```text
bn-IN / as-IN / or-IN / bpy-IN
```

### VM5 NORTH-WEST / FRONTIER

```text
pa-IN / ks-IN / sd-IN / ur-IN
```

### VM6 WEST

```text
gu-IN / raj-IN / kut-IN
```

The current scaffold detector services may return hardcoded/mock scores. Those values are **not real ML detections** and are not to be treated as production results.

Missing for these regional detectors includes real model artifacts, complete executable service entrypoints, containerization where absent, test suites, and real inference validation.

---

## 10. VM7 — English AASIST-L

**Service:** `vmc-english`

**Language:** `en-IN`

**Detector:** AASIST-L

CloudForge contains a real AASIST-L ONNX artifact and an ONNX Runtime inference path.

Current capabilities:

- real ONNX model loading
- deterministic detector input processing
- forward inference
- detector metadata
- artifact integrity verification
- service/runtime code

### Important limitation

The production threshold is not empirically calibrated yet.

Therefore:

```text
AASIST-L inference works
≠
production decision threshold is calibrated
```

The audited baseline records the default threshold as `0.50` with `calibrated: false`. This must not be represented as a production-calibrated operating point.

---

## 11. RIAS — Risk Identification & Assessment System

**Current status: Not implemented.**

The audit found no current RIAS source package, protobuf contract, scoring engine, taxonomy implementation, service implementation, model, or tests.

RIAS is the **next major CloudForge development track**.

Its intended responsibility is to provide independent risk intelligence rather than duplicate language identification or deepfake detection.

Expected conceptual outputs include:

```text
Risk Score
Risk Category
Severity
Risk Indicators
Explanation Tags
Confidence
```

The exact risk taxonomy and implementation contract are intentionally not assumed here; they should be formally designed as part of the RIAS track.

---

## 12. TCED — Threat/Call Evaluation & Decision Engine

**Current status: Not implemented.**

TCED is intended as the decision-fusion layer that eventually combines evidence from:

```text
LID / LACR
DSR / Deepfake Detector
RIAS
Call/session context
Confidence / severity
```

Its output will form the basis for the final VoiceShield app decision/report.

No production TCED implementation exists in the current repository.

---

## 13. gRPC Architecture

CloudForge uses protobuf/gRPC as the service contract foundation.

Current audited services include:

```text
cloudforge-reference
vmc-lid
vmc-english
```

Current service port plan:

```text
Reference:    50051
VM1 LID:      50051
VM2 SOUTH:    50052
VM3 NORTH:    50053
VM4 EAST:     50054
VM5 FRONTIER: 50055
VM6 WEST:     50056
VM7 ENGLISH:  50057
Metrics:      9090
```

Port `50051` is used by different standalone services in their respective local contexts; actual distributed deployment/service discovery will define production endpoints.

### Current gRPC limitation

There is no completed distributed inter-VM orchestration mesh yet. Existing gRPC must be distinguished from a real distributed end-to-end pipeline.

A future orchestrator is expected to do:

```text
Incoming audio
      │
      ├──→ LID
      ├──→ DSR
      └──→ RIAS
              │
              ▼
          Evidence aggregation
              │
              ▼
             TCED
              │
              ▼
        Android report
```

---

## 14. Audio Processing

Reusable audio processing currently includes:

- WAV/header validation
- PCM validation
- mono/stereo conversion
- polyphase resampling to 16 kHz
- zero-mean/unit-variance normalization
- NaN/Inf sanitization
- deterministic detector window preparation

The preprocessing layer is part of the reusable foundation for LID and detector services.

---

## 15. Model Registry

CloudForge maintains structured model metadata under the model registry.

The registry records concepts such as:

- model ID
- task
- architecture
- framework/runtime
- version/revision
- VM/group
- artifact location
- status

Representative roles include:

```text
END_TO_END
BACKBONE
DETECTOR
CANDIDATE
PLANNED
```

A registry entry alone does not mean that the corresponding model artifact is available or that inference has been validated.

---

## 16. Model Artifacts and Integrity

CloudForge uses SHA-256 verification for model artifacts.

The audited baseline contains these real artifacts:

| Model | Role | Format | State |
|---|---|---|---|
| Open-Vakgyata | LID | ONNX | Real + loadable + validated |
| XLS-R 300M | Speech backbone | PyTorch weights | Real + loadable; no 19-class head |
| AASIST-L | Deepfake detector | ONNX | Real + loadable; threshold uncalibrated |

Large model binaries should not be added to source control merely because they exist locally. Model cache/artifact policy should continue to rely on explicit tracking, registries, hashes, and external storage when appropriate.

---

## 17. Observability

CloudForge includes Prometheus-compatible telemetry.

Current metrics include categories for:

- gRPC request counts
- gRPC request duration
- gRPC errors
- service health
- ML inference duration
- model load time
- LID inference count
- LID inference latency
- low-confidence cases

Metrics and health endpoints use port `9090` in the current service design.

```text
/metrics
/healthz
```

---

## 18. Security Baseline

Current service/Kubernetes security patterns include:

- non-root execution
- UID/GID `65532:65532`
- read-only root filesystem where configured
- dropped Linux capabilities
- raw audio redaction from logs
- gRPC payload limits
- SHA-256 model integrity verification

### Not yet implemented

Production gRPC authentication/TLS/mTLS is not currently implemented. Internal service communication is therefore not yet equivalent to a production authenticated service mesh.

---

## 19. Deployment Status

Current repository infrastructure supports local and container/Kubernetes development for the implemented services.

```text
Local development       ✅
Core Dockerfiles        ✅ where implemented
Local Kubernetes/Kind   ✅ scaffolding
AWS/EKS live deployment ❌ not implemented in repository
```

CloudForge must not be described as currently deployed to AWS/EKS based only on Kubernetes manifests.

---

## 20. Testing and Evaluation

The repository includes unit, integration, routing, model, gRPC, benchmark, and dataset evaluation tooling.

The audited baseline recorded:

```text
VMC-LID:        92 tests passed
VMC-English:    30 tests passed
CloudForge Go:   6 tests passed
```

Evaluation tooling also covers:

- multilingual routing
- 200-sample routing/E2E selection
- WaveFake
- Deepfake Dataset
- model runtime validation

### Testing governance

The following must never be conflated:

```text
Routing test
≠
ML inference test

Local function call
≠
Distributed gRPC E2E

Mock detector
≠
Real model detector

Model forward pass
≠
Calibrated production decision
```

---

## 21. Dataset Baseline

### Multilingual Dataset

Audited baseline:

```text
33,737 WAV files
~62.59 hours
16 dataset languages
12 directly usable canonical-language matches
```

### Deepfake Dataset

Audited baseline:

```text
12,000 WAV files
~13.33 hours
4 categories
```

### WaveFake

Audited baseline:

```text
2,007 WAV files
```

These datasets support evaluation tooling; dataset presence does not imply that every CloudForge detector model exists or has been trained/calibrated.

---

## 22. What Is Actually Working

### Verified working

- audio format validation and preprocessing
- 16 kHz resampling and normalization
- VM1 Open-Vakgyata LID inference
- canonical 19-language resolver
- LACR session/routing logic
- XLS-R 1024-dimensional representation extraction
- AASIST-L ONNX loading and forward inference
- model registry
- SHA-256 artifact verification
- Prometheus metrics/health infrastructure
- standalone VM1 gRPC infrastructure
- CloudForge reference gRPC infrastructure
- automated test suites

### Implemented but incomplete

- DSR/XLS-R without the 19-class classification head
- VM7 AASIST-L service without empirical threshold calibration
- broader Kubernetes workload set
- regional detector interfaces

### Scaffolded

- VM2 SOUTH detector
- VM3 NORTH detector
- VM4 EAST detector
- VM5 FRONTIER detector
- VM6 WEST detector
- distributed inter-VM mesh

### Not implemented

- RIAS
- TCED
- realtime fan-out/orchestrator
- final Android reporting contract
- real regional detector models
- production 19-class XLS-R classifier head
- production gRPC mTLS/authentication
- live AWS/EKS deployment infrastructure

---

## 23. Current Strongest Verified Pipeline

The strongest current repository path is:

```text
Raw Audio
   │
   ▼
Audio Validation / Preprocessing
   │
   ▼
VM1 Open-Vakgyata LID
   │
   ▼
LACR / Canonical Resolver
   │
   ▼
Semantic Group
   │
   ▼
Target VM Resolution
   │
   ├──────────────→ VM2–VM6
   │                 (scaffolds)
   │
   └──────────────→ VM7 English
                      │
                      ▼
                   AASIST-L
                   (real inference,
                    threshold uncalibrated)
```

This is the current repository reality. It is not yet the complete production VoiceShield pipeline.

---

## 24. RIAS Starting Point

CloudForge is intentionally entering the **RIAS track** after establishing the infrastructure baseline.

RIAS should reuse existing CloudForge infrastructure where appropriate:

```text
AudioProcessor
ArtifactResolver
Model Registry
Telemetry
Logging
gRPC server patterns
Health/readiness patterns
Protobuf governance
Kubernetes security baseline
```

RIAS should not duplicate:

```text
VM1 LID
LACR
Deepfake detection
AASIST-L
Regional detector responsibilities
```

RIAS should remain a separate risk-intelligence service and contribute its assessment to the future decision layer.

---

## 25. Target Decision Architecture

After the missing orchestration, RIAS, TCED, and detector capabilities are implemented, the target is:

```text
                         CALL / AUDIO
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
                   LID        DSR       RIAS
                    │         │         │
                   LACR       │     Risk Intelligence
                    │         │         │
                    ▼         ▼         │
                 VM Routing  Detector   │
                    │         │         │
                    └─────────┴────┬────┘
                                   │
                                   ▼
                                  TCED
                                   │
                                   ▼
                       VoiceShield Android Report
```

The core principle is:

```text
LID/LACR ──────────────┐
DSR/Detector ──────────┼──→ TCED
RIAS ──────────────────┘
```

---

## 26. Repository Structure

```text
CloudForge/
├── services/
│   ├── cloudforge-reference/
│   ├── vmc-lid/
│   ├── vmc-south/
│   ├── vmc-north/
│   ├── vmc-east/
│   ├── vmc-frontier/
│   ├── vmc-west/
│   └── vmc-english/
│
├── proto/
│   ├── cloudforge/
│   └── vmc/
│
├── models/
│   ├── open-vakgyata/
│   ├── huggingface/
│   └── registry.json
│
├── model-cache/
├── configs/
├── kubernetes/
├── infrastructure/
├── benchmarks/
├── reports/
├── docs/
├── scripts/
└── README.md
```

The repository will evolve as additional services and pipelines are implemented.

---

## 27. Engineering Principles

### Evidence before claims

A directory, proto, manifest, mock implementation, or documentation entry is not proof of a production capability.

### No fabricated ML results

CloudForge must never fabricate model predictions, confidence values, benchmark numbers, unsupported-language results, or missing detector outputs.

When a model is unavailable, the system should expose an explicit unavailable/error state.

### Real artifacts and integrity

Model artifacts should be traceable through explicit metadata, version/revision information, and SHA-256 verification.

### Separation of responsibilities

```text
LID      → language identification
LACR     → language-aware routing/session intelligence
DSR      → speech representation
Detector → deepfake/spoof evidence
RIAS     → risk intelligence
TCED     → threat/call decision fusion
Report   → Android-facing result
```

---

## 28. Development Workflow

A normal CloudForge implementation cycle is:

```text
1. Define architecture / protobuf contract
2. Implement service
3. Register models/artifacts
4. Add unit tests
5. Add integration tests
6. Validate real inference where applicable
7. Add telemetry
8. Validate container/runtime
9. Validate service boundaries / gRPC
10. Document evidence and limitations
11. Commit the verified baseline
```

Large dataset evaluations should remain separate from lightweight repository validation.

---

## 29. Running Current Core Tests

### VM1 / VMC-LID

```powershell
$env:PYTHONPATH = "C:\CloudForge\services\vmc-lid\src"
python -m pytest services/vmc-lid/tests -v
```

### VM7 / VMC-English

```powershell
$env:PYTHONPATH = "C:\CloudForge\services\vmc-english\src"
python -m pytest services/vmc-english/tests -v
```

### Go reference service

```powershell
go test -v ./...
```

Repository-wide verification helpers are available under `scripts/`.

---

## 30. Important Documentation

```text
docs/architecture.md
docs/canonical-7vm-architecture.md
docs/open-vakgyata-lid.md
docs/xlsr-300m-integration-report.md
docs/cloudforge-e2e-200-sample-routing-test.md
docs/xlsr-12-language-routing-evaluation.md
docs/xlsr-multilingual-dataset-evaluation.md
docs/vm7-aasist-l-deepfake-dataset-evaluation.md
docs/metrics-observability.md
docs/protobuf-governance.md
docs/cloudforge-full-implementation-audit.md
reports/cloudforge-full-implementation-audit.json
```

The full implementation audit is the authoritative snapshot for current repository status.

---

## 31. Roadmap

### Phase 1 — Core CloudForge Foundation

- [x] Audio preprocessing
- [x] VM1 Open-Vakgyata
- [x] LACR
- [x] Canonical registry
- [x] Model registry
- [x] Artifact integrity
- [x] Telemetry
- [x] Core gRPC infrastructure
- [x] VM7 AASIST-L runtime

### Phase 2 — RIAS

- [ ] Define RIAS architecture
- [ ] Define risk taxonomy
- [ ] Define `risk.proto`
- [ ] Implement RIAS service
- [ ] Implement risk scoring/assessment
- [ ] Add telemetry
- [ ] Add tests
- [ ] Validate RIAS independently
- [ ] Integrate RIAS into the parallel intelligence architecture

### Phase 3 — TCED

- [ ] Define TCED contract
- [ ] Fuse LID/LACR evidence
- [ ] Fuse deepfake evidence
- [ ] Consume RIAS assessment
- [ ] Define threat categories
- [ ] Define call decision policy
- [ ] Produce structured decision output

### Phase 4 — Regional Detectors

- [ ] VM2 model
- [ ] VM3 model
- [ ] VM4 model
- [ ] VM5 model
- [ ] VM6 model
- [ ] Real inference validation
- [ ] Calibration
- [ ] Distributed deployment

### Phase 5 — Orchestration

- [ ] Realtime audio gateway
- [ ] Concurrent fan-out
- [ ] Result aggregation
- [ ] Timeout/error policy
- [ ] Service discovery
- [ ] Distributed gRPC
- [ ] mTLS/authentication

### Phase 6 — Android Integration

- [ ] Final protobuf/report contract
- [ ] Gateway
- [ ] Android-facing result delivery
- [ ] User action/severity presentation

### Phase 7 — Cloud Deployment

- [ ] AWS container deployment
- [ ] Production service networking
- [ ] Observability stack
- [ ] Secure model storage
- [ ] Autoscaling
- [ ] Production security
- [ ] Reliability/load testing

---

## 32. Current Limitations

CloudForge is **not yet a complete production VoiceShield backend**.

Current limitations include:

- regional VMs are not real production detectors
- XLS-R lacks a trained 19-class classification head
- VM7 threshold calibration is incomplete
- RIAS does not yet exist
- TCED does not yet exist
- final Android reporting does not yet exist
- distributed orchestration is not yet complete
- production cloud deployment is not yet implemented
- production gRPC authentication/mTLS is not yet implemented

These limitations are documented rather than hidden.

---

## 33. Relationship to VoiceShield

CloudForge is one major infrastructure track inside the broader VoiceShield project.

```text
VoiceShield Android Application
             │
             ▼
          CloudForge
             │
     ┌───────┼────────┐
     │       │        │
    LID     DSR      RIAS
     │       │        │
    LACR  Detectors   │
     │       │        │
     └───────┼────────┘
             │
            TCED
             │
             ▼
      Android Report
```

CloudForge is responsible for backend/cloud inference architecture and is intended to remain synchronized with the VoiceShield Android and ML-core tracks.

---

## 34. Current Baseline

The repository currently represents the **pre-RIAS baseline**.

The authoritative evidence-based audit is:

```text
docs/cloudforge-full-implementation-audit.md
reports/cloudforge-full-implementation-audit.json
```

Before starting a new major track, review the audit and confirm the live repository state.

---

## 35. Repository

**VoiceShield CloudForge**

```text
https://github.com/yuvatejasainadh/vs-cloud-forge
```

Core purpose:

```text
Cloud-native speech intelligence,
language-aware routing,
deepfake analysis,
risk intelligence,
and threat/call decision infrastructure
for VoiceShield.
```

---

> **Status note:** This README reflects the repository baseline audited on **2026-10-08**. As CloudForge evolves, implementation status should be updated from repository evidence rather than architectural intent alone.
