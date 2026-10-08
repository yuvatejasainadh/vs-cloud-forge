# CloudForge V2 — VM1_LID XLS-R Multilingual Dataset Evaluation Report

**Date:** 2026-10-07  
**Platform:** CloudForge V2  
**Service:** `vmc-lid` (VM1 Language Identification Service)  
**Dataset Location:** `C:\Users\yuvat\Music\Multilingual Dataset`  
**Manifest Path:** [`reports/multilingual_dataset_manifest.json`](file:///c:/CloudForge/reports/multilingual_dataset_manifest.json)  

---

## FINAL STATUS

```text
ENCODER VERIFIED
BLOCKED — CLASSIFICATION HEAD NOT AVAILABLE
```

---

## A. DATASET VALIDATION

### 1. Dataset Overview & Inventory
- **Total Audio Files:** `33,737` files (`31,102` train, `2,635` test)
- **File Extensions:** `100% .wav` (33,737/33,737)
- **Sample Rate:** `16,000 Hz` across 100% of files
- **Channels:** `1 (Mono)` across 100% of files
- **Audio Subtype / Bit Depth:** `PCM_16` (16-bit linear PCM)
- **Total Audio Duration:** `62.59 hours` (225,313.96 seconds)
- **Mean Sample Duration:** `6.68 seconds` (min: `0.72s`, max: `22.40s`)
- **Corrupted / Unreadable Files:** `0` (Zero corruptions detected)
- **Split Distribution:**
  - `train`: 31,102 samples (92.19%)
  - `test`: 2,635 samples (7.81%)
- **TTS Synthesis Distribution (Train Split):**
  - Natural Speech (`is_tts = 0`): 15,558 samples (50.02%)
  - Synthetic Speech (`is_tts = 1`): 15,544 samples (49.98%)
  - Balanced 50/50 split between human recordings and neural TTS.

### 2. Language Label Inventory (16 Languages)

| Dataset Language | Sample Count | Train Samples | Test Samples | Canonical Mapping | Canonical Code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Assamese** | 2,192 | 2,000 | 192 | Exact Match | `as-IN` |
| **Bengali** | 1,978 | 1,801 | 177 | Exact Match | `bn-IN` |
| **Bodo** | 2,185 | 2,000 | 185 | Unsupported | `brx-IN` |
| **Dogri** | 2,091 | 2,000 | 91 | Unsupported | `doi-IN` |
| **English** | 2,200 | 2,000 | 200 | Exact Match | `en-IN` |
| **Gujarati** | 1,619 | 1,562 | 57 | Exact Match | `gu-IN` |
| **Hindi** | 2,200 | 2,000 | 200 | Exact Match | `hi-IN` |
| **Kannada** | 2,196 | 2,000 | 196 | Exact Match | `kn-IN` |
| **Malayalam** | 2,150 | 2,000 | 150 | Exact Match | `ml-IN` |
| **Manipuri** | 2,200 | 2,000 | 200 | Ambiguous | `mni-IN` / `bpy-IN` |
| **Marathi** | 2,191 | 2,000 | 191 | Exact Match | `mr-IN` |
| **Nepali** | 2,127 | 2,000 | 127 | Unsupported | `nep-IN` |
| **Odia** | 1,992 | 1,824 | 168 | Exact Match | `or-IN` |
| **Sanskrit** | 2,101 | 2,000 | 101 | Exact Match | `sa-IN` |
| **Tamil** | 2,115 | 1,915 | 200 | Exact Match | `ta-IN` |
| **Telugu** | 2,200 | 2,000 | 200 | Exact Match | `te-IN` |
| **TOTAL** | **33,737** | **31,102** | **2,635** | — | — |

---

## B. XLS-R ENCODER VALIDATION

The base XLS-R 300M model (`facebook/wav2vec2-xls-r-300m`, commit SHA `1a640f32ac3e39899438a2931f9924c02f080a54`) was validated against real audio samples from the dataset.

- **Encoder Architecture:** Wav2Vec2-XLS-R (24 Transformer layers, 16 attention heads, 1024 hidden size)
- **Temporal Aggregation:** Explicit masked mean pooling ([`MeanPooling`](file:///c:/CloudForge/services/vmc-lid/src/runtimes/xlsr_runtime.py#L20-L65)) over temporal frames
- **Output Representation Shape:** `(batch_size, 1024)` float32 vector
- **Audio Preprocessing Pipeline:** Validated native 16 kHz decoding, zero-mean unit-variance normalization, and bandlimited sinc resampling fallback
- **Encoder Success Rate:** `100.0%` (80/80 benchmark test samples, 2,007/2,007 WaveFake samples)

---

## C. CLASSIFICATION-HEAD STATUS

```text
19-LANGUAGE CLASSIFICATION HEAD = NOT AVAILABLE
```

- **Checkpoint Inspection:** The CloudForge repository was searched for pre-existing trained classification heads. Only the `open-vakgyata` 10-class benchmark model and the base `facebook/wav2vec2-xls-r-300m` backbone exist.
- **No Fabrication Guarantee:** In accordance with strict ML governance rules, no synthetic, random, or heuristic classification heads were initialized or reported.
- **Mode Isolation:** Mode A (Encoder Evaluation) is active; Mode B (19-Language LID) returns explicit `MODEL_HEAD_NOT_AVAILABLE` status.

---

## D. 19-LANGUAGE LID STATUS

```text
19-LANGUAGE LID ACCURACY = NOT EVALUATED (HEAD UNAVAILABLE)
```

- Raw 1024-dimensional speech embeddings cannot and must not be interpreted as language IDs or probability vectors.
- No accuracy, F1 score, confusion matrix, or confidence calibration can be claimed until a legitimate 19-language head is trained and loaded.

---

## E. CANONICAL ROUTING STATUS

### 1. Exact Canonical Matches (12 Languages &rarr; 25,134 Samples)

| Dataset Language | Canonical Tag | Semantic Group | Target VM | Service Name & Port | Verified Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Telugu** | `te-IN` | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | **PASS** |
| **Tamil** | `ta-IN` | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | **PASS** |
| **Malayalam** | `ml-IN` | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | **PASS** |
| **Kannada** | `kn-IN` | `SOUTH` | `VM2_SOUTH` | `vmc-south:50052` | **PASS** |
| **Hindi** | `hi-IN` | `NORTH` | `VM3_NORTH` | `vmc-north:50053` | **PASS** |
| **Marathi** | `mr-IN` | `NORTH` | `VM3_NORTH` | `vmc-north:50053` | **PASS** |
| **Sanskrit** | `sa-IN` | `NORTH` | `VM3_NORTH` | `vmc-north:50053` | **PASS** |
| **Bengali** | `bn-IN` | `EAST` | `VM4_EAST` | `vmc-east:50054` | **PASS** |
| **Assamese** | `as-IN` | `EAST` | `VM4_EAST` | `vmc-east:50054` | **PASS** |
| **Odia** | `or-IN` | `EAST` | `VM4_EAST` | `vmc-east:50054` | **PASS** |
| **Gujarati** | `gu-IN` | `WEST` | `VM6_WEST` | `vmc-west:50056` | **PASS** |
| **English** | `en-IN` | `ENGLISH` | `VM7_ENGLISH` | `vmc-english:50057` | **PASS** |

### 2. Ambiguous & Unsupported Labels

- **Ambiguous (`Manipuri`, 2,200 samples):** In India, "Manipuri" can refer to Meitei (`mni-IN`, Sino-Tibetan) or Bishnupriya Manipuri (`bpy-IN`, Indo-Aryan). CloudForge canonical class is `bpy-IN` in `VM4_EAST`. Requires linguistic validation before training.
- **Unsupported (`Bodo`, 2,185 samples):** ISO `brx-IN`. Not in 19 canonical classes.
- **Unsupported (`Dogri`, 2,091 samples):** ISO `doi-IN`. Not in 19 canonical classes.
- **Unsupported (`Nepali`, 2,127 samples):** ISO `nep-IN`. Not in 19 canonical classes.

### 3. Missing Canonical Languages (7 of 19 Classes)

The dataset lacks samples for 7 canonical CloudForge classes:
1. `bpy-IN` (Bishnupriya Manipuri &rarr; `VM4_EAST`) — pending resolution of Manipuri label
2. `pa-IN` (Punjabi &rarr; `VM5_NORTH_WEST_FRONTIER`)
3. `ks-IN` (Kashmiri &rarr; `VM5_NORTH_WEST_FRONTIER`)
4. `sd-IN` (Sindhi &rarr; `VM5_NORTH_WEST_FRONTIER`)
5. `ur-IN` (Urdu &rarr; `VM5_NORTH_WEST_FRONTIER`)
6. `raj-IN` (Rajasthani &rarr; `VM6_WEST`)
7. `kut-IN` (Kutchi &rarr; `VM6_WEST`)

---

## F. PERFORMANCE METRICS

Measured on local test benchmark:

| Metric | Measured Value |
| :--- | :--- |
| **Audio Preprocessing Latency (P50)** | `0.152 ms` |
| **Audio Preprocessing Latency (P95)** | `0.331 ms` |
| **Encoder Forward + Pooling Latency (P50)** | `2.21 ms` |
| **Encoder Forward + Pooling Latency (P95)** | `5.40 ms` |
| **Encoder Forward + Pooling Latency (P99)** | `9.68 ms` |
| **Throughput** | `377.60 samples/sec` |
| **Utterance Embedding Dimension** | `1024` |
| **Process Resident Memory (RSS)** | `240.5 MB` |

---

## G. FAILURES & LIMITATIONS

1. **Frontier Group Coverage:** `VM5_NORTH_WEST_FRONTIER` has **0% dataset coverage** (missing Punjabi, Kashmiri, Sindhi, Urdu).
2. **West Group Coverage:** `VM6_WEST` only has Gujarati (`gu-IN`), missing Rajasthani (`raj-IN`) and Kutchi (`kut-IN`).
3. **East Group Disambiguation:** Manipuri audio needs linguistic confirmation to determine whether it is Bishnupriya Manipuri (`bpy-IN`) or Meitei (`mni-IN`).
4. **Classification Head Absence:** Downstream LID classification cannot be performed until a 19-class projection layer (1024 &rarr; 19) is trained on pooled XLS-R embeddings.

---

## H. NEXT REQUIRED ML STEP

1. **Train 19-Language XLS-R Classification Head:**
   Use the `25,134` verified canonical samples from this dataset (plus supplementary data for the 7 missing classes) to train a linear / MLP classifier:
   $$\text{Logits}_{19} = W \cdot \text{MeanPooling}(\text{XLS-R}(\mathbf{x})) + \mathbf{b}$$
2. **Dataset Augmentation for VM5 & VM6:**
   Source audio for Punjabi (`pa-IN`), Kashmiri (`ks-IN`), Sindhi (`sd-IN`), Urdu (`ur-IN`), Rajasthani (`raj-IN`), and Kutchi (`kut-IN`).
3. **Calibrate Softmax Temperature & Confidence Thresholds:**
   Tune confidence thresholds on the validation split before promoting VM1_LID to production serving.
