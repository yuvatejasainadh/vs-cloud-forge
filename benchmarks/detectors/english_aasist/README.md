# VM7 English AASIST-L Detector Benchmark

## Overview
This benchmark suite measures inference performance, latency distributions ($P_{50}, P_{95}, P_{99}$), memory overhead, and throughput for the VoiceShield VM7 English Deepfake Detection Service (`vmc-english`).

## Execution
```bash
python benchmarks/detectors/english_aasist/runner.py
```

## Model Status Policy
If the Azure endpoint credentials (`AZURE_AASIST_API_KEY` / `AZURE_AASIST_ENDPOINT`) or local model checkpoint are unprovisioned, the benchmark explicitly records `MODEL_ARTIFACT_STATUS = BLOCKED / NOT AVAILABLE` without generating false metrics.
