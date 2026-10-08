import os
import sys
import time
import argparse
import numpy as np
import psutil

# Add src and service directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.settings import Settings
from core.model_registry import ModelRegistry, ModelRole, ModelStatus
from core.artifact_resolver import ArtifactResolver, HuggingFaceArtifactSource, LocalFilesystemSource
from core.audio_processor import AudioProcessor
from core.classification_head import (
    NoClassificationHead,
    LinearClassificationHead,
    ClassificationHeadNotAvailableError,
)
from runtimes.xlsr_runtime import XLSRRuntime
from telemetry.logging import get_logger

logger = get_logger("benchmark_xlsr")


def run_xlsr_benchmark(num_iterations: int = 20, warmup_runs: int = 5):
    print("=" * 80)
    print(" CloudForge V2 — XLS-R 300M (facebook/wav2vec2-xls-r-300m) Benchmark")
    print("=" * 80)

    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    registry_path = os.path.join(BASE_DIR, "..", "..", "models", "registry.json")
    models_dir = os.path.join(BASE_DIR, "..", "..", "models")
    
    registry = ModelRegistry(registry_path=registry_path, models_base_dir=models_dir)
    xlsr_meta = registry.get_model("facebook-wav2vec2-xls-r-300m")

    print(f"\n[1] Model Definition:")
    print(f"    - Model ID:       {xlsr_meta.model_id}")
    print(f"    - Architecture:   {xlsr_meta.architecture}")
    print(f"    - Role:           {xlsr_meta.role}")
    print(f"    - Status:         {xlsr_meta.status}")
    print(f"    - Parameters:     {xlsr_meta.parameters:,} (300M)")
    print(f"    - Repository:     {xlsr_meta.repository}")
    print(f"    - Pinned Rev:     {xlsr_meta.revision}")

    # Check Artifact Hydration Status
    resolver = ArtifactResolver(sources=[
        LocalFilesystemSource(),
        HuggingFaceArtifactSource(
            repository_id=xlsr_meta.repository or "facebook/wav2vec2-xls-r-300m",
            revision=xlsr_meta.revision or "1a640f32ac3e39899438a2931f9924c02f080a54"
        )
    ])

    settings = Settings()
    runtime = XLSRRuntime(metadata=xlsr_meta, settings=settings, resolver=resolver)

    print(f"\n[2] Artifact Resolution & Load:")
    load_start = time.perf_counter()
    runtime.load()
    load_duration_s = time.perf_counter() - load_start

    health_info = runtime.health()
    print(f"    - Load Duration:  {load_duration_s:.3f} s")
    print(f"    - Health Status:  {health_info['status']}")
    print(f"    - Artifact Path:  {health_info['artifact_path'] or 'NOT_HYDRATED'}")
    print(f"    - Session Ready:  {health_info['session_initialized']}")
    print(f"    - Providers:      {health_info['execution_providers']}")
    print(f"    - Head Available: {health_info['classification_head_available']}")

    rss_after_load_mb = process.memory_info().rss / (1024 * 1024)
    print(f"    - Process RSS:    {rss_after_load_mb:.1f} MB (Delta: +{rss_after_load_mb - rss_before_mb:.1f} MB)")

    # Generate synthetic 3-second 16kHz speech waveform
    sample_rate = 16000
    duration_sec = 3.0
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    synthetic_waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    synthetic_pcm_bytes = (synthetic_waveform * 32767.0).astype(np.int16).tobytes()

    print(f"\n[3] Audio Preprocessing Benchmark:")
    prep_times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = AudioProcessor.process_raw_audio(
            audio_bytes=synthetic_pcm_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            target_sample_rate=16000,
            do_normalize=True
        )
        prep_times.append((time.perf_counter() - t0) * 1000.0)

    p50_prep = np.percentile(prep_times, 50)
    p95_prep = np.percentile(prep_times, 95)
    p99_prep = np.percentile(prep_times, 99)
    print(f"    - Samples:        {num_samples} (3.0s @ 16kHz)")
    print(f"    - P50 Preproc:    {p50_prep:.3f} ms")
    print(f"    - P95 Preproc:    {p95_prep:.3f} ms")
    print(f"    - P99 Preproc:    {p99_prep:.3f} ms")

    print(f"\n[4] Encoder Representation Inference Benchmark:")
    if not runtime.is_loaded:
        print("    [!] Backbone ONNX artifact is not loaded/hydrated in local environment.")
        print("    [!] BENCHMARK STATUS: AWAITING_ARTIFACT (Encoder execution skipped)")
        print("    [!] 19-Language Classification Head: NOT AVAILABLE")
        print("    [!] Complete 19-Language LID: NOT VERIFIED (Backbone only)")
        print("=" * 80)
        return

    # Warmup runs
    for _ in range(warmup_runs):
        _ = runtime.encode(synthetic_waveform)

    # Timed encoder inference
    encode_times = []
    cpu_percentages = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = runtime.encode(synthetic_waveform)
        encode_times.append((time.perf_counter() - t0) * 1000.0)
        cpu_percentages.append(psutil.cpu_percent(interval=None))

    p50_enc = np.percentile(encode_times, 50)
    p95_enc = np.percentile(encode_times, 95)
    p99_enc = np.percentile(encode_times, 99)
    avg_cpu = np.mean(cpu_percentages)
    rss_end_mb = process.memory_info().rss / (1024 * 1024)

    print(f"    - Benchmark Type: ENCODER ONLY (Raw Hidden State Representations)")
    print(f"    - Iterations:     {num_iterations} (+ {warmup_runs} warmup)")
    print(f"    - P50 Latency:    {p50_enc:.2f} ms")
    print(f"    - P95 Latency:    {p95_enc:.2f} ms")
    print(f"    - P99 Latency:    {p99_enc:.2f} ms")
    print(f"    - Mean CPU Util:  {avg_cpu:.1f}%")
    print(f"    - Final RSS:      {rss_end_mb:.1f} MB")

    print(f"\n[5] Classification Head & 19-Language Readiness Status:")
    print(f"    - Classification Head:        NOT AVAILABLE (No trained checkpoint)")
    print(f"    - Output Language Labels:     19 Canonical Labels Ready ({len(health_info['supported_languages'] if 'supported_languages' in health_info else [])} mapped)")
    print(f"    - Real 19-Language Accuracy:  NOT VALIDATED (Base backbone only)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudForge XLS-R 300M Benchmark Runner")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()

    run_xlsr_benchmark(num_iterations=args.iterations, warmup_runs=args.warmup)
