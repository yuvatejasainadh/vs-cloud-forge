import os
import sys
import time
import argparse
import json
import numpy as np
import psutil

# Add src and service directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

for p in [SRC_DIR, BASE_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import Settings
from core.model_registry import ModelRegistry, ModelRole, ModelStatus
from core.artifact_resolver import ArtifactResolver
from core.audio_processor import AudioProcessor
from core.model_runtime import AudioInferenceInput
from runtimes.open_vakgyata_runtime import OpenVakgyataRuntime
from runtimes.xlsr_runtime import XLSRRuntime, MeanPooling


def compare_models(num_iterations: int = 10, duration_sec: float = 3.0):
    print("=" * 90)
    print(" CloudForge V2 — LID Model Architecture & Performance Comparison")
    print("=" * 90)

    reg_path = os.path.join(REPO_ROOT, "models", "registry.json")
    models_dir = os.path.join(REPO_ROOT, "models")
    registry = ModelRegistry(reg_path, models_dir)
    settings = Settings()

    # Generate 3-second 16kHz test audio
    num_samples = int(16000 * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    synthetic_waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    synthetic_pcm_bytes = (synthetic_waveform * 32767.0).astype(np.int16).tobytes()
    audio_inp = AudioInferenceInput(
        audio_data=synthetic_pcm_bytes,
        sample_rate=16000,
        channels=1,
        audio_format="pcm_s16le"
    )

    process = psutil.Process(os.getpid())

    models_to_evaluate = [
        {"id": "open-vakgyata", "name": "Open-Vakgyata (10-Class Benchmark)"},
        {"id": "facebook-wav2vec2-xls-r-300m", "name": "Facebook XLS-R 300M (Speech Backbone)"},
    ]

    comparison_results = []

    for item in models_to_evaluate:
        m_id = item["id"]
        meta = registry.get_model(m_id)
        print(f"\nEvaluating: {meta.model_name} [{m_id}]")
        print(f"  - Role:           {meta.role}")
        print(f"  - Task:           {meta.task}")
        print(f"  - Status:         {meta.status}")
        print(f"  - Framework:      {meta.framework}")
        print(f"  - Parameters:     {meta.parameters:,}")
        print(f"  - Pinned Rev:     {meta.revision or 'N/A'}")
        print(f"  - Languages:      {len(meta.supported_languages)} classes: {meta.supported_languages}")

        # Measure Load Time and Memory
        rss_before = process.memory_info().rss / (1024 * 1024)
        t_load_0 = time.perf_counter()

        runtime = None
        is_ready = False

        if m_id == "open-vakgyata":
            runtime = OpenVakgyataRuntime(metadata=meta, settings=settings)
            try:
                runtime.load()
                is_ready = runtime.is_loaded
            except Exception as e:
                print(f"  [!] Open-Vakgyata load notice: {e}")
        elif "xls-r" in m_id:
            runtime = XLSRRuntime(metadata=meta, settings=settings)
            try:
                runtime.load()
                is_ready = runtime.is_loaded
            except Exception as e:
                print(f"  [!] XLS-R load notice: {e}")

        load_time_s = time.perf_counter() - t_load_0
        rss_after = process.memory_info().rss / (1024 * 1024)
        delta_rss = max(0.0, rss_after - rss_before)

        print(f"  - Load Duration:  {load_time_s:.3f} s")
        print(f"  - Loaded Status:  {'READY' if is_ready else 'AWAITING_ARTIFACT'}")
        print(f"  - Memory Delta:   +{delta_rss:.1f} MB (Process RSS: {rss_after:.1f} MB)")

        # Inference Latency Benchmark
        latencies = []
        throughput_samples_per_sec = 0.0

        if is_ready and runtime is not None:
            # Warmup
            for _ in range(3):
                if m_id == "open-vakgyata":
                    _ = runtime.infer(audio_inp)
                else:
                    _ = runtime.encode(synthetic_waveform)

            # Timed runs
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                if m_id == "open-vakgyata":
                    _ = runtime.infer(audio_inp)
                else:
                    _ = runtime.encode(synthetic_waveform)
                latencies.append((time.perf_counter() - t0) * 1000.0)

            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            mean_lat = np.mean(latencies)
            throughput_samples_per_sec = 1000.0 / mean_lat if mean_lat > 0 else 0.0

            print(f"  - Latency (P50):  {p50:.2f} ms")
            print(f"  - Latency (P95):  {p95:.2f} ms")
            print(f"  - Throughput:     {throughput_samples_per_sec:.2f} req/s (Audio: {duration_sec}s)")
        else:
            p50 = 0.0
            p95 = 0.0
            p99 = 0.0
            mean_lat = 0.0
            print(f"  - Latency:        N/A (Backbone awaiting artifact)")

        # Unload
        if runtime is not None:
            runtime.unload()

        comparison_results.append({
            "model_id": meta.model_id,
            "model_name": meta.model_name,
            "role": meta.role,
            "task": meta.task,
            "status": meta.status,
            "architecture": meta.architecture,
            "parameters": meta.parameters,
            "framework": meta.framework,
            "revision": meta.revision,
            "supported_languages_count": len(meta.supported_languages),
            "supported_languages": meta.supported_languages,
            "load_time_s": round(load_time_s, 4),
            "memory_delta_mb": round(delta_rss, 1),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "throughput_req_per_s": round(throughput_samples_per_sec, 2),
            "classification_head": "TRAINED (10-Class)" if m_id == "open-vakgyata" else "NOT_AVAILABLE (Backbone Only)"
        })

    # Summary table
    print("\n" + "=" * 90)
    print(" COMPARISON SUMMARY MATRIX")
    print("=" * 90)
    header = f"{'Model ID':<30} | {'Role':<12} | {'Params':<10} | {'Lang Count':<10} | {'Load(s)':<8} | {'P50(ms)':<8} | {'Head Status':<20}"
    print(header)
    print("-" * 90)
    for r in comparison_results:
        row = f"{r['model_id']:<30} | {r['role']:<12} | {r['parameters']:<10,} | {r['supported_languages_count']:<10} | {r['load_time_s']:<8.2f} | {r['p50_latency_ms']:<8.2f} | {r['classification_head']:<20}"
        print(row)
    print("=" * 90)

    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudForge LID Models Comparison Tool")
    parser.add_argument("-n", "--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("-d", "--duration", type=float, default=3.0, help="Synthetic audio duration in seconds")
    args = parser.parse_args()

    compare_models(num_iterations=args.iterations, duration_sec=args.duration)
