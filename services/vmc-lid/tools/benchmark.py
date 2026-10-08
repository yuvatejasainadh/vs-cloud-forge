#!/usr/bin/env python3
"""
CloudForge VMC-LID Benchmarking Tool.
Measures cold start, first inference, warm inference latencies, throughput, memory, and CPU metrics.
"""

import argparse
import os
import sys
import time
import numpy as np
import psutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
GEN_DIR = os.path.join(SRC_DIR, "gen")

sys.path = [SRC_DIR, GEN_DIR, BASE_DIR] + [p for p in sys.path if p not in (SRC_DIR, GEN_DIR, BASE_DIR, os.path.dirname(__file__))]

from config.settings import Settings
from core.model_registry import ModelRegistry
from core.model_runtime import AudioInferenceInput
from core.runtime_factory import ModelRuntimeFactory
import runtimes  # Auto-registers runtimes


def run_benchmark(iterations: int = 50, audio_duration_sec: float = 2.0):
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    settings = Settings.load()
    registry = ModelRegistry(settings.model.registry_path, settings.model.models_dir)
    meta = registry.get_model(settings.model.active_model_id)

    print("================================================================================")
    print("                    CloudForge Model Runtime Benchmark                          ")
    print("================================================================================")
    print(f"Model ID:              {meta.model_id}")
    print(f"Model Name:            {meta.model_name}")
    print(f"Provider:              {meta.provider}")
    print(f"Framework:             {meta.framework}")
    print(f"Version:               {meta.version}")
    print(f"Parameters:            {meta.parameters:,} (~{meta.parameters/1e6:.1f}M)")
    print(f"Artifact Size:         {meta.artifact_size_mb:.2f} MB")
    print(f"Execution Providers:   {settings.model.execution_providers}")
    print(f"Threads:               {settings.model.num_threads}")
    print(f"Benchmark Iterations:  {iterations}")
    print(f"Audio Duration:        {audio_duration_sec}s (16 kHz mono PCM)")
    print("--------------------------------------------------------------------------------")

    # 1. Model Loading Benchmark
    t_load_start = time.perf_counter()
    runtime = ModelRuntimeFactory.create(meta, settings)
    runtime.load()
    t_load_duration = (time.perf_counter() - t_load_start) * 1000.0

    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = mem_after_load_mb - mem_before_mb

    print(f"Model Load Time:       {t_load_duration:.2f} ms ({t_load_duration/1000:.3f} s)")
    print(f"Process RAM (Total):   {mem_after_load_mb:.2f} MB")
    print(f"Model RAM Overhead:    ~{model_ram_mb:.2f} MB")
    print("--------------------------------------------------------------------------------")

    # Generate synthetic 16kHz PCM audio payload
    sample_rate = 16000
    num_samples = int(sample_rate * audio_duration_sec)
    t = np.linspace(0, audio_duration_sec, num_samples, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(num_samples)
    audio_bytes = (tone * 32767).astype(np.int16).tobytes()

    sample_input = AudioInferenceInput(
        audio_data=audio_bytes,
        sample_rate=sample_rate,
        channels=1,
        audio_format="pcm_s16le",
        request_id="bench_0"
    )

    # 2. First Inference Latency (Cold Start)
    t_first_start = time.perf_counter()
    first_result = runtime.infer(sample_input)
    t_first_duration = (time.perf_counter() - t_first_start) * 1000.0

    print(f"First Inference:       {t_first_duration:.2f} ms")
    print(f"First Prediction:      {first_result.language} (conf: {first_result.confidence:.4f})")
    print("--------------------------------------------------------------------------------")

    # 3. Warm Inference Benchmark Loop
    latencies = []
    cpu_percent_start = process.cpu_percent(interval=None)
    t_warm_start = time.perf_counter()

    for i in range(iterations):
        t0 = time.perf_counter()
        res = runtime.infer(sample_input)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    t_total_warm_sec = time.perf_counter() - t_warm_start
    cpu_percent_end = process.cpu_percent(interval=None)

    lat_arr = np.array(latencies)
    p50 = np.percentile(lat_arr, 50)
    p90 = np.percentile(lat_arr, 90)
    p95 = np.percentile(lat_arr, 95)
    p99 = np.percentile(lat_arr, 99)
    mean_lat = np.mean(lat_arr)
    min_lat = np.min(lat_arr)
    max_lat = np.max(lat_arr)
    std_lat = np.std(lat_arr)
    throughput = iterations / t_total_warm_sec

    mem_final_mb = process.memory_info().rss / (1024 * 1024)

    print("Warm Inference Latencies:")
    print(f"  - Mean:              {mean_lat:.2f} ms (+/- {std_lat:.2f} ms)")
    print(f"  - Min:               {min_lat:.2f} ms")
    print(f"  - P50 (Median):      {p50:.2f} ms")
    print(f"  - P90:               {p90:.2f} ms")
    print(f"  - P95:               {p95:.2f} ms")
    print(f"  - P99:               {p99:.2f} ms")
    print(f"  - Max:               {max_lat:.2f} ms")
    print(f"Throughput:            {throughput:.2f} inferences/sec (QPS)")
    print(f"Total Test Time:       {t_total_warm_sec:.2f} s")
    print(f"CPU Utilization:       ~{cpu_percent_end:.1f}%")
    print(f"Final RAM (RSS):       {mem_final_mb:.2f} MB")
    print("================================================================================")

    return {
        "model_id": meta.model_id,
        "parameters": meta.parameters,
        "artifact_size_mb": meta.artifact_size_mb,
        "load_time_ms": t_load_duration,
        "first_inference_ms": t_first_duration,
        "mean_latency_ms": mean_lat,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "throughput_qps": throughput,
        "ram_rss_mb": mem_final_mb,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudForge Model Runtime Benchmark")
    parser.add_argument("-n", "--iterations", type=int, default=50, help="Number of warm benchmark iterations")
    parser.add_argument("-d", "--duration", type=float, default=2.0, help="Audio duration in seconds")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, audio_duration_sec=args.duration)
