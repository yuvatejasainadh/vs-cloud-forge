import os
import sys
import time
import json
import psutil
from typing import Dict, List, Any
import numpy as np

# Add vmc-lid src to python path
sys.path.insert(0, r"C:\CloudForge\services\vmc-lid\src")

from config.settings import Settings
from core.model_registry import ModelRegistry
from core.runtime_factory import ModelRuntimeFactory
from core.model_runtime import AudioInferenceInput
from routing.canonical_registry import get_canonical_registry
import runtimes


def generate_synthetic_audio(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generates synthetic multi-frequency speech-like PCM s16le bytes."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Fundamental and formants
    signal = (
        0.5 * np.sin(2 * np.pi * 220 * t) +
        0.25 * np.sin(2 * np.pi * 440 * t) +
        0.15 * np.sin(2 * np.pi * 880 * t) +
        0.05 * np.random.normal(0, 0.05, len(t))
    )
    signal = np.clip(signal, -1.0, 1.0)
    int16_audio = (signal * 32767).astype(np.int16)
    return int16_audio.tobytes()


def run_benchmark(
    iterations: int = 50,
    warmup: int = 5,
    model_id: str = "open-vakgyata"
) -> Dict[str, Any]:
    print(f"=== Starting LID Benchmark for model '{model_id}' ===")
    
    settings = Settings.load()
    settings.model.active_model_id = model_id
    
    registry = ModelRegistry(
        registry_path=settings.model.registry_path,
        models_base_dir=settings.model.models_dir
    )
    meta = registry.get_model(model_id)
    runtime = ModelRuntimeFactory.create(meta, settings)
    
    # Measure load time and memory
    proc = psutil.Process(os.getpid())
    mem_before_mb = proc.memory_info().rss / (1024 * 1024)
    
    load_start = time.perf_counter()
    runtime.load()
    load_time_sec = time.perf_counter() - load_start
    
    mem_after_mb = proc.memory_info().rss / (1024 * 1024)
    model_rss_mb = mem_after_mb - mem_before_mb
    
    print(f"Model loaded in {load_time_sec:.3f}s (Memory delta: {model_rss_mb:.2f} MB, Total RSS: {mem_after_mb:.2f} MB)")
    
    canonical_reg = get_canonical_registry()
    test_audio = generate_synthetic_audio(duration_sec=3.0, sample_rate=16000)
    
    inp = AudioInferenceInput(
        audio_data=test_audio,
        sample_rate=16000,
        channels=1,
        audio_format="pcm_s16le",
        request_id="bench_warmup"
    )
    
    # Warmup runs
    for w in range(warmup):
        runtime.infer(inp)
        
    latencies_ms: List[float] = []
    engine_latencies_ms: List[float] = []
    
    print(f"Executing {iterations} inference iterations...")
    total_start = time.perf_counter()
    
    for i in range(iterations):
        inp.request_id = f"bench_req_{i}"
        t0 = time.perf_counter()
        res = runtime.infer(inp)
        t1 = time.perf_counter()
        
        latencies_ms.append((t1 - t0) * 1000.0)
        engine_latencies_ms.append(res.inference_duration_ms)
        
    total_elapsed = time.perf_counter() - total_start
    throughput = iterations / total_elapsed
    
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    
    results = {
        "model_id": model_id,
        "model_version": meta.version,
        "parameters": meta.parameters,
        "iterations": iterations,
        "total_elapsed_sec": round(total_elapsed, 3),
        "throughput_inf_per_sec": round(throughput, 2),
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "mean": round(mean_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2)
        },
        "engine_latency_ms": {
            "mean": round(float(np.mean(engine_latencies_ms)), 2),
            "p95": round(float(np.percentile(engine_latencies_ms, 95)), 2)
        },
        "memory_mb": {
            "model_rss_delta": round(model_rss_mb, 2),
            "total_rss": round(mem_after_mb, 2)
        },
        "supported_languages_count": len(meta.supported_languages),
        "canonical_coverage_pct": round(len(meta.supported_languages) / canonical_reg.EXPECTED_CANONICAL_COUNT * 100.0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    out_dir = r"C:\CloudForge\benchmarks\lid\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"benchmark_{model_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {out_path}")
    print(f"P50: {p50:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms | Throughput: {throughput:.2f} inf/sec")
    return results


if __name__ == "__main__":
    run_benchmark(iterations=30, warmup=3)
