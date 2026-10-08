import os
import sys
import time
import json
import psutil
from typing import Dict, Any, List
import numpy as np

# Add vmc-english src to path
sys.path.insert(0, r"C:\CloudForge\services\vmc-english\src")
sys.path.insert(0, r"C:\CloudForge\services\vmc-english\src\gen")

from config.settings import Settings
from core.runtime_factory import AASISTLRuntimeFactory
from core.detector import EnglishDeepfakeDetector
from core.runtime import ModelUnavailableError


def run_benchmark(iterations: int = 30) -> Dict[str, Any]:
    print("================================================================================")
    print("      CloudForge Native VM7 English AASIST-L Deepfake Detector Benchmark        ")
    print("================================================================================")

    settings = Settings.load()
    runtime = AASISTLRuntimeFactory.create(settings)
    runtime.load()

    health_info = runtime.health()
    print(f"Platform:              CloudForge Native")
    print(f"Model ID:              {runtime.model_id}")
    print(f"Version:               {runtime.version}")
    print(f"Execution Provider:    {runtime.execution_provider}")
    print(f"Runtime Status:        {health_info.get('status')}")
    print(f"Ready:                 {runtime.is_ready()}")
    print("--------------------------------------------------------------------------------")

    if not runtime.is_ready():
        print("[!] NOTICE: AASIST-L model artifact is NOT HYDRATED in model-cache.")
        print("    Status: MODEL_ARTIFACT_STATUS = BLOCKED / AWAITING ARTIFACT")
        print("    No synthetic/fabricated benchmark numbers will be generated.")
        
        result = {
            "model_id": runtime.model_id,
            "version": runtime.version,
            "vm": "VM7_ENGLISH",
            "group": "ENGLISH",
            "language": "en-IN",
            "provider": "cloudforge",
            "status": "BLOCKED / AWAITING ARTIFACT",
            "reason": "Model artifact not found in model-cache directory",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        out_dir = r"C:\CloudForge\benchmarks\detectors\english_aasist\results"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark_english_aasist.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    detector = EnglishDeepfakeDetector(settings, runtime)
    
    # Generate test 16kHz PCM audio
    sr = 16000
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_bytes = (tone * 32767).astype(np.int16).tobytes()

    print(f"Executing {iterations} inference iterations on verified AASIST-L model...")
    latencies_ms: List[float] = []
    total_start = time.perf_counter()

    for i in range(iterations):
        t0 = time.perf_counter()
        res = detector.analyze(
            audio_content=audio_bytes,
            sample_rate=16000,
            language="en-IN",
            request_id=f"bench_{i}"
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_elapsed = time.perf_counter() - total_start
    throughput = iterations / total_elapsed

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)

    result = {
        "model_id": runtime.model_id,
        "version": runtime.version,
        "vm": "VM7_ENGLISH",
        "group": "ENGLISH",
        "language": "en-IN",
        "provider": "cloudforge",
        "status": "EVALUATED",
        "iterations": iterations,
        "throughput_inf_per_sec": round(throughput, 2),
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "mean": round(mean_lat, 2)
        },
        "memory_rss_mb": round(rss_mb, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    out_dir = r"C:\CloudForge\benchmarks\detectors\english_aasist\results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark_english_aasist.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"P50: {p50:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms | Throughput: {throughput:.2f} inf/sec")
    return result


if __name__ == "__main__":
    run_benchmark(iterations=10)
