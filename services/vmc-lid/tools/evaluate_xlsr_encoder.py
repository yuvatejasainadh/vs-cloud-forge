import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import soundfile as sf
import psutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))

for path in [os.path.join(SERVICE_DIR, "src"), SERVICE_DIR, REPO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from config.settings import Settings
from core.model_registry import ModelRegistry, ModelRole, ModelStatus
from core.artifact_resolver import ArtifactResolver
from runtimes.xlsr_runtime import XLSRRuntime
from core.audio_processor import AudioProcessor


def evaluate_xlsr_encoder_on_dataset(
    dataset_path: str,
    limit: int = 20,
    output_path: Optional[str] = None
):
    print("=" * 80)
    print(" CloudForge V2 — XLS-R 300M Encoder-Only Dataset Experiment")
    print(" [!] BENCHMARK TYPE: ENCODER ONLY (Raw Hidden Representations)")
    print(" [!] Note: Base XLS-R 300M is not a 19-class language classifier.")
    print("=" * 80)

    reg_path = os.path.join(REPO_ROOT, "models", "registry.json")
    models_dir = os.path.join(REPO_ROOT, "models")
    registry = ModelRegistry(reg_path, models_dir)
    meta = registry.get_model("facebook-wav2vec2-xls-r-300m")

    settings = Settings()
    resolver = ArtifactResolver()
    runtime = XLSRRuntime(metadata=meta, settings=settings, resolver=resolver)

    print(f"\n[*] Model: {meta.model_id} ({meta.parameters:,} params, Role: {meta.role})")
    runtime.load()
    health = runtime.health()
    print(f"[*] Runtime Status: {health['status']} (Head Available: {health['classification_head_available']})")

    # Discover files
    files = []
    for root, _, f_list in os.walk(dataset_path):
        for f in f_list:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))
                if limit and limit > 0 and len(files) >= limit:
                    break
        if limit and limit > 0 and len(files) >= limit:
            break

    print(f"[*] Evaluating {len(files)} files with XLS-R 300M encoder...")

    results = []
    latencies = []
    process = psutil.Process(os.getpid())

    for idx, fpath in enumerate(files):
        t0 = time.perf_counter()
        try:
            waveform, sr = sf.read(fpath)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Resample / process
            proc_waveform = AudioProcessor.process_raw_audio(
                audio_bytes=(np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes(),
                sample_rate=sr,
                channels=1,
                audio_format="pcm_s16le",
                target_sample_rate=16000,
                do_normalize=True
            )

            if runtime.is_loaded:
                emb = runtime.encode(proc_waveform)
                emb_shape = list(emb.shape)
                emb_mean = float(np.mean(emb))
                emb_std = float(np.std(emb))
                status = "ENCODED"
            else:
                emb_shape = None
                emb_mean = None
                emb_std = None
                status = "BACKBONE_UNHYDRATED"

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            results.append({
                "file": os.path.basename(fpath),
                "duration_seconds": len(proc_waveform) / 16000.0,
                "status": status,
                "embedding_shape": emb_shape,
                "embedding_mean": emb_mean,
                "embedding_std": emb_std,
                "encoder_latency_ms": lat_ms
            })
            print(f"  [{idx+1:02d}/{len(files):02d}] {os.path.basename(fpath)} -> {status} ({lat_ms:.2f} ms)")
        except Exception as e:
            results.append({"file": os.path.basename(fpath), "status": "ERROR", "error": str(e)})

    p50 = np.percentile(latencies, 50) if latencies else 0.0
    p95 = np.percentile(latencies, 95) if latencies else 0.0
    rss_mb = process.memory_info().rss / (1024 * 1024)

    summary = {
        "benchmark_type": "ENCODER_ONLY",
        "model_id": meta.model_id,
        "samples_evaluated": len(results),
        "p50_latency_ms": float(p50),
        "p95_latency_ms": float(p95),
        "process_rss_mb": float(rss_mb),
        "classification_head": "NOT_AVAILABLE"
    }

    print("\n" + "=" * 80)
    print(f" XLS-R 300M Encoder Evaluation Summary: P50 Latency = {p50:.2f} ms | RSS = {rss_mb:.1f} MB")
    print("=" * 80)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump({"summary": summary, "results": results}, out_f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLS-R 300M Dataset Encoder Evaluation")
    parser.add_argument("--dataset", type=str, default=r"C:\Users\yuvat\Downloads\sources_wavefake\_sources_wavefake", help="Path to dataset")
    parser.add_argument("--limit", type=int, default=0, help="Number of files to evaluate (0 for full dataset)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_xlsr_encoder_on_dataset(args.dataset, limit=args.limit, output_path=args.output)
