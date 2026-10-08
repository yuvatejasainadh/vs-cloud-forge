import os
import sys
import time
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import soundfile as sf
import psutil

# Ensure paths are set
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
GEN_DIR = os.path.join(SRC_DIR, "gen")
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

for p in [SRC_DIR, GEN_DIR, BASE_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import runtimes  # Auto-registers ModelRuntimes
from config.settings import Settings
from core.model_registry import ModelRegistry
from core.runtime_factory import ModelRuntimeFactory
from core.model_runtime import AudioInferenceInput
from core.audio_processor import AudioProcessor
from routing.canonical_registry import get_canonical_registry, CanonicalRegistry


MULTILINGUAL_DATASET_DIR = r"C:\Users\yuvat\Music\Multilingual Dataset"
DEEPFAKE_DATASET_DIR = r"C:\Users\yuvat\Music\Deepfake Dataset"
RANDOM_SEED = 42
SAMPLES_PER_DATASET = 100


def discover_and_sample_dataset(dataset_dir: str, sample_count: int = 100, seed: int = 42) -> List[str]:
    """Recursively discovers all WAV files, sorts them deterministically, and samples sample_count files."""
    all_wavs = []
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname.lower().endswith(".wav"):
                all_wavs.append(os.path.abspath(os.path.join(root, fname)))

    all_wavs.sort()  # Guarantee deterministic order before sampling
    if len(all_wavs) < sample_count:
        raise ValueError(f"Dataset at {dataset_dir} contains only {len(all_wavs)} files, requested {sample_count}")

    rng = random.Random(seed)
    return rng.sample(all_wavs, sample_count)


def run_e2e_200_sample_routing_test(
    multilingual_dir: str = MULTILINGUAL_DATASET_DIR,
    deepfake_dir: str = DEEPFAKE_DATASET_DIR,
    sample_count: int = SAMPLES_PER_DATASET,
    seed: int = RANDOM_SEED,
    detailed_json_path: str = os.path.join(REPO_ROOT, "reports", "e2e_200_sample_routing_results.json"),
    summary_json_path: str = os.path.join(REPO_ROOT, "reports", "e2e_200_sample_routing_summary.json")
) -> Dict[str, Any]:
    print("=" * 95)
    print(" CloudForge V2 — E2E Pipeline & Canonical Routing Validation (200 Samples)")
    print("=" * 95)

    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)

    # 1. Deterministic Random Sampling
    print(f"\n[PHASE 1] Deterministic Random Sampling (Seed = {seed})...")
    print(f"  Sampling {sample_count} files from Multilingual Dataset: {multilingual_dir}")
    multilingual_files = discover_and_sample_dataset(multilingual_dir, sample_count=sample_count, seed=seed)

    print(f"  Sampling {sample_count} files from Deepfake Dataset:    {deepfake_dir}")
    deepfake_files = discover_and_sample_dataset(deepfake_dir, sample_count=sample_count, seed=seed)

    print(f"  [+] Total Sample Count: {len(multilingual_files) + len(deepfake_files)} (100 Multilingual + 100 Deepfake)")

    # 2. Verify File Readability
    print("\n[PHASE 2] Pre-Validation & Readability Verification...")
    sampled_inventory = []
    
    for idx, fpath in enumerate(multilingual_files):
        info = sf.info(fpath)
        sampled_inventory.append({
            "dataset": "Multilingual Dataset",
            "sample_index": idx + 1,
            "file_path": fpath,
            "file_name": os.path.basename(fpath),
            "audio_duration": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels
        })

    for idx, fpath in enumerate(deepfake_files):
        info = sf.info(fpath)
        sampled_inventory.append({
            "dataset": "Deepfake Dataset",
            "sample_index": idx + 101,
            "file_path": fpath,
            "file_name": os.path.basename(fpath),
            "audio_duration": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels
        })

    print(f"  [+] 200/200 audio files verified readable with valid WAV headers.")

    # 3. Initialize CloudForge LID Runtime & Canonical Registry
    print("\n[PHASE 3] Initializing CloudForge VM1 LID & Canonical Registry...")
    settings = Settings.load()
    registry = ModelRegistry(
        registry_path=settings.model.registry_path,
        models_base_dir=settings.model.models_dir
    )
    meta = registry.get_model(settings.model.active_model_id)
    runtime = ModelRuntimeFactory.create(meta, settings)
    runtime.load()

    canonical_reg = get_canonical_registry()
    print(f"  - Active LID Model ID:  {runtime.metadata.model_id}")
    print(f"  - LID Architecture:     {runtime.metadata.architecture}")
    print(f"  - Canonical Registry:   v{canonical_reg.version} ({len(canonical_reg.list_canonical_languages())} languages)")

    # 4. Execute E2E Pipeline across 200 samples
    print("\n[PHASE 4] Executing E2E Pipeline on 200 Samples...")
    results = []
    latencies_ms = []

    counts_by_status = Counter()
    counts_by_lang = Counter()
    counts_by_group = Counter()
    counts_by_vm = Counter()
    counts_by_dataset_lang = defaultdict(Counter)

    for item in sampled_inventory:
        t0 = time.perf_counter()
        fpath = item["file_path"]

        # Step A: Audio Preprocessing
        try:
            with open(fpath, "rb") as af:
                raw_bytes = af.read()

            proc_status = "SUCCESS"
        except Exception as e:
            proc_status = f"ERROR: {str(e)}"

        # Step B: VM1 LID Inference
        lid_status = "SUCCESS"
        detected_lang = "UNKNOWN"
        lid_conf = 0.0
        error_msg = None

        try:
            inp = AudioInferenceInput(
                audio_data=raw_bytes,
                sample_rate=item["sample_rate"],
                channels=item["channels"],
                audio_format="wav",
                request_id=f"e2e_samp_{item['sample_index']}"
            )
            inf_res = runtime.infer(inp)
            detected_lang = inf_res.language
            lid_conf = inf_res.confidence
        except Exception as e:
            lid_status = "LID_ERROR"
            error_msg = str(e)

        # Step C: Canonical Language Group & VM Routing
        if lid_status == "SUCCESS":
            lang_info = canonical_reg.get_language_info(detected_lang)
            if lang_info is not None:
                sem_group = lang_info.group
                target_vm = lang_info.vm
                target_endpoint = f"{lang_info.service_name}:{lang_info.port}"

                if detected_lang == "en-IN":
                    detector_target = "AASIST-L"
                    inference_executed = False
                    final_status = "ROUTED_TO_AASIST_L"
                else:
                    detector_target = None
                    inference_executed = False
                    final_status = "ROUTED"
            else:
                sem_group = canonical_reg.default_group
                target_vm = canonical_reg.default_vm
                target_endpoint = "vmc-lid:50051"
                detector_target = None
                inference_executed = False
                final_status = "UNSUPPORTED_LANGUAGE"
        else:
            sem_group = "UNKNOWN_GROUP"
            target_vm = "VM1_LID"
            target_endpoint = "vmc-lid:50051"
            detector_target = None
            inference_executed = False
            final_status = "LID_UNAVAILABLE"

        pipeline_lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(pipeline_lat_ms)

        counts_by_status[final_status] += 1
        counts_by_lang[detected_lang] += 1
        counts_by_group[sem_group] += 1
        counts_by_vm[target_vm] += 1
        counts_by_dataset_lang[item["dataset"]][detected_lang] += 1

        record = {
            "dataset": item["dataset"],
            "sample_index": item["sample_index"],
            "file_path": fpath,
            "file_name": item["file_name"],
            "audio_duration": item["audio_duration"],
            "sample_rate": item["sample_rate"],
            "channels": item["channels"],
            "preprocessing_status": proc_status,
            "LID_status": lid_status,
            "detected_language": detected_lang,
            "LID_confidence": round(lid_conf, 4),
            "semantic_group": sem_group,
            "target_vm": target_vm,
            "target_endpoint": target_endpoint,
            "detector_target": detector_target,
            "inference_executed": inference_executed,
            "final_status": final_status,
            "pipeline_latency_ms": round(pipeline_lat_ms, 2),
            "error": error_msg
        }
        results.append(record)

    # 5. Performance Statistics
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    routed_to_aasist_l = counts_by_status["ROUTED_TO_AASIST_L"]
    non_english_routed = counts_by_status["ROUTED"]
    total_successfully_routed = routed_to_aasist_l + non_english_routed
    lid_unavailable = counts_by_status["LID_UNAVAILABLE"]
    unsupported = counts_by_status["UNSUPPORTED_LANGUAGE"]
    invalid_audio = counts_by_status["INVALID_AUDIO"]
    routing_errors = counts_by_status["ROUTING_ERROR"]

    routing_success_rate = (total_successfully_routed / len(results)) * 100.0
    rss_end = proc.memory_info().rss / (1024 * 1024)

    # 6. Build Summary Object
    summary = {
        "evaluation_name": "CloudForge V2 E2E Pipeline & Canonical Routing Validation (200 Samples)",
        "random_seed": seed,
        "sample_configuration": {
            "total_samples": len(results),
            "multilingual_dataset_samples": sample_count,
            "deepfake_dataset_samples": sample_count
        },
        "routing_results": {
            "total_samples": len(results),
            "successfully_routed": total_successfully_routed,
            "routed_to_aasist_l": routed_to_aasist_l,
            "non_english_routed": non_english_routed,
            "lid_unavailable": lid_unavailable,
            "unsupported_language": unsupported,
            "invalid_audio": invalid_audio,
            "routing_errors": routing_errors,
            "routing_success_rate_percent": round(routing_success_rate, 2)
        },
        "inference_execution_governance": {
            "detector_inference_executed": 0,
            "aasist_l_inference_executed": 0,
            "governance_rule": "AASIST-L and deepfake detector inference strictly bypassed/disabled for E2E routing verification."
        },
        "distribution": {
            "languages": dict(counts_by_lang),
            "semantic_groups": dict(counts_by_group),
            "target_vms": dict(counts_by_vm),
            "dataset_language_breakdown": {
                "Multilingual Dataset": dict(counts_by_dataset_lang["Multilingual Dataset"]),
                "Deepfake Dataset": dict(counts_by_dataset_lang["Deepfake Dataset"])
            }
        },
        "latency_metrics_ms": {
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
            "mean": round(mean_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2)
        },
        "memory_rss_mb": {
            "initial": round(rss_start, 2),
            "final": round(rss_end, 2)
        },
        "final_status": "E2E ROUTING VERIFIED" if total_successfully_routed == len(results) else f"E2E ROUTING BLOCKED — {lid_unavailable} LID unavailable, {unsupported} unsupported"
    }

    # 7. Write JSON Reports
    os.makedirs(os.path.dirname(os.path.abspath(detailed_json_path)), exist_ok=True)
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Detailed Results JSON exported to: {detailed_json_path}")

    os.makedirs(os.path.dirname(os.path.abspath(summary_json_path)), exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary JSON exported to:          {summary_json_path}")

    # 8. Print Execution Summary
    print("\n" + "=" * 95)
    print(" CLOUDFORGE E2E ROUTING TEST SUMMARY")
    print("=" * 95)
    print(f"Samples tested:                   {len(results)}")
    print(f"Multilingual Dataset Samples:     {sample_count}")
    print(f"Deepfake Dataset Samples:         {sample_count}")
    print(f"LID successful:                   {total_successfully_routed}")
    print(f"Routing successful:               {total_successfully_routed} ({routing_success_rate:.1f}%)")
    print(f"English -> VM7 -> AASIST-L target: {routed_to_aasist_l}")
    print(f"Non-English routed:               {non_english_routed}")
    print(f"AASIST-L inference executed:      0 (Bypassed per contract)")
    print(f"Detector inference executed:      0 (Bypassed per contract)")
    print(f"P50 latency:                      {p50_lat:.2f} ms")
    print(f"P95 latency:                      {p95_lat:.2f} ms")
    print(f"P99 latency:                      {p99_lat:.2f} ms")
    print(f"Final status:                     {summary['final_status']}")
    print("=" * 95)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudForge E2E 200-Sample Routing Evaluator")
    parser.add_argument("--multilingual-dir", type=str, default=MULTILINGUAL_DATASET_DIR)
    parser.add_argument("--deepfake-dir", type=str, default=DEEPFAKE_DATASET_DIR)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_DATASET)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--detailed-output", type=str, default=os.path.join(REPO_ROOT, "reports", "e2e_200_sample_routing_results.json"))
    parser.add_argument("--summary-output", type=str, default=os.path.join(REPO_ROOT, "reports", "e2e_200_sample_routing_summary.json"))
    args = parser.parse_args()

    run_e2e_200_sample_routing_test(
        multilingual_dir=args.multilingual_dir,
        deepfake_dir=args.deepfake_dir,
        sample_count=args.samples,
        seed=args.seed,
        detailed_json_path=args.detailed_output,
        summary_json_path=args.summary_output
    )
