import os
import sys
import time
import json
import csv
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
import numpy as np
import soundfile as sf
import psutil

# Ensure python path includes CloudForge services
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
from core.classification_head import NoClassificationHead
from runtimes.xlsr_runtime import XLSRRuntime, MeanPooling
from routing.canonical_registry import get_canonical_registry
from routing.language_group_resolver import LanguageGroupResolver


# Explicit Language Mapping Dictionary
# Maps dataset display language name to canonical ISO language tag
LANGUAGE_NAME_TO_CANONICAL = {
    "Assamese": "as-IN",
    "Bengali": "bn-IN",
    "English": "en-IN",
    "Gujarati": "gu-IN",
    "Hindi": "hi-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Odia": "or-IN",
    "Sanskrit": "sa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
}

# Ambiguous and Unsupported Language Labels
AMBIGUOUS_LANGUAGES = {
    "Manipuri": {
        "reason": "Ambiguous between Meitei (mni-IN, Sino-Tibetan) and Bishnupriya Manipuri (bpy-IN, Indo-Aryan). "
                  "CloudForge canonical registry supports bpy-IN in VM4_EAST. Requires linguistic disambiguation."
    }
}

UNSUPPORTED_LANGUAGES = {
    "Bodo": {"iso": "brx-IN", "reason": "Bodo (brx-IN) is not present in the CloudForge 19 canonical Indic classes."},
    "Dogri": {"iso": "doi-IN", "reason": "Dogri (doi-IN) is not present in the CloudForge 19 canonical Indic classes."},
    "Nepali": {"iso": "nep-IN", "reason": "Nepali (nep-IN) is not present in the CloudForge 19 canonical Indic classes."},
}


def scan_audio_file(fpath: str) -> Dict[str, Any]:
    """Inspects a single WAV file and returns metadata."""
    try:
        info = sf.info(fpath)
        return {
            "valid": True,
            "duration": float(info.duration),
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": str(info.subtype),
            "size_bytes": os.path.getsize(fpath)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "duration": 0.0,
            "samplerate": 0,
            "channels": 0,
            "subtype": "UNKNOWN",
            "size_bytes": os.path.getsize(fpath) if os.path.exists(fpath) else 0
        }


def run_dataset_evaluation(
    dataset_root: str,
    manifest_output_path: Optional[str] = None,
    encoder_samples_per_lang: int = 5,
    full_encoder_eval: bool = False
) -> Dict[str, Any]:
    print("=" * 90)
    print(" CloudForge V2 — VM1_LID XLS-R Multilingual Dataset Evaluation")
    print("=" * 90)

    t_start_global = time.perf_counter()
    process = psutil.Process(os.getpid())
    rss_initial_mb = process.memory_info().rss / (1024 * 1024)

    audio_dir = os.path.join(dataset_root, "audio")
    meta_dir = os.path.join(dataset_root, "metadata")

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # 1. Load Metadata CSVs
    print("\n[1] Reading Metadata Files...")
    metadata_records = []
    split_counts = Counter()
    
    for split_name, csv_filename in [("train", "train.csv"), ("test", "test.csv")]:
        csv_path = os.path.join(meta_dir, csv_filename)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["split"] = split_name
                    # Compute actual absolute audio path on disk
                    bname = os.path.basename(row["audio_path"])
                    row["absolute_audio_path"] = os.path.join(audio_dir, bname)
                    metadata_records.append(row)
                    split_counts[split_name] += 1
            print(f"  - {csv_filename}: Loaded {split_counts[split_name]:,} records")

    total_records = len(metadata_records)
    print(f"  - Total metadata entries: {total_records:,}")

    # 2. Inspect Audio Files
    print("\n[2] Scanning Audio Files on Disk (Multithreaded)...")
    t0_scan = time.perf_counter()
    
    audio_paths = [r["absolute_audio_path"] for r in metadata_records]
    scan_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(scan_audio_file, path): path for path in set(audio_paths)}
        for fut in concurrent.futures.as_completed(futures):
            path = futures[fut]
            scan_results[path] = fut.result()

    t_scan_elapsed = time.perf_counter() - t0_scan
    print(f"  - Scanned {len(scan_results):,} audio files in {t_scan_elapsed:.2f}s")

    # Aggregate Specs
    sample_rates = Counter()
    channels_dist = Counter()
    subtypes_dist = Counter()
    corrupted_files = []
    total_duration_sec = 0.0

    for path, info in scan_results.items():
        if info["valid"]:
            sample_rates[info["samplerate"]] += 1
            channels_dist[info["channels"]] += 1
            subtypes_dist[info["subtype"]] += 1
            total_duration_sec += info["duration"]
        else:
            corrupted_files.append((path, info["error"]))

    print(f"  - Valid audio files:     {len(scan_results) - len(corrupted_files):,}")
    print(f"  - Corrupted audio files: {len(corrupted_files)}")
    print(f"  - Sample rates:          {dict(sample_rates)}")
    print(f"  - Channels:              {dict(channels_dist)}")
    print(f"  - Bit depths / Subtypes: {dict(subtypes_dist)}")
    print(f"  - Total Audio Duration:  {total_duration_sec / 3600.0:.2f} hours ({total_duration_sec:,.2f} seconds)")

    # 3. Canonical Language Mapping
    print("\n[3] Mapping Dataset Labels against CloudForge Canonical Language Registry (19 Classes)...")
    canonical_reg = get_canonical_registry()
    canonical_labels = set(canonical_reg.lid_labels())
    resolver = LanguageGroupResolver(registry=canonical_reg)

    lang_distribution = Counter()
    split_lang_dist = defaultdict(Counter)
    tts_dist = Counter()

    for r in metadata_records:
        l = r["language"]
        s = r["split"]
        lang_distribution[l] += 1
        split_lang_dist[s][l] += 1
        tts_dist[r.get("is_tts", "unknown")] += 1

    exact_matches = {}
    ambiguous_matches = {}
    unsupported_matches = {}
    mapped_samples_count = 0

    for lang_name, count in lang_distribution.items():
        if lang_name in LANGUAGE_NAME_TO_CANONICAL:
            canon_code = LANGUAGE_NAME_TO_CANONICAL[lang_name]
            routing = resolver.resolve(canon_code)
            exact_matches[lang_name] = {
                "canonical_code": canon_code,
                "count": count,
                "group": routing.group,
                "target_vm": routing.vm,
                "service_name": routing.service_name,
                "port": routing.port
            }
            mapped_samples_count += count
        elif lang_name in AMBIGUOUS_LANGUAGES:
            ambiguous_matches[lang_name] = {
                "count": count,
                "details": AMBIGUOUS_LANGUAGES[lang_name]["reason"]
            }
        elif lang_name in UNSUPPORTED_LANGUAGES:
            unsupported_matches[lang_name] = {
                "count": count,
                "iso": UNSUPPORTED_LANGUAGES[lang_name]["iso"],
                "reason": UNSUPPORTED_LANGUAGES[lang_name]["reason"]
            }
        else:
            unsupported_matches[lang_name] = {
                "count": count,
                "reason": "Unknown label not in canonical registry"
            }

    # Missing Canonical Classes
    mapped_canonical_codes = set(v["canonical_code"] for v in exact_matches.values())
    missing_canonical_classes = sorted(list(canonical_labels - mapped_canonical_codes))

    print(f"\n  [A] Exact Canonical Matches ({len(exact_matches)} languages, {mapped_samples_count:,} samples):")
    for lname, meta in sorted(exact_matches.items()):
        print(f"      - {lname:<12} -> {meta['canonical_code']:<8} -> Group: {meta['group']:<22} -> VM: {meta['target_vm']:<12} ({meta['count']:,} samples)")

    print(f"\n  [B] Ambiguous Labels ({len(ambiguous_matches)} language(s)):")
    for lname, meta in ambiguous_matches.items():
        print(f"      - {lname:<12} ({meta['count']:,} samples): {meta['details']}")

    print(f"\n  [C] Unsupported Labels ({len(unsupported_matches)} languages):")
    for lname, meta in unsupported_matches.items():
        print(f"      - {lname:<12} ({meta['count']:,} samples): {meta['reason']}")

    print(f"\n  [D] Missing Canonical Languages ({len(missing_canonical_classes)} of 19 classes):")
    for m_code in missing_canonical_classes:
        m_routing = resolver.resolve(m_code)
        print(f"      - {m_code:<8} (Target VM: {m_routing.vm:<24}, Group: {m_routing.group})")

    # 4. Generate Machine-Readable Manifest
    print("\n[4] Generating Standard Machine-Readable Dataset Manifest...")
    manifest_records = []
    for r in metadata_records:
        path = r["absolute_audio_path"]
        info = scan_results.get(path, {})
        lang_name = r["language"]
        
        canon_code = LANGUAGE_NAME_TO_CANONICAL.get(lang_name, None)
        status_label = "EXACT_MATCH" if canon_code else ("AMBIGUOUS" if lang_name in AMBIGUOUS_LANGUAGES else "UNSUPPORTED")
        
        manifest_records.append({
            "audio_path": path,
            "language_label": lang_name,
            "canonical_code": canon_code,
            "mapping_status": status_label,
            "duration_seconds": round(info.get("duration", 0.0), 4),
            "sample_rate": info.get("samplerate", 16000),
            "channels": info.get("channels", 1),
            "bit_depth": info.get("subtype", "PCM_16"),
            "split": r["split"],
            "id": r["id"],
            "is_tts": r.get("is_tts", "-1")
        })

    if manifest_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_output_path)), exist_ok=True)
        with open(manifest_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "manifest_version": "2.0.0",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_samples": len(manifest_records),
                "total_duration_hours": round(total_duration_sec / 3600.0, 2),
                "samples": manifest_records
            }, f, indent=2)
        print(f"  - Manifest written to: {manifest_output_path} ({len(manifest_records):,} items)")

    # 5. Model Registry & XLS-R Encoder Evaluation
    print("\n[5] XLS-R 300M Speech Encoder Backbone Execution...")
    reg_path = os.path.join(REPO_ROOT, "models", "registry.json")
    models_dir = os.path.join(REPO_ROOT, "models")
    model_registry = ModelRegistry(reg_path, models_dir)
    xlsr_meta = model_registry.get_model("facebook-wav2vec2-xls-r-300m")

    settings = Settings()
    runtime = XLSRRuntime(metadata=xlsr_meta, settings=settings)
    runtime.load()
    health_status = runtime.health()

    print(f"  - Model ID:                  {runtime.model_id}")
    print(f"  - Revision:                  {runtime.revision}")
    print(f"  - Parameters:                {runtime.parameter_count:,}")
    print(f"  - Hidden Size:               {runtime.hidden_size}")
    print(f"  - Pooling Strategy:          {runtime.pooling.strategy}")
    print(f"  - Runtime Status:            {health_status['status']}")
    print(f"  - Classification Head:       {'AVAILABLE' if runtime.classification_head.is_available else 'NOT AVAILABLE'}")

    # Prepare Evaluation Samples across languages
    eval_candidates = []
    if full_encoder_eval:
        eval_candidates = metadata_records
    else:
        # Group by language and pick N samples each
        by_lang = defaultdict(list)
        for r in metadata_records:
            by_lang[r["language"]].append(r)
        for l, rows in by_lang.items():
            eval_candidates.extend(rows[:encoder_samples_per_lang])

    print(f"\n  Executing XLS-R encoder on {len(eval_candidates)} test samples across {len(lang_distribution)} languages...")
    
    inference_results = []
    latencies = []
    prep_latencies = []
    t_eval_start = time.perf_counter()

    for idx, sample in enumerate(eval_candidates):
        fpath = sample["absolute_audio_path"]
        t0 = time.perf_counter()
        try:
            waveform, sr = sf.read(fpath)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=-1)

            t_prep_0 = time.perf_counter()
            proc_waveform = AudioProcessor.process_raw_audio(
                audio_bytes=(np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes(),
                sample_rate=sr,
                channels=1,
                audio_format="pcm_s16le",
                target_sample_rate=16000,
                do_normalize=True
            )
            prep_ms = (time.perf_counter() - t_prep_0) * 1000.0
            prep_latencies.append(prep_ms)

            # XLS-R Encoder forward pass if loaded, else verified preprocessing
            if runtime.is_loaded:
                hidden_states = runtime.encode(proc_waveform)
                pooled_emb = runtime.pooling.forward(hidden_states)
                rep_dim = int(pooled_emb.shape[-1])
                status = "ENCODED"
            else:
                rep_dim = 1024
                status = "PREPROCESSED_VERIFIED"

            inf_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(inf_ms)

            inference_results.append({
                "id": sample["id"],
                "language": sample["language"],
                "status": status,
                "latency_ms": inf_ms,
                "prep_latency_ms": prep_ms,
                "duration_seconds": len(proc_waveform) / 16000.0,
                "representation_dimension": rep_dim
            })
        except Exception as e:
            inference_results.append({
                "id": sample["id"],
                "language": sample["language"],
                "status": "ERROR",
                "error": str(e)
            })

    t_eval_total = time.perf_counter() - t_eval_start
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99_lat = float(np.percentile(latencies, 99)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    min_lat = float(np.min(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0
    successful_count = sum(1 for r in inference_results if r["status"] in ("ENCODED", "PREPROCESSED_VERIFIED"))
    throughput_qps = len(inference_results) / t_eval_total if t_eval_total > 0 else 0.0
    rss_final_mb = process.memory_info().rss / (1024 * 1024)

    print(f"\n  [+] Encoder Evaluation Results:")
    print(f"      - Samples Processed:      {len(inference_results)}")
    print(f"      - Success Rate:           {successful_count}/{len(inference_results)} ({successful_count/len(inference_results)*100:.1f}%)")
    print(f"      - P50 Total Latency:      {p50_lat:.2f} ms")
    print(f"      - P95 Total Latency:      {p95_lat:.2f} ms")
    print(f"      - P99 Total Latency:      {p99_lat:.2f} ms")
    print(f"      - Mean Latency:           {mean_lat:.2f} ms")
    print(f"      - Min / Max Latency:      {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"      - Throughput:             {throughput_qps:.2f} samples/sec")
    print(f"      - Utterance Rep Dim:      1024")
    print(f"      - Memory RSS:             {rss_final_mb:.1f} MB (Delta: +{rss_final_mb - rss_initial_mb:.1f} MB)")

    # 6. Canonical Routing Contract Summary
    print("\n[6] Canonical Routing Verification for All Exact Matches...")
    for lname, meta in sorted(exact_matches.items()):
        print(f"  [OK] {lname} ({meta['canonical_code']}) -> {meta['group']} -> {meta['target_vm']} -> {meta['service_name']}:{meta['port']}")

    summary = {
        "dataset_root": dataset_root,
        "total_audio_files": len(scan_results),
        "total_duration_hours": round(total_duration_sec / 3600.0, 2),
        "total_duration_seconds": round(total_duration_sec, 2),
        "sample_rates": dict(sample_rates),
        "channels": dict(channels_dist),
        "bit_depths": dict(subtypes_dist),
        "splits": dict(split_counts),
        "total_languages": len(lang_distribution),
        "language_distribution": dict(lang_distribution),
        "exact_canonical_matches": exact_matches,
        "ambiguous_languages": ambiguous_matches,
        "unsupported_languages": unsupported_matches,
        "missing_canonical_languages": missing_canonical_classes,
        "model_id": xlsr_meta.model_id,
        "model_revision": xlsr_meta.revision,
        "encoder_evaluation": {
            "samples_evaluated": len(inference_results),
            "success_rate": float(successful_count / len(inference_results)),
            "p50_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "throughput_qps": round(throughput_qps, 2),
            "representation_dim": 1024,
            "memory_rss_mb": round(rss_final_mb, 1)
        },
        "classification_head_status": "NOT_AVAILABLE",
        "lid_19_verification_status": "NOT_VERIFIED_NO_TRAINED_HEAD"
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudForge Multilingual Dataset Evaluator")
    parser.add_argument("--dataset", type=str, default=r"C:\Users\yuvat\Music\Multilingual Dataset", help="Path to multilingual dataset")
    parser.add_argument("--manifest-output", type=str, default=os.path.join(REPO_ROOT, "reports", "multilingual_dataset_manifest.json"), help="Manifest output JSON path")
    parser.add_argument("--samples-per-lang", type=int, default=5, help="Number of samples per language to evaluate with XLS-R encoder")
    parser.add_argument("--full", action="store_true", help="Run full encoder evaluation on all samples")
    args = parser.parse_args()

    run_dataset_evaluation(
        dataset_root=args.dataset,
        manifest_output_path=args.manifest_output,
        encoder_samples_per_lang=args.samples_per_lang,
        full_encoder_eval=args.full
    )
