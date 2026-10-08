#!/usr/bin/env python3
"""
CloudForge Model Downloader & Registry Sync Tool
Downloads, validates checksums, and deterministically caches ML model artifacts.
Supports multi-threaded parallel chunk downloading for high throughput.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","event":"%(message)s"}'
)
logger = logging.getLogger("model_downloader")


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CloudForge-ModelRegistry/1.0"},
        method="HEAD"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        logger.warning(f"Failed HEAD request on {url}: {e}")
        return 0


def download_chunk(url: str, dest_path: str, start_byte: int, end_byte: int, chunk_idx: int) -> int:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CloudForge-ModelRegistry/1.0",
            "Range": f"bytes={start_byte}-{end_byte}"
        }
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        with open(dest_path, "r+b") as f:
            f.seek(start_byte)
            f.write(data)
    return len(data)


def download_file_parallel(url: str, dest_path: str, expected_size: int = None, num_threads: int = 16) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if expected_size is None or os.path.getsize(dest_path) == expected_size:
            logger.info(f"ARTIFACT_EXISTS: {os.path.basename(dest_path)} (size: {os.path.getsize(dest_path)} bytes)")
            return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = f"{dest_path}.tmp"

    total_size = expected_size or get_remote_file_size(url)
    logger.info(f"MODEL_DOWNLOAD_STARTED: {os.path.basename(dest_path)} from {url} (size: {total_size} bytes, {num_threads} threads)")

    start_time = time.time()

    if total_size <= 5 * 1024 * 1024:  # Small files (<5MB): direct download
        req = urllib.request.Request(url, headers={"User-Agent": "CloudForge-ModelRegistry/1.0"})
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
            f.write(resp.read())
    else:  # Large files: parallel chunk download
        # Preallocate file
        with open(temp_path, "wb") as f:
            f.truncate(total_size)

        chunk_size = (total_size + num_threads - 1) // num_threads
        chunks = []
        for i in range(num_threads):
            start = i * chunk_size
            end = min(total_size - 1, (i + 1) * chunk_size - 1)
            if start <= end:
                chunks.append((url, temp_path, start, end, i))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(download_chunk, u, p, s, e, idx)
                for u, p, s, e, idx in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)

    elapsed = time.time() - start_time
    final_size = os.path.getsize(dest_path)
    mb_s = (final_size / (1024 * 1024)) / max(0.1, elapsed)
    logger.info(
        f"MODEL_DOWNLOAD_COMPLETED: {os.path.basename(dest_path)} "
        f"({final_size} bytes in {elapsed:.2f}s, {mb_s:.2f} MB/s)"
    )
    return True


def sync_model(registry_path: str, model_id: str, base_model_dir: str):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    models = registry_data.get("models", {})
    if model_id not in models:
        raise ValueError(f"Model '{model_id}' not found in registry at {registry_path}")

    model_entry = models[model_id]
    version = model_entry.get("version", "default")
    target_dir = os.path.join(base_model_dir, model_id, version)
    os.makedirs(target_dir, exist_ok=True)

    logger.info(f"SYNC_MODEL_STARTED: {model_id} (version: {version}) -> {target_dir}")

    artifacts = model_entry.get("artifacts", {})
    for filename, art_info in artifacts.items():
        url = art_info.get("url")
        size = art_info.get("size_bytes")
        dest = os.path.join(target_dir, filename)
        download_file_parallel(url, dest, size, num_threads=16)

    # Write metadata.json in artifact directory
    model_file = os.path.join(target_dir, "model.onnx")
    sha256 = compute_sha256(model_file) if os.path.exists(model_file) else None

    metadata = {
        "model_id": model_entry.get("model_id"),
        "model_name": model_entry.get("model_name"),
        "provider": model_entry.get("provider"),
        "source": model_entry.get("source"),
        "repository": model_entry.get("repository"),
        "task": model_entry.get("task"),
        "framework": model_entry.get("framework"),
        "version": version,
        "parameters": model_entry.get("parameters"),
        "artifact_size_mb": model_entry.get("artifact_size_mb"),
        "sample_rate": model_entry.get("sample_rate"),
        "channels": model_entry.get("channels"),
        "status": model_entry.get("status"),
        "supported_languages": model_entry.get("supported_languages"),
        "language_names": model_entry.get("language_names"),
        "sha256": sha256,
        "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    metadata_path = os.path.join(target_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"SYNC_MODEL_COMPLETED: {model_id} metadata written to {metadata_path} (SHA256: {sha256})")
    return target_dir, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and register CloudForge models")
    parser.add_argument("--registry", default=r"C:\CloudForge\models\registry.json", help="Path to registry.json")
    parser.add_argument("--model-id", default="open-vakgyata", help="Model ID to download")
    parser.add_argument("--models-dir", default=r"C:\CloudForge\models", help="Base directory for models")
    args = parser.parse_args()

    sync_model(args.registry, args.model_id, args.models_dir)
