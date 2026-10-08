#!/usr/bin/env python3
"""
CloudForge VMC-LID Test Client CLI.
Sends audio payloads, health check probes, and model metadata queries over gRPC.
"""

import argparse
import os
import sys
import time
import grpc
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
GEN_DIR = os.path.join(SRC_DIR, "gen")

sys.path = [SRC_DIR, GEN_DIR, BASE_DIR] + [p for p in sys.path if p not in (SRC_DIR, GEN_DIR, BASE_DIR, os.path.dirname(__file__))]

from vmc.lid.v1 import lid_pb2, lid_pb2_grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def run_health_check(stub_health, service_name: str = ""):
    print(f"\n--- Checking Health for '{service_name}' ---")
    req = health_pb2.HealthCheckRequest(service=service_name)
    try:
        resp = stub_health.Check(req)
        status_str = health_pb2.HealthCheckResponse.ServingStatus.Name(resp.status)
        print(f"Health Status: {status_str}")
        return resp.status == health_pb2.HealthCheckResponse.SERVING
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return False


def run_model_info(stub_lid, model_id: str = ""):
    print(f"\n--- Querying Model Info (target='{model_id}') ---")
    req = lid_pb2.ModelInfoRequest(model_id=model_id)
    try:
        resp = stub_lid.GetModelInfo(req)
        print(f"Model ID:            {resp.model_id}")
        print(f"Model Name:          {resp.model_name}")
        print(f"Provider:            {resp.provider}")
        print(f"Framework:           {resp.framework}")
        print(f"Version:             {resp.version}")
        print(f"Parameters:          {resp.parameters:,}")
        print(f"Artifact Size:       {resp.artifact_size_mb:.2f} MB")
        print(f"Sample Rate:         {resp.sample_rate} Hz")
        print(f"Channels:            {resp.channels}")
        print(f"Status:              {resp.status}")
        print(f"Supported Languages: {', '.join(resp.supported_languages)}")
    except Exception as e:
        print(f"GetModelInfo Failed: {e}")


def run_inference(stub_lid, audio_path: str = None, duration_sec: float = 2.0):
    print(f"\n--- Running LID Inference ---")
    
    if audio_path and os.path.exists(audio_path):
        print(f"Loading audio file from: {audio_path}")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        sample_rate = 16000
        audio_format = "wav" if audio_path.endswith(".wav") else "pcm_s16le"
    else:
        print(f"Generating {duration_sec}s of synthetic test audio (16kHz PCM mono)...")
        sample_rate = 16000
        num_samples = int(sample_rate * duration_sec)
        # Generate tone with slight noise
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(num_samples)
        int16_samples = (tone * 32767).astype(np.int16)
        audio_bytes = int16_samples.tobytes()
        audio_format = "pcm_s16le"

    req = lid_pb2.IdentifyLanguageRequest(
        audio_content=audio_bytes,
        sample_rate=sample_rate,
        channels=1,
        audio_format=audio_format,
        request_id=f"cli_{int(time.time()*1000)}"
    )

    t0 = time.perf_counter()
    try:
        resp = stub_lid.IdentifyLanguage(req)
        round_trip_ms = (time.perf_counter() - t0) * 1000.0

        status_str = lid_pb2.LIDStatus.Name(resp.status)
        print(f"Status:              {status_str}")
        print(f"Predicted Language:  {resp.language} ({resp.language_name})")
        print(f"Confidence:          {resp.confidence:.4f}")
        print(f"Language Group:      {resp.language_group}")
        print(f"Model ID / Version:  {resp.model_id} ({resp.model_version})")
        print(f"Model Latency:       {resp.inference_duration_ms:.2f} ms")
        print(f"Client Round-Trip:   {round_trip_ms:.2f} ms")
        print("Top Probabilities:")
        sorted_probs = sorted(resp.probabilities.items(), key=lambda x: x[1], reverse=True)
        for lang, prob in sorted_probs[:5]:
            print(f"  - {lang:6s}: {prob*100:6.2f}%")
        return resp
    except Exception as e:
        print(f"Inference Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="CloudForge VMC-LID Test Client")
    parser.add_argument("--addr", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--audio", default=None, help="Path to test audio file")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--info", action="store_true", help="Query model info")
    parser.add_argument("--infer", action="store_true", help="Run inference request")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.addr)
    stub_lid = lid_pb2_grpc.LIDServiceStub(channel)
    stub_health = health_pb2_grpc.HealthStub(channel)

    if args.health:
        run_health_check(stub_health, "")
        run_health_check(stub_health, "vmc.lid.v1.LIDService")

    if args.info:
        run_model_info(stub_lid)

    if args.infer or (not args.health and not args.info):
        run_inference(stub_lid, args.audio)


if __name__ == "__main__":
    main()
