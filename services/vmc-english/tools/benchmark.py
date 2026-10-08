#!/usr/bin/env python3
"""
CloudForge VM7 English AASIST-L Service Benchmarking CLI.
"""

import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
GEN_DIR = os.path.join(SRC_DIR, "gen")

sys.path = [SRC_DIR, GEN_DIR, BASE_DIR] + [p for p in sys.path if p not in (SRC_DIR, GEN_DIR, BASE_DIR)]

from benchmarks.detectors.english_aasist.runner import run_benchmark


if __name__ == "__main__":
    run_benchmark(iterations=10)
