#!/usr/bin/env python3
"""
CloudForge VMC-LID Server Main Entrypoint.
"""

import os
import signal
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
GEN_DIR = os.path.join(SRC_DIR, "gen")

# Prioritize SRC_DIR and GEN_DIR ahead of script directory
sys.path = [SRC_DIR, GEN_DIR, BASE_DIR] + [p for p in sys.path if p not in (SRC_DIR, GEN_DIR, BASE_DIR, os.path.dirname(__file__))]

from config.settings import Settings
from server.grpc_server import CloudForgeLIDServer
from telemetry.logging import get_logger

logger = get_logger("main")


def main():
    logger.info("Initializing CloudForge VMC-LID Server...")
    settings = Settings.load()

    server = CloudForgeLIDServer(settings)
    server.start()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        server.stop(grace=settings.server.graceful_timeout_seconds)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        server.stop(grace=1)


if __name__ == "__main__":
    main()
