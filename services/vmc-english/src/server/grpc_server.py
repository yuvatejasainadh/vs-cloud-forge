import signal
import sys
import time
from concurrent import futures
from typing import Optional
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from config.settings import Settings
from service.detector_service import EnglishDetectorService
from server.metrics_server import MetricsServer
from telemetry.logging import get_logger

from vmc.detector.v1 import detector_pb2_grpc

logger = get_logger("grpc_server")


class EnglishDetectorGRPCServer:
    """
    Production-grade gRPC Server for VM7 English Deepfake Detection Service (vmc-english).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        self.grpc_server: Optional[grpc.Server] = None
        self.metrics_server: Optional[MetricsServer] = None
        self.detector_service: Optional[EnglishDetectorService] = None

    def start(self) -> None:
        port = self.settings.server.grpc_port
        max_msg_size = self.settings.server.max_recv_msg_size_mb * 1024 * 1024

        server_options = [
            ("grpc.max_receive_message_length", max_msg_size),
            ("grpc.max_send_message_length", max_msg_size),
            ("grpc.so_reuseport", 1),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ]

        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=server_options
        )

        # Instantiate service and register
        self.detector_service = EnglishDetectorService(self.settings)
        detector_pb2_grpc.add_DetectorServiceServicer_to_server(self.detector_service, self.grpc_server)

        # Standard gRPC health check service
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.grpc_server)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set("vmc.detector.v1.DetectorService", health_pb2.HealthCheckResponse.SERVING)

        self.grpc_server.add_insecure_port(f"0.0.0.0:{port}")
        self.grpc_server.start()
        logger.info(f"VMC-ENGLISH gRPC server running on 0.0.0.0:{port} (VM7_ENGLISH)")

        # Start metrics / HTTP health server
        if self.settings.server.metrics_enabled:
            readiness_fn = lambda: self.detector_service.runtime.is_ready() if self.detector_service else False
            self.metrics_server = MetricsServer(
                port=self.settings.server.metrics_port,
                readiness_check=readiness_fn
            )
            self.metrics_server.start()

    def stop(self, grace_period: int = 5) -> None:
        logger.info("Stopping VMC-ENGLISH service...")
        if self.metrics_server:
            self.metrics_server.stop()
        if self.grpc_server:
            self.grpc_server.stop(grace_period)
        logger.info("VMC-ENGLISH service stopped successfully")

    def serve_forever(self) -> None:
        self.start()

        def sig_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.stop(self.settings.server.graceful_timeout_seconds)
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop(self.settings.server.graceful_timeout_seconds)


if __name__ == "__main__":
    server = EnglishDetectorGRPCServer()
    server.serve_forever()
