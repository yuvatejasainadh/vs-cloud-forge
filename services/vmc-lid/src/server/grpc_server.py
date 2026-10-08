from concurrent import futures
import time
from typing import Optional
import grpc
from grpc_reflection.v1alpha import reflection
from grpc_health.v1 import health_pb2, health_pb2_grpc

from config.settings import Settings
from service.lid_service import LIDService
from server.health_server import HealthServicer
from server.metrics_server import MetricsServer
from telemetry.logging import get_logger
from vmc.lid.v1 import lid_pb2, lid_pb2_grpc

logger = get_logger("grpc_server")


class UnaryLoggingInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or not handler.unary_unary:
            return handler

        def logging_unary_unary(request, context):
            start = time.perf_counter()
            method = handler_call_details.method
            try:
                response = handler.unary_unary(request, context)
                duration_ms = (time.perf_counter() - start) * 1000.0
                return response
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.error(
                    f"gRPC call failed: method={method} duration_ms={duration_ms:.2f} error={e}",
                    extra={
                        "event": "GRPC_CALL_ERROR",
                        "method": method,
                        "duration_ms": duration_ms,
                        "error": str(e)
                    }
                )
                raise

        return grpc.unary_unary_rpc_method_handler(
            logging_unary_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


class CloudForgeLIDServer:
    """
    CloudForge VMC-LID gRPC and Metrics Server.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        self.lid_service = LIDService(self.settings)
        self.health_servicer = HealthServicer(self.settings, self.lid_service.runtime)
        self.metrics_server: Optional[MetricsServer] = None
        self._server: Optional[grpc.Server] = None

        # Record static build info
        self.lid_service.metrics.set_service_info(
            service=self.settings.service.name,
            version=self.settings.service.version,
            platform=self.settings.service.platform,
            environment=self.settings.service.environment
        )

    def start(self) -> None:
        # Start HTTP metrics server on port 9090
        if self.settings.server.metrics_enabled:
            self.metrics_server = MetricsServer(
                self.settings,
                health_servicer=self.health_servicer,
                registry=self.lid_service.metrics.registry
            )
            self.metrics_server.start()

        # Configure gRPC options (32 MB message limits)
        max_recv = self.settings.server.max_recv_msg_size_mb * 1024 * 1024
        max_send = self.settings.server.max_send_msg_size_mb * 1024 * 1024

        options = [
            ("grpc.max_receive_message_length", max_recv),
            ("grpc.max_send_message_length", max_send),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ]

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=options,
            interceptors=[UnaryLoggingInterceptor()]
        )

        # Register LID Service
        lid_pb2_grpc.add_LIDServiceServicer_to_server(self.lid_service, self._server)

        # Register Health Service
        health_pb2_grpc.add_HealthServicer_to_server(self.health_servicer, self._server)

        # Register Reflection
        SERVICE_NAMES = (
            lid_pb2.DESCRIPTOR.services_by_name["LIDService"].full_name,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, self._server)

        # Bind gRPC port
        grpc_addr = f"0.0.0.0:{self.settings.server.grpc_port}"
        self._server.add_insecure_port(grpc_addr)
        self._server.start()

        # Update initial serving status
        self.health_servicer.check_readiness()

        logger.info(
            f"CloudForge VMC-LID gRPC Server started",
            extra={
                "event": "SERVER_STARTED",
                "service": self.settings.service.name,
                "version": self.settings.service.version,
                "grpc_address": grpc_addr,
                "metrics_port": self.settings.server.metrics_port,
                "active_model": self.lid_service.runtime.metadata.model_id,
                "max_recv_mb": self.settings.server.max_recv_msg_size_mb,
            }
        )

    def stop(self, grace: int = 5) -> None:
        logger.info(f"Stopping CloudForge VMC-LID Server gracefully (grace={grace}s)...")
        if self.metrics_server:
            self.metrics_server.stop()
        if self._server:
            event = self._server.stop(grace)
            event.wait(timeout=grace)
            logger.info("gRPC server stopped cleanly")

    def wait_for_termination(self) -> None:
        if self._server:
            self._server.wait_for_termination()
