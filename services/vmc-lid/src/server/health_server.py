from typing import Optional, Dict, Any
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from core.model_runtime import ModelRuntime
from config.settings import Settings
from telemetry.logging import get_logger

logger = get_logger("health_server")


class HealthServicer(health_pb2_grpc.HealthServicer):
    """
    gRPC and deep HTTP readiness health checker for CloudForge VMC-LID.
    Guarantees that readiness checks only succeed when model artifacts exist and runtime is fully initialized.
    """

    def __init__(self, settings: Settings, runtime: ModelRuntime):
        self.settings = settings
        self.runtime = runtime
        self._statuses: Dict[str, health_pb2.HealthCheckResponse.ServingStatus] = {
            "": health_pb2.HealthCheckResponse.UNKNOWN,
            self.settings.service.name: health_pb2.HealthCheckResponse.UNKNOWN,
            "vmc.lid.v1.LIDService": health_pb2.HealthCheckResponse.UNKNOWN,
        }

    def check_readiness(self) -> Dict[str, Any]:
        """
        Performs deep verification:
        1. Model artifact exists on disk.
        2. Runtime session is initialized.
        3. Runtime is loaded and responsive.
        4. Configuration exists.
        """
        artifact_exists = False
        if self.runtime and self.runtime.metadata and self.runtime.metadata.artifact_path:
            import os
            artifact_exists = os.path.exists(self.runtime.metadata.artifact_path)

        runtime_health = self.runtime.health() if self.runtime else {}
        is_ready = bool(
            self.runtime is not None
            and self.runtime.is_loaded
            and artifact_exists
            and runtime_health.get("status") == "SERVING"
        )

        overall_status = (
            health_pb2.HealthCheckResponse.SERVING
            if is_ready
            else health_pb2.HealthCheckResponse.NOT_SERVING
        )

        self._statuses[""] = overall_status
        self._statuses[self.settings.service.name] = overall_status
        self._statuses["vmc.lid.v1.LIDService"] = overall_status

        return {
            "ready": is_ready,
            "serving_status": "SERVING" if is_ready else "NOT_SERVING",
            "service": self.settings.service.name,
            "version": self.settings.service.version,
            "model_id": self.runtime.metadata.model_id if self.runtime else "unknown",
            "model_version": self.runtime.metadata.version if self.runtime else "unknown",
            "artifact_exists": artifact_exists,
            "runtime_initialized": self.runtime.is_loaded if self.runtime else False,
            "diagnostics": runtime_health,
        }

    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext
    ) -> health_pb2.HealthCheckResponse:
        self.check_readiness()
        service_name = request.service
        status = self._statuses.get(service_name, health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)
        return health_pb2.HealthCheckResponse(status=status)

    def Watch(self, request: health_pb2.HealthCheckRequest, context: grpc.ServicerContext):
        self.check_readiness()
        service_name = request.service
        status = self._statuses.get(service_name, health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)
        yield health_pb2.HealthCheckResponse(status=status)
