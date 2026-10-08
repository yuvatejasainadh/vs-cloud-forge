import time
import grpc
from typing import Optional

from config.settings import Settings
from core.detector import EnglishDeepfakeDetector, DetectionResult
from core.runtime import AASISTLRuntime
from core.runtime_factory import AASISTLRuntimeFactory
from core.artifact_resolver import ArtifactResolver
from telemetry.logging import get_logger
from telemetry.metrics import DeepfakeMetrics, get_deepfake_metrics

from vmc.detector.v1 import detector_pb2, detector_pb2_grpc

logger = get_logger("detector_service")


class EnglishDetectorService(detector_pb2_grpc.DetectorServiceServicer):
    """
    gRPC Servicer for VM7 English Deepfake Detection Service (vmc-english).
    Coordinates CloudForge AASISTLRuntime, EnglishDeepfakeDetector, and Prometheus Telemetry.
    """

    def __init__(
        self,
        settings: Settings,
        runtime: Optional[AASISTLRuntime] = None,
        detector: Optional[EnglishDeepfakeDetector] = None,
        resolver: Optional[ArtifactResolver] = None,
        metrics: Optional[DeepfakeMetrics] = None
    ):
        self.settings = settings
        self.metrics = metrics or get_deepfake_metrics()
        self.resolver = resolver or ArtifactResolver()
        self.runtime = runtime or AASISTLRuntimeFactory.create(settings, self.resolver)
        self.detector = detector or EnglishDeepfakeDetector(
            settings=settings,
            runtime=self.runtime,
            metrics=self.metrics
        )

        # Pre-load runtime
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        start_time = time.perf_counter()
        try:
            self.runtime.load()
            duration_sec = time.perf_counter() - start_time
            self.metrics.record_model_load_time(
                self.runtime.model_id,
                self.runtime.execution_provider,
                duration_sec
            )
            self.metrics.set_model_health(
                self.runtime.model_id,
                self.runtime.execution_provider,
                self.runtime.is_ready()
            )
        except Exception as e:
            logger.error(f"Failed to initialize AASIST-L runtime: {e}")
            self.metrics.set_model_health(
                self.runtime.model_id,
                self.runtime.execution_provider,
                False
            )

    def DetectSpoof(
        self,
        request: detector_pb2.DetectSpoofRequest,
        context: grpc.ServicerContext
    ) -> detector_pb2.DetectSpoofResponse:
        req_id = request.request_id or f"req_{int(time.time()*1000)}"

        logger.info(
            f"DEEPFAKE_REQUEST_RECEIVED: req_id={req_id} lang={request.language} group={request.language_group}",
            extra={
                "event": "DEEPFAKE_REQUEST_RECEIVED",
                "request_id": req_id,
                "language": request.language,
                "language_group": request.language_group,
                "session_id": request.session_id,
                "chunk_index": request.chunk_index
            }
        )

        result: DetectionResult = self.detector.analyze(
            audio_content=request.audio_content,
            sample_rate=request.sample_rate or 16000,
            channels=request.channels or 1,
            audio_format=request.audio_format or "pcm_s16le",
            language=request.language or "en-IN",
            language_group=request.language_group or "ENGLISH",
            request_id=req_id,
            session_id=request.session_id,
            chunk_index=request.chunk_index
        )

        return detector_pb2.DetectSpoofResponse(
            spoof_score=result.spoof_score,
            bonafide_score=result.bonafide_score,
            confidence=result.confidence,
            status=result.status,
            language_group=result.language_group,
            target_vm=result.target_vm,
            language=result.language,
            model_id=result.model_id,
            model_version=result.model_version,
            request_id=result.request_id,
            inference_duration_ms=result.inference_duration_ms
        )

    def GetDetectorInfo(
        self,
        request: detector_pb2.DetectorInfoRequest,
        context: grpc.ServicerContext
    ) -> detector_pb2.DetectorInfoResponse:
        health_info = self.runtime.health()
        return detector_pb2.DetectorInfoResponse(
            model_id=self.runtime.model_id,
            model_name=self.settings.detector.model_name,
            target_vm=self.settings.service.target_vm,
            group=self.settings.service.group,
            model_type=self.settings.detector.model_type,
            supported_languages=self.settings.detector.supported_languages,
            status=health_info.get("status", "UNKNOWN"),
            provider="cloudforge",
            framework="onnx"
        )
