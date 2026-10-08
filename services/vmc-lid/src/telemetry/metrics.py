from typing import Optional
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY


class LIDMetrics:
    """
    Prometheus telemetry collector for CloudForge VMC-LID service.
    Exposes metrics on HTTP port 9090 (/metrics).
    """

    _global_instance: Optional["LIDMetrics"] = None

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry if registry is not None else CollectorRegistry()

        # Service metadata gauge
        self.service_info = Gauge(
            "cloudforge_service_info",
            "Static CloudForge service build metadata",
            ["service", "version", "platform", "environment"],
            registry=self.registry
        )

        # Service health status (1 = SERVING, 0 = NOT_SERVING)
        self.service_health_status = Gauge(
            "cloudforge_service_health_status",
            "Real-time service health status (1 = SERVING, 0 = NOT_SERVING)",
            ["service"],
            registry=self.registry
        )

        # Total inference requests
        self.lid_inference_total = Counter(
            "lid_inference_total",
            "Total number of LID inference requests processed",
            ["model_id", "model_version", "status", "language"],
            registry=self.registry
        )

        # Inference latency histogram (platform standard)
        self.cloudforge_ml_inference_duration_seconds = Histogram(
            "cloudforge_ml_inference_duration_seconds",
            "Distribution of ML inference latency in seconds",
            ["service", "model", "version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )

        # Dedicated LID latency histogram
        self.lid_inference_latency = Histogram(
            "lid_inference_latency",
            "LID inference duration in seconds",
            ["model_id", "model_version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )

        # Inference errors counter
        self.lid_inference_errors = Counter(
            "lid_inference_errors",
            "Total number of LID inference errors",
            ["model_id", "error_type"],
            registry=self.registry
        )

        # Low confidence counter
        self.lid_low_confidence_total = Counter(
            "lid_low_confidence_total",
            "Total number of LID inferences falling below the confidence threshold",
            ["model_id", "model_version"],
            registry=self.registry
        )

        # Model load time gauge
        self.lid_model_load_time = Gauge(
            "lid_model_load_time_seconds",
            "Time taken to initialize and load the model into memory in seconds",
            ["model_id", "model_version"],
            registry=self.registry
        )

        # Model load failures counter
        self.lid_model_load_failures = Counter(
            "lid_model_load_failures_total",
            "Total number of model load failures",
            ["model_id"],
            registry=self.registry
        )

    def set_service_info(self, service: str, version: str, platform: str, environment: str):
        self.service_info.labels(
            service=service,
            version=version,
            platform=platform,
            environment=environment
        ).set(1.0)

    def set_serving_status(self, service: str, serving: bool):
        self.service_health_status.labels(service=service).set(1.0 if serving else 0.0)

    def record_model_load(self, model_id: str, version: str, duration_sec: float, success: bool):
        if success:
            self.lid_model_load_time.labels(model_id=model_id, model_version=version).set(duration_sec)
        else:
            self.lid_model_load_failures.labels(model_id=model_id).inc()

    def record_inference(
        self,
        service: str,
        model_id: str,
        version: str,
        duration_sec: float,
        status: str,
        language: str = "UNKNOWN",
        is_low_confidence: bool = False
    ):
        self.lid_inference_total.labels(
            model_id=model_id,
            model_version=version,
            status=status,
            language=language
        ).inc()

        self.cloudforge_ml_inference_duration_seconds.labels(
            service=service,
            model=model_id,
            version=version
        ).observe(duration_sec)

        self.lid_inference_latency.labels(
            model_id=model_id,
            model_version=version
        ).observe(duration_sec)

        if is_low_confidence:
            self.lid_low_confidence_total.labels(
                model_id=model_id,
                model_version=version
            ).inc()

    def record_error(self, model_id: str, error_type: str):
        self.lid_inference_errors.labels(model_id=model_id, error_type=error_type).inc()
