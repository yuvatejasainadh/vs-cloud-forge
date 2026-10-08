from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Optional


class DeepfakeMetrics:
    """
    Standard CloudForge VM7 Prometheus Metrics Collector for Deepfake Spoof Detection.
    """

    def __init__(self):
        self.inference_total = Counter(
            "deepfake_inference_total",
            "Total number of deepfake spoof detection inference requests",
            ["model_id", "model_version", "vm", "language", "status", "provider"]
        )

        self.inference_latency = Histogram(
            "deepfake_inference_latency_seconds",
            "Latency of deepfake spoof detection inference requests in seconds",
            ["model_id", "vm", "provider"],
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)
        )

        self.inference_errors = Counter(
            "deepfake_inference_errors_total",
            "Total number of inference errors in deepfake detector",
            ["error_type", "provider"]
        )

        self.real_total = Counter(
            "deepfake_real_total",
            "Total count of genuine/bonafide speech detections",
            ["model_id", "vm"]
        )

        self.spoof_total = Counter(
            "deepfake_spoof_total",
            "Total count of spoof/fake speech detections",
            ["model_id", "vm"]
        )

        self.low_confidence_total = Counter(
            "deepfake_low_confidence_total",
            "Total count of low confidence / uncertain deepfake classifications",
            ["model_id", "vm"]
        )

        self.model_load_time = Gauge(
            "deepfake_model_load_time_seconds",
            "Time taken to load or initialize the deepfake detector model",
            ["model_id", "provider"]
        )

        self.model_health = Gauge(
            "deepfake_model_health",
            "Health status of the deepfake detector (1 = healthy, 0 = unhealthy)",
            ["model_id", "provider"]
        )

    def record_inference(
        self,
        model_id: str,
        model_version: str,
        vm: str,
        language: str,
        status: str,
        provider: str,
        duration_sec: float,
        is_spoof: Optional[bool] = None,
        is_low_confidence: bool = False
    ) -> None:
        self.inference_total.labels(
            model_id=model_id,
            model_version=model_version,
            vm=vm,
            language=language,
            status=status,
            provider=provider
        ).inc()

        self.inference_latency.labels(
            model_id=model_id,
            vm=vm,
            provider=provider
        ).observe(duration_sec)

        if is_spoof is True:
            self.spoof_total.labels(model_id=model_id, vm=vm).inc()
        elif is_spoof is False:
            self.real_total.labels(model_id=model_id, vm=vm).inc()

        if is_low_confidence:
            self.low_confidence_total.labels(model_id=model_id, vm=vm).inc()

    def record_error(self, error_type: str, provider: str) -> None:
        self.inference_errors.labels(error_type=error_type, provider=provider).inc()

    def set_model_health(self, model_id: str, provider: str, healthy: bool) -> None:
        self.model_health.labels(model_id=model_id, provider=provider).set(1 if healthy else 0)

    def record_model_load_time(self, model_id: str, provider: str, duration_sec: float) -> None:
        self.model_load_time.labels(model_id=model_id, provider=provider).set(duration_sec)


# Global metrics instance
_metrics_instance: Optional[DeepfakeMetrics] = None


def get_deepfake_metrics() -> DeepfakeMetrics:
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = DeepfakeMetrics()
    return _metrics_instance
