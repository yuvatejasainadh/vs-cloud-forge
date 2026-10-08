import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class ServiceSettings:
    name: str = "vmc-english"
    version: str = "v2.0.0"
    platform: str = "cloudforge"
    environment: str = "development"
    target_vm: str = "VM7_ENGLISH"
    group: str = "ENGLISH"
    language: str = "en-IN"


@dataclass
class ServerSettings:
    grpc_port: int = 50057
    metrics_port: int = 9090
    metrics_enabled: bool = True
    max_recv_msg_size_mb: int = 32
    max_send_msg_size_mb: int = 32
    graceful_timeout_seconds: int = 5


@dataclass
class RuntimeSettings:
    execution_provider: str = "CPUExecutionProvider"
    num_threads: int = 4


@dataclass
class ThresholdSettings:
    value: float = 0.50
    calibrated: bool = False


@dataclass
class DetectorSettings:
    model_id: str = "voiceshield-aasist-english"
    model_name: str = "VoiceShield AASIST-L English Detector"
    model_version: str = "v1.0.0"
    model_type: str = "AASIST-L"
    artifact_filename: str = "model.onnx"
    expected_sha256: Optional[str] = None
    model_cache_dir: str = r"C:\CloudForge\model-cache"
    supported_languages: List[str] = field(default_factory=lambda: ["en-IN"])
    threshold: ThresholdSettings = field(default_factory=ThresholdSettings)


@dataclass
class Settings:
    service: ServiceSettings = field(default_factory=ServiceSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "Settings":
        cfg = cls()

        candidate_paths = [
            config_file,
            os.getenv("CLOUDFORGE_CONFIG_PATH"),
            r"C:\CloudForge\services\vmc-english\configs\config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml"),
            "/app/configs/config.yaml",
        ]

        loaded_yaml = {}
        for path in candidate_paths:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    loaded_yaml = yaml.safe_load(f) or {}
                break

        if "service" in loaded_yaml:
            s_dict = loaded_yaml["service"]
            cfg.service.name = s_dict.get("name", cfg.service.name)
            cfg.service.version = s_dict.get("version", cfg.service.version)
            cfg.service.target_vm = s_dict.get("target_vm", cfg.service.target_vm)
            cfg.service.group = s_dict.get("group", cfg.service.group)
            cfg.service.language = s_dict.get("language", cfg.service.language)

        if "server" in loaded_yaml:
            srv_dict = loaded_yaml["server"]
            cfg.server.grpc_port = int(srv_dict.get("grpc_port", cfg.server.grpc_port))
            cfg.server.metrics_port = int(srv_dict.get("metrics_port", cfg.server.metrics_port))
            cfg.server.metrics_enabled = bool(srv_dict.get("metrics_enabled", cfg.server.metrics_enabled))

        if "runtime" in loaded_yaml:
            r_dict = loaded_yaml["runtime"]
            cfg.runtime.execution_provider = r_dict.get("execution_provider", cfg.runtime.execution_provider)
            cfg.runtime.num_threads = int(r_dict.get("num_threads", cfg.runtime.num_threads))

        if "detector" in loaded_yaml:
            d_dict = loaded_yaml["detector"]
            cfg.detector.model_id = d_dict.get("model_id", cfg.detector.model_id)
            cfg.detector.model_name = d_dict.get("model_name", cfg.detector.model_name)
            cfg.detector.model_version = d_dict.get("model_version", cfg.detector.model_version)
            cfg.detector.model_type = d_dict.get("model_type", cfg.detector.model_type)
            cfg.detector.artifact_filename = d_dict.get("artifact_filename", cfg.detector.artifact_filename)
            cfg.detector.expected_sha256 = d_dict.get("expected_sha256", cfg.detector.expected_sha256)
            cfg.detector.model_cache_dir = d_dict.get("model_cache_dir", cfg.detector.model_cache_dir)
            cfg.detector.supported_languages = d_dict.get("supported_languages", cfg.detector.supported_languages)

            if "threshold" in d_dict:
                t_dict = d_dict["threshold"]
                cfg.detector.threshold.value = float(t_dict.get("value", cfg.detector.threshold.value))
                cfg.detector.threshold.calibrated = bool(t_dict.get("calibrated", cfg.detector.threshold.calibrated))

        # Environment variable overrides
        if "CLOUDFORGE_SERVICE_NAME" in os.environ:
            cfg.service.name = os.environ["CLOUDFORGE_SERVICE_NAME"]
        if "CF_GRPC_PORT" in os.environ:
            cfg.server.grpc_port = int(os.environ["CF_GRPC_PORT"])
        if "CF_METRICS_PORT" in os.environ:
            cfg.server.metrics_port = int(os.environ["CF_METRICS_PORT"])
        if "AASIST_MODEL_ID" in os.environ:
            cfg.detector.model_id = os.environ["AASIST_MODEL_ID"]
        if "AASIST_MODEL_VERSION" in os.environ:
            cfg.detector.model_version = os.environ["AASIST_MODEL_VERSION"]
        if "AASIST_MODEL_PATH" in os.environ:
            cfg.detector.model_cache_dir = os.environ["AASIST_MODEL_PATH"]
        if "AASIST_THRESHOLD" in os.environ:
            cfg.detector.threshold.value = float(os.environ["AASIST_THRESHOLD"])
        if "AASIST_EXECUTION_PROVIDER" in os.environ:
            cfg.runtime.execution_provider = os.environ["AASIST_EXECUTION_PROVIDER"]

        return cfg
