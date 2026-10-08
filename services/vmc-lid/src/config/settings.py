import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class ServiceSettings:
    name: str = "vmc-lid"
    version: str = "v2.0.0"
    platform: str = "cloudforge"
    environment: str = "development"


@dataclass
class ServerSettings:
    grpc_port: int = 50051
    metrics_port: int = 9090
    metrics_enabled: bool = True
    max_recv_msg_size_mb: int = 32
    max_send_msg_size_mb: int = 32
    graceful_timeout_seconds: int = 5


@dataclass
class ModelSettings:
    registry_path: str = r"C:\CloudForge\models\registry.json"
    models_dir: str = r"C:\CloudForge\models"
    active_model_id: str = "open-vakgyata"
    confidence_threshold: float = 0.50
    accept_threshold: float = 0.80
    execution_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    num_threads: int = 4


@dataclass
class RoutingSettings:
    canonical_registry_file: str = r"C:\CloudForge\services\vmc-lid\configs\canonical_registry.yaml"
    language_groups_file: str = r"C:\CloudForge\services\vmc-lid\configs\canonical_registry.yaml"
    lacr_checkpoint_interval: int = 5


@dataclass
class Settings:
    service: ServiceSettings = field(default_factory=ServiceSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    routing: RoutingSettings = field(default_factory=RoutingSettings)

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "Settings":
        cfg = cls()

        # Load from config.yaml if provided or exists
        candidate_paths = [
            config_file,
            os.getenv("CLOUDFORGE_CONFIG_PATH"),
            r"C:\CloudForge\services\vmc-lid\configs\config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml"),
        ]

        loaded_yaml = {}
        for path in candidate_paths:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    loaded_yaml = yaml.safe_load(f) or {}
                break

        # Map YAML fields
        if "service" in loaded_yaml:
            s_dict = loaded_yaml["service"]
            cfg.service.name = s_dict.get("name", cfg.service.name)
            cfg.service.version = s_dict.get("version", cfg.service.version)
            cfg.service.platform = s_dict.get("platform", cfg.service.platform)
            cfg.service.environment = s_dict.get("environment", cfg.service.environment)

        if "server" in loaded_yaml:
            srv_dict = loaded_yaml["server"]
            cfg.server.grpc_port = int(srv_dict.get("grpc_port", cfg.server.grpc_port))
            cfg.server.metrics_port = int(srv_dict.get("metrics_port", cfg.server.metrics_port))
            cfg.server.metrics_enabled = bool(srv_dict.get("metrics_enabled", cfg.server.metrics_enabled))
            cfg.server.max_recv_msg_size_mb = int(srv_dict.get("max_recv_msg_size_mb", cfg.server.max_recv_msg_size_mb))
            cfg.server.max_send_msg_size_mb = int(srv_dict.get("max_send_msg_size_mb", cfg.server.max_send_msg_size_mb))
            cfg.server.graceful_timeout_seconds = int(srv_dict.get("graceful_timeout_seconds", cfg.server.graceful_timeout_seconds))

        if "model" in loaded_yaml:
            m_dict = loaded_yaml["model"]
            cfg.model.registry_path = m_dict.get("registry_path", cfg.model.registry_path)
            cfg.model.models_dir = m_dict.get("models_dir", cfg.model.models_dir)
            cfg.model.active_model_id = m_dict.get("active_model_id", cfg.model.active_model_id)
            cfg.model.confidence_threshold = float(m_dict.get("confidence_threshold", cfg.model.confidence_threshold))
            cfg.model.accept_threshold = float(m_dict.get("accept_threshold", cfg.model.accept_threshold))
            cfg.model.execution_providers = m_dict.get("execution_providers", cfg.model.execution_providers)
            cfg.model.num_threads = int(m_dict.get("num_threads", cfg.model.num_threads))

        if "routing" in loaded_yaml:
            r_dict = loaded_yaml["routing"]
            cfg.routing.canonical_registry_file = r_dict.get("canonical_registry_file", cfg.routing.canonical_registry_file)
            cfg.routing.language_groups_file = r_dict.get("language_groups_file", cfg.routing.language_groups_file)
            cfg.routing.lacr_checkpoint_interval = int(r_dict.get("lacr_checkpoint_interval", cfg.routing.lacr_checkpoint_interval))

        # Environment variable overrides
        if "CLOUDFORGE_SERVICE_NAME" in os.environ:
            cfg.service.name = os.environ["CLOUDFORGE_SERVICE_NAME"]
        if "CLOUDFORGE_ENVIRONMENT" in os.environ:
            cfg.service.environment = os.environ["CLOUDFORGE_ENVIRONMENT"]
        if "CF_GRPC_PORT" in os.environ:
            cfg.server.grpc_port = int(os.environ["CF_GRPC_PORT"])
        if "CF_METRICS_PORT" in os.environ:
            cfg.server.metrics_port = int(os.environ["CF_METRICS_PORT"])
        if "LID_MODEL_ID" in os.environ:
            cfg.model.active_model_id = os.environ["LID_MODEL_ID"]
        if "LID_MODELS_DIR" in os.environ:
            cfg.model.models_dir = os.environ["LID_MODELS_DIR"]
        if "LID_CONFIDENCE_THRESHOLD" in os.environ:
            cfg.model.confidence_threshold = float(os.environ["LID_CONFIDENCE_THRESHOLD"])
        if "LID_CANONICAL_REGISTRY_FILE" in os.environ:
            cfg.routing.canonical_registry_file = os.environ["LID_CANONICAL_REGISTRY_FILE"]
            cfg.routing.language_groups_file = os.environ["LID_CANONICAL_REGISTRY_FILE"]

        return cfg
