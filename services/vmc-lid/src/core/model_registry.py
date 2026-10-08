import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib

from telemetry.logging import get_logger

logger = get_logger("model_registry")


class ModelRole(str, Enum):
    BACKBONE = "BACKBONE"
    CLASSIFIER = "CLASSIFIER"
    DETECTOR = "DETECTOR"
    END_TO_END = "END_TO_END"
    BENCHMARK = "BENCHMARK"


class ModelTask(str, Enum):
    SPEECH_ENCODER = "SPEECH_ENCODER"
    LID = "LID"
    DEEPFAKE_DETECTION = "DEEPFAKE_DETECTION"


class ModelStatus(str, Enum):
    ACTIVE = "ACTIVE"
    BENCHMARK = "BENCHMARK"
    BACKBONE = "BACKBONE"
    CANDIDATE = "CANDIDATE"
    REGISTERED_CANDIDATE = "REGISTERED_CANDIDATE"
    PLANNED = "PLANNED"
    BLOCKED = "BLOCKED"


@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    role: str = ModelRole.END_TO_END.value
    task: str = ModelTask.LID.value
    architecture: str = ""
    family: str = ""
    parameter_count: str = ""
    parameters: int = 0
    provider: str = "unknown"
    source: str = "unknown"
    repository: str = ""
    revision: Optional[str] = None
    runtime: str = "onnx"
    framework: str = "onnx"
    version: str = "default"
    artifact_size_mb: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    status: str = "active"
    supported_languages: List[str] = field(default_factory=list)
    language_labels: List[str] = field(default_factory=list)
    language_names: Dict[str, str] = field(default_factory=dict)
    artifact_dict: Dict[str, Any] = field(default_factory=dict)
    artifact_path: Optional[str] = None
    config_path: Optional[str] = None
    preprocessor_config_path: Optional[str] = None
    sha256: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_backbone(self) -> bool:
        return self.role.upper() == ModelRole.BACKBONE.value

    @property
    def is_active(self) -> bool:
        return self.status.upper() in ("ACTIVE", "SERVING")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "role": self.role,
            "task": self.task,
            "architecture": self.architecture,
            "family": self.family,
            "parameter_count": self.parameter_count,
            "parameters": self.parameters,
            "provider": self.provider,
            "source": self.source,
            "repository": self.repository,
            "revision": self.revision,
            "runtime": self.runtime,
            "framework": self.framework,
            "version": self.version,
            "artifact_size_mb": self.artifact_size_mb,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "status": self.status,
            "supported_languages": self.supported_languages,
            "language_labels": self.language_labels,
            "language_names": self.language_names,
            "artifact": self.artifact_dict,
            "artifact_path": self.artifact_path,
            "config_path": self.config_path,
            "preprocessor_config_path": self.preprocessor_config_path,
            "sha256": self.sha256,
        }


class ModelRegistry:
    """
    CloudForge Central Model Registry.
    Discovers, validates, and resolves model definitions, roles, and artifact paths.
    """

    def __init__(self, registry_path: str, models_base_dir: str):
        self.registry_path = registry_path
        self.models_base_dir = models_base_dir
        self._models: Dict[str, ModelMetadata] = {}
        self._default_model_id: Optional[str] = None
        self.refresh()

    def refresh(self) -> None:
        """Loads registry entries from registry.json and discovers local filesystem caches."""
        if not os.path.exists(self.registry_path):
            logger.warning(f"Registry file not found at {self.registry_path}")
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._default_model_id = data.get("default_model_id")
        raw_models = data.get("models", {})

        for model_id, info in raw_models.items():
            version = info.get("version") or info.get("revision") or "default"
            revision = info.get("revision")

            # Try finding artifact directory
            artifact_dict = info.get("artifact", {})
            artifact_dir = os.path.join(self.models_base_dir, model_id, version)

            # Check specific HF cache structure if present
            hf_cache = artifact_dict.get("cache_dir")
            if hf_cache:
                resolved_hf = os.path.abspath(os.path.join(self.models_base_dir, "..", hf_cache))
                if os.path.exists(resolved_hf):
                    artifact_dir = resolved_hf

            model_file = os.path.join(artifact_dir, "model.onnx")
            if not os.path.exists(model_file):
                # Fallback check for single file or pytorch_model
                candidate_files = ["model.onnx", "model_quantized.onnx", "pytorch_model.bin"]
                for cf in candidate_files:
                    test_f = os.path.join(artifact_dir, cf)
                    if os.path.exists(test_f):
                        model_file = test_f
                        break

            config_file = os.path.join(artifact_dir, "config.json")
            preproc_file = os.path.join(artifact_dir, "preprocessor_config.json")
            meta_file = os.path.join(artifact_dir, "metadata.json")

            sha256 = info.get("sha256") or artifact_dict.get("sha256")
            if not sha256 and os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                        sha256 = m_data.get("sha256")
                except Exception:
                    pass

            role = info.get("role", ModelRole.END_TO_END.value).upper()
            task = info.get("task", ModelTask.LID.value).upper()
            status = info.get("status", "ACTIVE").upper()

            meta = ModelMetadata(
                model_id=info.get("model_id", model_id),
                model_name=info.get("model_name", model_id),
                role=role,
                task=task,
                architecture=info.get("architecture", ""),
                family=info.get("family", ""),
                parameter_count=str(info.get("parameter_count", "")),
                parameters=int(info.get("parameters", 0)),
                provider=info.get("provider", "unknown"),
                source=info.get("source", "unknown"),
                repository=info.get("repository", ""),
                revision=revision,
                runtime=info.get("runtime", info.get("framework", "onnx")),
                framework=info.get("framework", info.get("runtime", "onnx")),
                version=version,
                artifact_size_mb=float(info.get("artifact_size_mb", 0.0)),
                sample_rate=int(info.get("sample_rate", 16000)),
                channels=int(info.get("channels", 1)),
                status=status,
                supported_languages=info.get("supported_languages", []),
                language_labels=info.get("language_labels", info.get("supported_languages", [])),
                language_names=info.get("language_names", {}),
                artifact_dict=artifact_dict,
                artifact_path=model_file if os.path.exists(model_file) else None,
                config_path=config_file if os.path.exists(config_file) else None,
                preprocessor_config_path=preproc_file if os.path.exists(preproc_file) else None,
                sha256=sha256,
            )
            self._models[model_id] = meta

    def get_model(self, model_id: Optional[str] = None) -> ModelMetadata:
        target_id = model_id or self._default_model_id
        if not target_id:
            raise ValueError("No model_id specified and no default_model_id set in registry")
        if target_id not in self._models:
            raise KeyError(f"Model '{target_id}' is not registered in CloudForge registry")
        return self._models[target_id]

    def list_models(self) -> List[ModelMetadata]:
        return list(self._models.values())

    def validate_artifact(self, model_id: str) -> bool:
        meta = self.get_model(model_id)
        if not meta.artifact_path or not os.path.exists(meta.artifact_path):
            return False
        if os.path.getsize(meta.artifact_path) == 0:
            return False
        return True
