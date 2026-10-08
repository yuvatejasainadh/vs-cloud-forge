from typing import Dict, Type
from core.model_registry import ModelMetadata
from core.model_runtime import ModelRuntime
from config.settings import Settings
from telemetry.logging import get_logger

logger = get_logger("runtime_factory")


class ModelRuntimeFactory:
    """
    CloudForge Model Runtime Factory.
    Enables pluggable runtime instantiation (ONNX, PyTorch, TensorRT, Custom)
    based on model framework and task metadata without conditional branches in service logic.
    """

    _registry: Dict[str, Type[ModelRuntime]] = {}

    @classmethod
    def register(cls, framework: str, runtime_cls: Type[ModelRuntime]) -> None:
        """Registers a runtime class for a specific framework or custom model type."""
        key = framework.lower().strip()
        cls._registry[key] = runtime_cls
        logger.info(f"Registered ModelRuntime: framework='{key}' -> {runtime_cls.__name__}")

    @classmethod
    def create(cls, metadata: ModelMetadata, settings: Settings) -> ModelRuntime:
        """
        Creates and returns an uninitialized ModelRuntime instance matching the metadata framework.
        """
        framework_key = metadata.framework.lower().strip()
        
        # Check direct framework match (e.g. "onnx")
        if framework_key in cls._registry:
            runtime_cls = cls._registry[framework_key]
            logger.info(f"Instantiating runtime for model '{metadata.model_id}' using {runtime_cls.__name__}")
            return runtime_cls(metadata, settings)

        # Check model_id specific override if any
        model_id_key = metadata.model_id.lower().strip()
        if model_id_key in cls._registry:
            runtime_cls = cls._registry[model_id_key]
            logger.info(f"Instantiating custom runtime for '{metadata.model_id}' using {runtime_cls.__name__}")
            return runtime_cls(metadata, settings)

        raise ValueError(
            f"No runtime registered for framework '{metadata.framework}' or model_id '{metadata.model_id}'. "
            f"Available runtimes: {list(cls._registry.keys())}"
        )
