from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.model_registry import ModelMetadata


@dataclass
class AudioInferenceInput:
    """Standardized inference input payload passed to ModelRuntime."""
    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    audio_format: str = "pcm_s16le"
    request_id: str = ""


@dataclass
class LIDInferenceResult:
    """Standardized inference output returned by ModelRuntime."""
    language: str
    confidence: float
    probabilities: Dict[str, float]
    model_id: str
    model_version: str
    inference_duration_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)


InferenceResult = LIDInferenceResult


class ModelRuntime(ABC):
    """
    CloudForge Generic Model Runtime Interface.
    All ML runtimes (ONNX, PyTorch, TensorRT, Custom) must implement this contract.
    """

    @abstractmethod
    def load(self) -> None:
        """Loads and initializes the underlying model artifact and execution backend."""
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """
        Performs a runtime readiness check.
        Returns a dictionary containing health status and diagnostic info.
        """
        pass

    @abstractmethod
    def infer(self, input_data: AudioInferenceInput) -> LIDInferenceResult:
        """
        Executes model inference on the provided audio input.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Frees model resources and unloads execution sessions from memory."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Returns the ModelMetadata associated with this runtime instance."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the runtime is loaded and ready for inference."""
        pass
