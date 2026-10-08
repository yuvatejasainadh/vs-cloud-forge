import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

from core.artifact_resolver import ArtifactResolver, ArtifactNotFoundError, ChecksumMismatchError
from telemetry.logging import get_logger

logger = get_logger("aasist_runtime")


class ModelUnavailableError(Exception):
    """Raised when the requested model artifact is not available, not hydrated, or unverified."""
    pass


class DeepfakeInferenceError(Exception):
    """Raised when deepfake inference computation fails."""
    pass


@dataclass
class AASISTLPrediction:
    spoof_score: float
    bonafide_score: float
    confidence: float
    is_spoof: bool
    inference_duration_ms: float
    raw_logits: Optional[List[float]] = None
    model_id: str = "voiceshield-aasist-english"
    model_version: str = "v1.0.0"
    execution_provider: str = "CPUExecutionProvider"


class AASISTLRuntime:
    """
    CloudForge-native AASIST-L Deepfake Detection Runtime.
    Loads and runs local ONNX/PyTorch model artifacts verified via the CloudForge Artifact Resolver.
    """

    def __init__(
        self,
        model_id: str = "voiceshield-aasist-english",
        version: str = "v1.0.0",
        artifact_filename: str = "model.onnx",
        expected_sha256: Optional[str] = None,
        threshold: float = 0.50,
        execution_provider: str = "CPUExecutionProvider",
        resolver: Optional[ArtifactResolver] = None
    ):
        self.model_id = model_id
        self.version = version
        self.artifact_filename = artifact_filename
        self.expected_sha256 = expected_sha256
        self.threshold = threshold
        self.execution_provider = execution_provider
        self.resolver = resolver or ArtifactResolver()

        self._session = None
        self._loaded = False
        self._artifact_path: Optional[str] = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        """Resolves artifact, verifies checksum, and initializes inference session."""
        logger.info(f"MODEL_LOAD_STARTED: {self.model_id} (version: {self.version})")
        start_time = time.perf_counter()

        try:
            self._artifact_path = self.resolver.resolve_and_verify(
                model_id=self.model_id,
                version=self.version,
                filename=self.artifact_filename,
                expected_sha256=self.expected_sha256
            )
        except (ArtifactNotFoundError, ChecksumMismatchError) as e:
            logger.warning(
                f"MODEL_LOAD_SKIPPED: {self.model_id} artifact not available or unverified: {str(e)}"
            )
            self._loaded = False
            return

        try:
            import onnxruntime as ort

            # Configure ONNX execution provider
            available_providers = ort.get_available_providers()
            chosen_provider = (
                self.execution_provider
                if self.execution_provider in available_providers
                else "CPUExecutionProvider"
            )

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                self._artifact_path,
                sess_options=sess_options,
                providers=[chosen_provider]
            )

            self._input_name = self._session.get_inputs()[0].name
            self._loaded = True
            self.execution_provider = chosen_provider

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"MODEL_LOAD_COMPLETED: {self.model_id} loaded successfully from {self._artifact_path} "
                f"using {chosen_provider} in {duration_ms:.2f}ms"
            )
        except Exception as e:
            logger.error(f"MODEL_LOAD_FAILED: {self.model_id} runtime initialization error: {str(e)}")
            self._loaded = False
            self._session = None

    def predict(self, waveform: np.ndarray) -> AASISTLPrediction:
        """Executes deepfake spoof inference on normalized 16kHz audio waveform."""
        if not self._loaded or self._session is None:
            raise ModelUnavailableError(
                f"AASIST-L model runtime '{self.model_id}' is not loaded or artifact is missing. "
                "Ensure verified model artifact is placed in model-cache."
            )

        start_time = time.perf_counter()

        try:
            # Shape tensor: (1, num_samples)
            if waveform.ndim == 1:
                input_tensor = np.expand_dims(waveform, axis=0).astype(np.float32)
            else:
                input_tensor = waveform.astype(np.float32)

            outputs = self._session.run(None, {self._input_name: input_tensor})
            logits = outputs[0][0]  # Expected [bonafide_logit, spoof_logit]

            # Softmax calculation: index 0 = spoof, index 1 = bonafide
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            spoof_prob = float(probs[0])
            bonafide_prob = float(probs[1])

            is_spoof = spoof_prob >= self.threshold
            confidence = float(max(spoof_prob, bonafide_prob))
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return AASISTLPrediction(
                spoof_score=spoof_prob,
                bonafide_score=bonafide_prob,
                confidence=confidence,
                is_spoof=is_spoof,
                inference_duration_ms=duration_ms,
                raw_logits=[float(logits[0]), float(logits[1])],
                model_id=self.model_id,
                model_version=self.version,
                execution_provider=self.execution_provider
            )
        except Exception as e:
            if isinstance(e, ModelUnavailableError):
                raise
            raise DeepfakeInferenceError(f"AASIST-L inference execution failed: {str(e)}")

    @property
    def model_version(self) -> str:
        return self.version

    @property
    def architecture(self) -> str:
        return "AASIST-L"

    @property
    def runtime(self) -> str:
        return "onnx"

    @property
    def artifact_status(self) -> str:
        return "AVAILABLE" if self._loaded else "NOT_AVAILABLE"

    @property
    def device(self) -> str:
        return self.execution_provider

    @property
    def input_shape(self) -> List[Any]:
        return ["batch", 64600]

    @property
    def output_shape(self) -> List[Any]:
        return ["batch", 2]

    def health(self) -> Dict[str, Any]:
        """Reports runtime health and readiness status."""
        return {
            "model_id": self.model_id,
            "model_version": self.version,
            "version": self.version,
            "architecture": self.architecture,
            "runtime": self.runtime,
            "artifact_status": self.artifact_status,
            "device": self.device,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "loaded": self._loaded,
            "execution_provider": self.execution_provider,
            "artifact_path": self._artifact_path,
            "status": "SERVING" if self._loaded else "AWAITING_ARTIFACT"
        }

    def is_ready(self) -> bool:
        """Returns True if the verified model artifact is loaded and serving."""
        return self._loaded and self._session is not None

    def unload(self) -> None:
        """Releases session resources."""
        self._session = None
        self._loaded = False
        logger.info(f"MODEL_UNLOADED: {self.model_id}")
