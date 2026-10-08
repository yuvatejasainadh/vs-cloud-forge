import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort

from core.model_registry import ModelMetadata, ModelRole, ModelTask, ModelStatus
from core.model_runtime import ModelRuntime, AudioInferenceInput, LIDInferenceResult
from core.audio_processor import AudioProcessor, AudioProcessingError
from core.classification_head import (
    LIDClassificationHead,
    NoClassificationHead,
    ClassificationHeadNotAvailableError,
)
from core.artifact_resolver import ArtifactResolver, ArtifactNotFoundError
from config.settings import Settings
from telemetry.logging import get_logger

logger = get_logger("xlsr_runtime")


class MeanPooling:
    """
    Explicit masked mean pooling layer for speech embeddings.
    Aggregates frame-level XLS-R hidden states into a single fixed-dimensional utterance representation.
    """

    def __init__(self, strategy: str = "masked_mean"):
        self.strategy = strategy

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Pools temporal frames into utterance embeddings.
        Args:
            hidden_states: Tensor of shape (batch, time, hidden_dim) or (time, hidden_dim).
            attention_mask: Optional mask of shape (batch, time) or (time,).
        Returns:
            Pooled embeddings of shape (batch, hidden_dim).
        """
        if hidden_states.ndim == 2:
            hidden_states = np.expand_dims(hidden_states, axis=0)
        elif hidden_states.ndim != 3:
            raise ValueError(f"Expected 2D or 3D hidden_states array, got shape {hidden_states.shape}")

        if attention_mask is not None:
            if attention_mask.ndim == 1:
                attention_mask = np.expand_dims(attention_mask, axis=0)

            time_hidden = hidden_states.shape[1]
            time_mask = attention_mask.shape[1]

            if time_mask != time_hidden and time_hidden > 0:
                indices = np.linspace(0, time_mask - 1, time_hidden).astype(int)
                subsampled_mask = attention_mask[:, indices]
            else:
                subsampled_mask = attention_mask

            expanded_mask = np.expand_dims(subsampled_mask, axis=-1).astype(np.float32)
            sum_embeddings = np.sum(hidden_states * expanded_mask, axis=1)
            sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)
            return (sum_embeddings / sum_mask).astype(np.float32)
        else:
            return np.mean(hidden_states, axis=1).astype(np.float32)

    def __call__(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.forward(hidden_states, attention_mask)


class XLSRRuntime(ModelRuntime):
    """
    CloudForge Native Runtime for facebook/wav2vec2-xls-r-300m Multilingual Speech Backbone.
    Decoupled from downstream classification heads to preserve clean encoder semantics.
    Supports Mode A (Encoder Evaluation) and Mode B (Trained 19-Language LID Head).
    """

    def __init__(
        self,
        metadata: ModelMetadata,
        settings: Settings,
        classification_head: Optional[LIDClassificationHead] = None,
        resolver: Optional[ArtifactResolver] = None,
        pooling: Optional[MeanPooling] = None,
    ):
        self._metadata = metadata
        self._settings = settings
        self._classification_head = classification_head or NoClassificationHead()
        self._resolver = resolver or ArtifactResolver()
        self._pooling = pooling or MeanPooling(strategy=getattr(settings, "pooling_strategy", "masked_mean"))
        self._session: Optional[ort.InferenceSession] = None
        self._is_loaded = False
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._artifact_path: Optional[str] = metadata.artifact_path
        self._device: str = "cpu"
        self._hidden_size: int = 1024
        self._parameter_count: int = metadata.parameters or 317374720

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._metadata.model_id

    @property
    def model_version(self) -> str:
        return self._metadata.version

    @property
    def revision(self) -> str:
        return self._metadata.revision or "1a640f32ac3e39899438a2931f9924c02f080a54"

    @property
    def runtime(self) -> str:
        return self._metadata.runtime or "transformers-pytorch"

    @property
    def framework(self) -> str:
        return self._metadata.framework or "transformers-pytorch"

    @property
    def device(self) -> str:
        return self._device

    @property
    def sample_rate(self) -> int:
        return self._metadata.sample_rate or 16000

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @property
    def artifact_status(self) -> str:
        if self._is_loaded:
            return "READY"
        elif self._artifact_path and os.path.exists(self._artifact_path):
            return "HYDRATED"
        return "AWAITING_ARTIFACT"

    @property
    def pooling(self) -> MeanPooling:
        return self._pooling

    @property
    def classification_head(self) -> LIDClassificationHead:
        return self._classification_head

    @classification_head.setter
    def classification_head(self, head: LIDClassificationHead) -> None:
        self._classification_head = head

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def load(self) -> None:
        """
        Loads the XLS-R backbone artifact into the runtime session.
        """
        if self._is_loaded and self._session is not None:
            return

        # Attempt resolving artifact if not already located
        if not self._artifact_path or not os.path.exists(self._artifact_path):
            try:
                self._artifact_path = self._resolver.resolve_and_verify(
                    model_id=self._metadata.model_id,
                    version=self._metadata.version,
                    filename="model.onnx",
                    expected_sha256=self._metadata.sha256
                )
            except Exception as e:
                logger.warning(
                    f"XLS-R backbone artifact not found or unhydrated for '{self._metadata.model_id}': {e}"
                )
                self._is_loaded = False
                return

        start_time = time.time()
        logger.info(
            f"XLSR_LOAD_STARTED: {self._metadata.model_id} (version: {self._metadata.version})",
            extra={
                "event": "XLSR_LOAD_STARTED",
                "model_id": self._metadata.model_id,
                "version": self._metadata.version,
            }
        )

        try:
            if self._artifact_path.endswith(".onnx"):
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = self._settings.model.num_threads
                sess_options.inter_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                providers = self._settings.model.execution_providers or ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    self._artifact_path,
                    sess_options=sess_options,
                    providers=providers
                )

                self._input_names = [inp.name for inp in self._session.get_inputs()]
                self._output_names = [out.name for out in self._session.get_outputs()]
                self._is_loaded = True
                self._device = "cuda" if any("CUDA" in p for p in self._session.get_providers()) else "cpu"
            else:
                # Native PyTorch / HuggingFace weights
                logger.info(
                    f"XLS-R backbone artifact is PyTorch format ({os.path.basename(self._artifact_path)}). "
                    f"Ready for PyTorch/Transformers runtime."
                )
                self._is_loaded = False

            load_duration = time.time() - start_time
            logger.info(
                f"XLSR_LOAD_COMPLETED: {self._metadata.model_id} in {load_duration:.3f}s",
                extra={
                    "event": "XLSR_LOAD_COMPLETED",
                    "model_id": self._metadata.model_id,
                    "version": self._metadata.version,
                    "duration_ms": load_duration * 1000.0,
                }
            )
        except Exception as e:
            self._is_loaded = False
            self._session = None
            logger.warning(
                f"XLSR_LOAD_NOTICE: {self._metadata.model_id} could not load active session: {str(e)}",
                extra={
                    "event": "XLSR_LOAD_NOTICE",
                    "model_id": self._metadata.model_id,
                    "error": str(e),
                }
            )

    def unload(self) -> None:
        """Unloads session from memory and frees resources."""
        self._session = None
        self._is_loaded = False
        logger.info(f"Unloaded XLS-R backbone {self._metadata.model_id}")

    # -------------------------------------------------------------------------
    # Audio Preprocessing & Inference
    # -------------------------------------------------------------------------

    def preprocess(self, audio: Union[AudioInferenceInput, np.ndarray, bytes]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates, mono-converts, resamples to 16kHz, and normalizes raw audio.
        Supports 8kHz, 16kHz, 22.05kHz, 44.1kHz, 48kHz.
        Returns (input_values [1, T], attention_mask [1, T]).
        """
        if isinstance(audio, np.ndarray):
            waveform = audio
        elif isinstance(audio, AudioInferenceInput):
            waveform = AudioProcessor.process_raw_audio(
                audio_bytes=audio.audio_data,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                audio_format=audio.audio_format,
                target_sample_rate=self.sample_rate,
                do_normalize=True
            )
        elif isinstance(audio, bytes):
            waveform = AudioProcessor.process_raw_audio(
                audio_bytes=audio,
                sample_rate=16000,
                channels=1,
                audio_format="pcm_s16le",
                target_sample_rate=16000,
                do_normalize=True
            )
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio)}")

        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)

        num_samples = len(waveform)
        input_values = np.expand_dims(waveform, axis=0).astype(np.float32)
        attention_mask = np.ones((1, num_samples), dtype=np.int32)
        return input_values, attention_mask

    def encode(
        self,
        audio: Union[AudioInferenceInput, np.ndarray, bytes],
        pool: bool = False
    ) -> np.ndarray:
        """
        Executes XLS-R encoder forward pass.
        Args:
            audio: Raw audio input.
            pool: If True, returns pooled utterance embedding (1, hidden_dim).
                  If False, returns full frame-level hidden states (1, time, hidden_dim).
        Returns:
            Speech representation array.
        """
        if not self._is_loaded or self._session is None:
            self.load()
            if not self._is_loaded or self._session is None:
                raise ArtifactNotFoundError(
                    f"XLS-R backbone artifact is unhydrated or unavailable for model '{self._metadata.model_id}'."
                )

        input_values, attention_mask = self.preprocess(audio)

        feed_dict = {}
        if "input_values" in self._input_names:
            feed_dict["input_values"] = input_values
        else:
            feed_dict[self._input_names[0]] = input_values

        if "attention_mask" in self._input_names:
            feed_dict["attention_mask"] = attention_mask

        outputs = self._session.run(None, feed_dict)
        hidden_states = outputs[0]

        if pool:
            return self._pooling.forward(hidden_states, attention_mask)
        return hidden_states

    def infer(self, input_data: AudioInferenceInput) -> LIDInferenceResult:
        """
        Executes end-to-end inference if classification head is present,
        or returns explicit MODEL_HEAD_NOT_AVAILABLE status if backbone only.
        """
        start_time = time.perf_counter()

        # If head is not available, do not run fake prediction
        if not self._classification_head.is_available:
            backbone_status = "AWAITING_ARTIFACT"
            if self._is_loaded and self._session is not None:
                try:
                    _ = self.encode(input_data)
                    backbone_status = "AVAILABLE"
                except Exception as e:
                    logger.warning(f"Backbone encoding check failed: {e}")
                    backbone_status = "ERROR"
            elif self._artifact_path and os.path.exists(self._artifact_path):
                backbone_status = "AVAILABLE"

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return LIDInferenceResult(
                language="UNKNOWN",
                confidence=0.0,
                probabilities={},
                model_id=self._metadata.model_id,
                model_version=self._metadata.version,
                inference_duration_ms=duration_ms,
                extra={
                    "status": "MODEL_HEAD_NOT_AVAILABLE",
                    "backbone_status": backbone_status,
                    "role": self._metadata.role,
                    "note": "XLS-R 300M backbone loaded without trained 19-language classification head"
                }
            )

        # Head is available: run encoder, pool, and head predict
        hidden_states = self.encode(input_data)
        pooled = self._pooling.forward(hidden_states)
        top_lang, confidence, probabilities = self._classification_head.predict(pooled)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return LIDInferenceResult(
            language=top_lang,
            confidence=confidence,
            probabilities=probabilities,
            model_id=self._metadata.model_id,
            model_version=self._metadata.version,
            inference_duration_ms=duration_ms,
            extra={
                "status": "SUCCESS",
                "role": self._metadata.role
            }
        )

    def health(self) -> Dict[str, Any]:
        """
        Performs diagnostic check of XLS-R backbone and classification head status.
        """
        artifact_exists = bool(
            self._artifact_path and os.path.exists(self._artifact_path)
        )
        is_ready = self._is_loaded and self._session is not None and artifact_exists

        return {
            "status": "SERVING" if is_ready else ("AWAITING_ARTIFACT" if not artifact_exists else "NOT_SERVING"),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "revision": self.revision,
            "runtime": self.runtime,
            "framework": self.framework,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "hidden_size": self.hidden_size,
            "parameter_count": self.parameter_count,
            "parameters": self._metadata.parameters or 317374720,
            "role": self._metadata.role,
            "task": self._metadata.task,
            "architecture": self._metadata.architecture,
            "artifact_exists": artifact_exists,
            "artifact_status": self.artifact_status,
            "artifact_path": self._artifact_path,
            "session_initialized": self._is_loaded,
            "execution_providers": self._session.get_providers() if self._session else [],
            "classification_head_available": self._classification_head.is_available,
            "input_names": self._input_names,
            "output_names": self._output_names,
            "pooling_strategy": self._pooling.strategy,
        }
