import json
import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import onnxruntime as ort

from core.model_registry import ModelMetadata
from core.model_runtime import ModelRuntime, AudioInferenceInput, LIDInferenceResult
from core.audio_processor import AudioProcessor, AudioProcessingError
from config.settings import Settings
from telemetry.logging import get_logger

logger = get_logger("open_vakgyata_runtime")


class OpenVakgyataRuntime(ModelRuntime):
    """
    ONNX Runtime adapter for the locked Open-Vakgyata (onecxi/open-vakgyata) LID model.
    Implements ModelRuntime generic interface.
    """

    DEFAULT_ID2LABEL = {
        0: "en-IN",
        1: "hi-IN",
        2: "or-IN",
        3: "bn-IN",
        4: "ta-IN",
        5: "te-IN",
        6: "kn-IN",
        7: "ml-IN",
        8: "mr-IN",
        9: "gu-IN"
    }

    def __init__(self, metadata: ModelMetadata, settings: Settings):
        self._metadata = metadata
        self._settings = settings
        self._session: Optional[ort.InferenceSession] = None
        self._id2label: Dict[int, str] = dict(self.DEFAULT_ID2LABEL)
        self._label2id: Dict[str, int] = {v: k for k, v in self._id2label.items()}
        self._is_loaded = False
        self._input_names = []
        self._output_names = []
        self._load_config_labels()

    def _load_config_labels(self) -> None:
        """Loads label mappings from config.json if present."""
        if self._metadata.config_path and os.path.exists(self._metadata.config_path):
            try:
                with open(self._metadata.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "id2label" in cfg:
                        self._id2label = {int(k): str(v) for k, v in cfg["id2label"].items()}
                        self._label2id = {v: k for k, v in self._id2label.items()}
            except Exception as e:
                logger.warning(f"Failed to parse id2label from config.json: {e}")

    def load(self) -> None:
        """Loads ONNX model artifact and configures ONNX Runtime InferenceSession."""
        if self._is_loaded and self._session is not None:
            return

        if not self._metadata.artifact_path or not os.path.exists(self._metadata.artifact_path):
            raise FileNotFoundError(
                f"Model artifact not found for '{self._metadata.model_id}' at {self._metadata.artifact_path}"
            )

        start_time = time.time()
        logger.info(
            f"MODEL_LOAD_STARTED: {self._metadata.model_id} (version: {self._metadata.version})",
            extra={
                "event": "MODEL_LOAD_STARTED",
                "model_id": self._metadata.model_id,
                "model_version": self._metadata.version,
            }
        )

        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self._settings.model.num_threads
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = self._settings.model.execution_providers or ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                self._metadata.artifact_path,
                sess_options=sess_options,
                providers=providers
            )

            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            self._is_loaded = True

            load_duration = time.time() - start_time
            logger.info(
                f"MODEL_LOAD_COMPLETED: {self._metadata.model_id} in {load_duration:.3f}s",
                extra={
                    "event": "MODEL_LOAD_COMPLETED",
                    "model_id": self._metadata.model_id,
                    "model_version": self._metadata.version,
                    "duration_ms": load_duration * 1000.0,
                }
            )
        except Exception as e:
            self._is_loaded = False
            logger.error(
                f"MODEL_LOAD_FAILED: {self._metadata.model_id} error: {str(e)}",
                extra={
                    "event": "MODEL_LOAD_FAILED",
                    "model_id": self._metadata.model_id,
                    "model_version": self._metadata.version,
                    "error": str(e),
                }
            )
            raise

    def health(self) -> Dict[str, Any]:
        """Validates model artifact existence, runtime initialization, and session readiness."""
        artifact_exists = bool(
            self._metadata.artifact_path and os.path.exists(self._metadata.artifact_path)
        )
        is_ready = self._is_loaded and self._session is not None and artifact_exists

        status_info = {
            "status": "SERVING" if is_ready else "NOT_SERVING",
            "model_id": self._metadata.model_id,
            "model_version": self._metadata.version,
            "framework": self._metadata.framework,
            "artifact_exists": artifact_exists,
            "artifact_path": self._metadata.artifact_path,
            "session_initialized": self._is_loaded,
            "execution_providers": self._session.get_providers() if self._session else [],
            "input_names": self._input_names,
            "output_names": self._output_names,
        }
        return status_info

    def preprocess(self, input_data: AudioInferenceInput) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates, mono-converts, resamples to 16kHz, and normalizes raw audio.
        Returns input_values array shaped (1, num_samples) and attention_mask array (1, num_samples).
        """
        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=input_data.audio_data,
            sample_rate=input_data.sample_rate,
            channels=input_data.channels,
            audio_format=input_data.audio_format,
            target_sample_rate=self._metadata.sample_rate,
            do_normalize=True
        )

        num_samples = len(waveform)
        input_values = np.expand_dims(waveform, axis=0).astype(np.float32)
        attention_mask = np.ones((1, num_samples), dtype=np.int32)
        return input_values, attention_mask

    def infer(self, input_data: AudioInferenceInput) -> LIDInferenceResult:
        """Executes full inference lifecycle: preprocess -> ONNX infer -> postprocess."""
        if not self._is_loaded or self._session is None:
            self.load()

        start_time = time.perf_counter()

        # 1. Preprocess audio
        input_values, attention_mask = self.preprocess(input_data)

        # 2. Prepare feed dict
        feed_dict = {}
        if "input_values" in self._input_names:
            feed_dict["input_values"] = input_values
        else:
            feed_dict[self._input_names[0]] = input_values

        if "attention_mask" in self._input_names:
            feed_dict["attention_mask"] = attention_mask

        # 3. Execute ONNX inference session
        outputs = self._session.run(None, feed_dict)
        logits = outputs[0]  # Shape: (1, 10)

        # 4. Postprocess logits to probabilities and predicted language
        lang_code, confidence, probabilities = self.postprocess(logits)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return LIDInferenceResult(
            language=lang_code,
            confidence=confidence,
            probabilities=probabilities,
            model_id=self._metadata.model_id,
            model_version=self._metadata.version,
            inference_duration_ms=duration_ms
        )

    def postprocess(self, logits: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Applies softmax and formats probabilities dictionary across all 10 Indic languages."""
        if logits.ndim > 1:
            logits_1d = logits[0]
        else:
            logits_1d = logits

        # Numerically stable softmax
        shifted = logits_1d - np.max(logits_1d)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)

        probabilities: Dict[str, float] = {}
        for idx, prob in enumerate(probs):
            lang = self._id2label.get(idx, f"lang_{idx}")
            probabilities[lang] = float(prob)

        top_idx = int(np.argmax(probs))
        top_lang = self._id2label.get(top_idx, "UNKNOWN")
        confidence = float(probs[top_idx])

        return top_lang, confidence, probabilities

    def unload(self) -> None:
        """Unloads ONNX session from memory."""
        self._session = None
        self._is_loaded = False
        logger.info(f"Unloaded model {self._metadata.model_id}")

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
