import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from core.model_runtime import ModelRuntime, LIDInferenceResult, AudioInferenceInput, InferenceResult
from routing.canonical_registry import CanonicalRegistry, get_canonical_registry
from routing.lacr import LACRManager, get_lacr_manager
from telemetry.logging import get_logger

logger = get_logger("decision_engine")


@dataclass
class DecisionOutput:
    language: str
    language_name: str
    confidence: float
    language_group: str
    target_vm: str
    status: int  # maps to LIDStatus enum in proto
    model_id: str
    model_version: str
    inference_duration_ms: float
    probabilities: Dict[str, float]
    is_switched: bool = False


class LIDDecisionEngine:
    """
    Orchestrates LID model inference, multi-model ensemble/fallback, confidence thresholding,
    canonical group resolution, and continuous session tracking (LACR).
    """

    # Protobuf LIDStatus enum values
    STATUS_UNSPECIFIED = 0
    STATUS_SUCCESS = 1
    STATUS_LOW_CONFIDENCE = 2
    STATUS_UNKNOWN = 3
    STATUS_INVALID_AUDIO = 4
    STATUS_INFERENCE_ERROR = 5
    STATUS_UNSUPPORTED = 6
    STATUS_MODEL_NOT_AVAILABLE = 7
    STATUS_MODEL_HEAD_NOT_AVAILABLE = 8

    def __init__(
        self,
        runtime: ModelRuntime,
        registry: Optional[CanonicalRegistry] = None,
        lacr_mgr: Optional[LACRManager] = None,
        accept_threshold: float = 0.80,
        low_confidence_threshold: float = 0.50
    ):
        self.runtime = runtime
        self.registry = registry or get_canonical_registry()
        self.lacr = lacr_mgr or get_lacr_manager()
        self.accept_threshold = accept_threshold
        self.low_confidence_threshold = low_confidence_threshold

    def evaluate(
        self,
        waveform: np.ndarray,
        request_id: str = "",
        session_id: str = "",
        chunk_index: int = 1
    ) -> DecisionOutput:
        """
        Executes LID inference on audio waveform and maps to canonical 7-VM topology.
        """
        start_time = time.perf_counter()

        # Handle inference call
        if hasattr(self.runtime, "infer"):
            # Convert float32 numpy waveform to int16 PCM bytes for standard infer contract
            pcm_data = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            inp = AudioInferenceInput(
                audio_data=pcm_data,
                sample_rate=16000,
                channels=1,
                audio_format="pcm_s16le",
                request_id=request_id
            )
            inf_result = self.runtime.infer(inp)
        elif hasattr(self.runtime, "predict"):
            inf_result = self.runtime.predict(waveform)
        else:
            raise NotImplementedError("Runtime does not implement infer() or predict()")

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        extra_status = inf_result.extra.get("status") if hasattr(inf_result, "extra") else None
        raw_lang = inf_result.language
        confidence = inf_result.confidence
        probabilities = inf_result.probabilities

        if extra_status == "MODEL_HEAD_NOT_AVAILABLE":
            status = self.STATUS_MODEL_HEAD_NOT_AVAILABLE
            group = self.registry.default_group
            target_vm = self.registry.default_vm
            lang_name = "Unknown"
        elif extra_status == "MODEL_NOT_AVAILABLE":
            status = self.STATUS_MODEL_NOT_AVAILABLE
            group = self.registry.default_group
            target_vm = self.registry.default_vm
            lang_name = "Unknown"
        else:
            # Map to canonical registry
            lang_info = self.registry.get_language_info(raw_lang)
            if lang_info is None:
                # Predicted language is not in 19 canonical languages
                group = self.registry.default_group
                target_vm = self.registry.default_vm
                lang_name = raw_lang
                status = self.STATUS_UNSUPPORTED
            else:
                group = lang_info.group
                target_vm = lang_info.vm
                lang_name = lang_info.name

                # Determine confidence status
                if confidence >= self.accept_threshold:
                    status = self.STATUS_SUCCESS
                elif confidence >= self.low_confidence_threshold:
                    status = self.STATUS_LOW_CONFIDENCE
                else:
                    status = self.STATUS_UNKNOWN

        # Update LACR session state if session_id is provided
        effective_lang, effective_group, effective_vm, has_switched = self.lacr.update_session(
            session_id=session_id,
            chunk_index=chunk_index,
            predicted_language=raw_lang,
            predicted_group=group,
            predicted_vm=target_vm,
            confidence=confidence
        )

        model_id = self.runtime.metadata.model_id if hasattr(self.runtime, "metadata") else getattr(self.runtime, "model_id", "unknown")
        model_version = self.runtime.metadata.version if hasattr(self.runtime, "metadata") else getattr(self.runtime, "version", "default")

        return DecisionOutput(
            language=effective_lang,
            language_name=self.registry.get_language_name(effective_lang, lang_name),
            confidence=confidence,
            language_group=effective_group,
            target_vm=effective_vm,
            status=status,
            model_id=model_id,
            model_version=model_version,
            inference_duration_ms=inference_time_ms,
            probabilities=probabilities,
            is_switched=has_switched,
        )
