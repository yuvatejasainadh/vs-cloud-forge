import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np

from config.settings import Settings
from core.audio_processor import EnglishAudioProcessor, AudioProcessingError
from core.runtime import AASISTLRuntime, AASISTLPrediction, ModelUnavailableError, DeepfakeInferenceError
from telemetry.logging import get_logger
from telemetry.metrics import DeepfakeMetrics, get_deepfake_metrics

logger = get_logger("detector_core")


@dataclass
class DetectionResult:
    spoof_score: float
    bonafide_score: float
    confidence: float
    status: int  # maps to DetectionStatus enum in detector.proto
    target_vm: str
    language_group: str
    language: str
    model_id: str
    model_version: str
    request_id: str
    inference_duration_ms: float
    is_spoof: Optional[bool] = None


class EnglishDeepfakeDetector:
    """
    CloudForge VM7 English Deepfake Detector Core.
    Coordinates language gating (en-IN), audio preprocessing, and CloudForge-native AASIST-L runtime execution.
    """

    # Protobuf DetectionStatus enum values
    STATUS_UNSPECIFIED = 0
    STATUS_GENUINE = 1
    STATUS_SPOOF = 2
    STATUS_UNCERTAIN = 3
    STATUS_INVALID_AUDIO = 4
    STATUS_INFERENCE_ERROR = 5
    STATUS_MODEL_UNAVAILABLE = 6
    STATUS_UNSUPPORTED_LANGUAGE = 7
    STATUS_LOW_CONFIDENCE = 8

    def __init__(
        self,
        settings: Settings,
        runtime: AASISTLRuntime,
        metrics: Optional[DeepfakeMetrics] = None
    ):
        self.settings = settings
        self.runtime = runtime
        self.metrics = metrics or get_deepfake_metrics()
        self.target_vm = settings.service.target_vm
        self.group = settings.service.group
        self.threshold = settings.detector.threshold.value
        self.supported_languages = set(settings.detector.supported_languages)

    def analyze(
        self,
        audio_content: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_format: str = "pcm_s16le",
        language: str = "en-IN",
        language_group: str = "ENGLISH",
        request_id: str = "",
        session_id: str = "",
        chunk_index: int = 1
    ) -> DetectionResult:
        start_time = time.perf_counter()
        req_id = request_id or f"req_{int(time.time()*1000)}"

        # 1. Language Gating: VM7 strictly accepts English (en-IN / ENGLISH)
        norm_lang = language.strip()
        if norm_lang and norm_lang not in self.supported_languages and norm_lang != "en":
            logger.warning(
                f"VM7 Language Gating Reject: received '{language}' (group: '{language_group}'). "
                f"VM7 only serves English detector requests.",
                extra={"event": "LANGUAGE_GATING_REJECT", "language": language, "request_id": req_id}
            )
            return DetectionResult(
                spoof_score=0.0,
                bonafide_score=0.0,
                confidence=0.0,
                status=self.STATUS_UNSUPPORTED_LANGUAGE,
                target_vm=self.target_vm,
                language_group=self.group,
                language=language,
                model_id=self.runtime.model_id,
                model_version=self.runtime.version,
                request_id=req_id,
                inference_duration_ms=0.0
            )

        # 2. Audio Validation and Preprocessing
        if not audio_content or len(audio_content) == 0:
            logger.warning(f"Audio payload empty for request_id={req_id}")
            self.metrics.record_error("EMPTY_AUDIO", self.runtime.execution_provider)
            return DetectionResult(
                spoof_score=0.0,
                bonafide_score=0.0,
                confidence=0.0,
                status=self.STATUS_INVALID_AUDIO,
                target_vm=self.target_vm,
                language_group=self.group,
                language=language or "en-IN",
                model_id=self.runtime.model_id,
                model_version=self.runtime.version,
                request_id=req_id,
                inference_duration_ms=0.0
            )

        try:
            waveform = EnglishAudioProcessor.process_raw_audio(
                audio_bytes=audio_content,
                sample_rate=sample_rate or 16000,
                channels=channels or 1,
                audio_format=audio_format or "pcm_s16le",
                target_sample_rate=16000,
                do_normalize=True,
                fix_length=True
            )
        except AudioProcessingError as ape:
            logger.warning(f"Audio processing failure for request_id={req_id}: {str(ape)}")
            self.metrics.record_error("AUDIO_PROCESSING_ERROR", self.runtime.execution_provider)
            return DetectionResult(
                spoof_score=0.0,
                bonafide_score=0.0,
                confidence=0.0,
                status=self.STATUS_INVALID_AUDIO,
                target_vm=self.target_vm,
                language_group=self.group,
                language=language or "en-IN",
                model_id=self.runtime.model_id,
                model_version=self.runtime.version,
                request_id=req_id,
                inference_duration_ms=0.0
            )

        # 3. CloudForge-Native Runtime Inference Execution
        try:
            prediction = self.runtime.predict(waveform)
        except ModelUnavailableError as mue:
            logger.error(f"Model unavailable for request_id={req_id}: {str(mue)}")
            self.metrics.record_error("MODEL_UNAVAILABLE", self.runtime.execution_provider)
            return DetectionResult(
                spoof_score=0.0,
                bonafide_score=0.0,
                confidence=0.0,
                status=self.STATUS_MODEL_UNAVAILABLE,
                target_vm=self.target_vm,
                language_group=self.group,
                language=language or "en-IN",
                model_id=self.runtime.model_id,
                model_version=self.runtime.version,
                request_id=req_id,
                inference_duration_ms=0.0
            )
        except Exception as e:
            logger.error(f"Inference computation error for request_id={req_id}: {str(e)}")
            self.metrics.record_error("INFERENCE_ERROR", self.runtime.execution_provider)
            return DetectionResult(
                spoof_score=0.0,
                bonafide_score=0.0,
                confidence=0.0,
                status=self.STATUS_INFERENCE_ERROR,
                target_vm=self.target_vm,
                language_group=self.group,
                language=language or "en-IN",
                model_id=self.runtime.model_id,
                model_version=self.runtime.version,
                request_id=req_id,
                inference_duration_ms=0.0
            )

        # 4. Decision Calibration & Classification
        spoof_score = prediction.spoof_score
        bonafide_score = prediction.bonafide_score
        confidence = prediction.confidence
        is_spoof = prediction.is_spoof

        if confidence < 0.60:
            status = self.STATUS_LOW_CONFIDENCE
        elif is_spoof:
            status = self.STATUS_SPOOF
        else:
            status = self.STATUS_GENUINE

        total_duration_sec = time.perf_counter() - start_time
        total_duration_ms = total_duration_sec * 1000.0

        # 5. Metrics Recording
        status_name = "SPOOF" if is_spoof else ("REAL" if status == self.STATUS_GENUINE else "LOW_CONFIDENCE")
        self.metrics.record_inference(
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            vm=self.target_vm,
            language=language or "en-IN",
            status=status_name,
            provider="cloudforge",
            duration_sec=total_duration_sec,
            is_spoof=is_spoof if status != self.STATUS_LOW_CONFIDENCE else None,
            is_low_confidence=(status == self.STATUS_LOW_CONFIDENCE)
        )

        logger.info(
            f"DEEPFAKE_INFERENCE_COMPLETED: decision={'SPOOF' if is_spoof else 'GENUINE'} "
            f"spoof_score={spoof_score:.3f} conf={confidence:.3f} ({total_duration_ms:.2f}ms)",
            extra={
                "event": "DEEPFAKE_INFERENCE_COMPLETED",
                "request_id": req_id,
                "session_id": session_id,
                "chunk_index": chunk_index,
                "spoof_score": spoof_score,
                "bonafide_score": bonafide_score,
                "confidence": confidence,
                "is_spoof": is_spoof,
                "provider": "cloudforge",
                "duration_ms": total_duration_ms
            }
        )

        return DetectionResult(
            spoof_score=spoof_score,
            bonafide_score=bonafide_score,
            confidence=confidence,
            status=status,
            target_vm=self.target_vm,
            language_group=self.group,
            language=language or "en-IN",
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            request_id=req_id,
            inference_duration_ms=prediction.inference_duration_ms,
            is_spoof=is_spoof
        )
