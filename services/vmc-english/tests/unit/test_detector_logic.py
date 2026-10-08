import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from config.settings import Settings
from core.detector import EnglishDeepfakeDetector
from core.runtime import AASISTLRuntime, AASISTLPrediction, ModelUnavailableError, DeepfakeInferenceError


class MockRuntime(AASISTLRuntime):
    def __init__(self, spoof_score: float = 0.1, confidence: float = 0.9):
        super().__init__(model_id="voiceshield-aasist-english", version="v1.0.0")
        self._spoof_score = spoof_score
        self._confidence = confidence
        self._loaded = True

    def load(self): pass

    def predict(self, waveform: np.ndarray) -> AASISTLPrediction:
        return AASISTLPrediction(
            spoof_score=self._spoof_score,
            bonafide_score=1.0 - self._spoof_score,
            confidence=self._confidence,
            is_spoof=self._spoof_score >= 0.50,
            inference_duration_ms=15.0,
            model_id="voiceshield-aasist-english",
            model_version="v1.0.0",
            execution_provider="CPUExecutionProvider"
        )

    def health(self):
        return {"status": "SERVING", "model_id": self.model_id, "loaded": True}

    def is_ready(self):
        return self._loaded


class TestDetectorLogic:
    def test_language_gating_accepts_english(self):
        settings = Settings()
        runtime = MockRuntime(spoof_score=0.1, confidence=0.9)
        detector = EnglishDeepfakeDetector(settings, runtime)

        audio_bytes = b"\x00" * 32000
        result = detector.analyze(audio_bytes, language="en-IN", language_group="ENGLISH")
        assert result.status == EnglishDeepfakeDetector.STATUS_GENUINE
        assert result.is_spoof is False
        assert result.target_vm == "VM7_ENGLISH"

    def test_language_gating_rejects_non_english(self):
        settings = Settings()
        runtime = MockRuntime()
        detector = EnglishDeepfakeDetector(settings, runtime)

        audio_bytes = b"\x00" * 32000
        result = detector.analyze(audio_bytes, language="hi-IN", language_group="NORTH")
        assert result.status == EnglishDeepfakeDetector.STATUS_UNSUPPORTED_LANGUAGE

    def test_spoof_detection_decision(self):
        settings = Settings()
        runtime = MockRuntime(spoof_score=0.92, confidence=0.95)
        detector = EnglishDeepfakeDetector(settings, runtime)

        audio_bytes = b"\x00" * 32000
        result = detector.analyze(audio_bytes, language="en-IN")
        assert result.status == EnglishDeepfakeDetector.STATUS_SPOOF
        assert result.is_spoof is True
        assert result.spoof_score == pytest.approx(0.92)

    def test_low_confidence_decision(self):
        settings = Settings()
        runtime = MockRuntime(spoof_score=0.52, confidence=0.52)
        detector = EnglishDeepfakeDetector(settings, runtime)

        audio_bytes = b"\x00" * 32000
        result = detector.analyze(audio_bytes, language="en-IN")
        assert result.status == EnglishDeepfakeDetector.STATUS_LOW_CONFIDENCE

    def test_model_unavailable_error_handling(self):
        settings = Settings()
        runtime = MockRuntime()
        runtime.predict = MagicMock(side_effect=ModelUnavailableError("Artifact missing"))
        detector = EnglishDeepfakeDetector(settings, runtime)

        audio_bytes = b"\x00" * 32000
        result = detector.analyze(audio_bytes, language="en-IN")
        assert result.status == EnglishDeepfakeDetector.STATUS_MODEL_UNAVAILABLE

    def test_runtime_inference_error_handling(self):
        settings = Settings()
        runtime = MockRuntime()
        runtime.predict = MagicMock(side_effect=DeepfakeInferenceError("Computation crash"))
        detector = EnglishDeepfakeDetector(settings, runtime)

        audio_bytes = b"\x00" * 32000
        result = detector.analyze(audio_bytes, language="en-IN")
        assert result.status == EnglishDeepfakeDetector.STATUS_INFERENCE_ERROR
