import pytest
import sys, os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen")))

from config.settings import Settings
from core.model_runtime import ModelRuntime, LIDInferenceResult, AudioInferenceInput
from core.model_registry import ModelMetadata
from service.lid_service import LIDService
from vmc.lid.v1 import lid_pb2


class DummyRuntime(ModelRuntime):
    def __init__(self, confidence_to_return: float):
        self._confidence = confidence_to_return
        self._metadata = ModelMetadata(
            model_id="test-model",
            model_name="Test Model",
            provider="test",
            source="test",
            repository="test",
            task="speech-language-identification",
            framework="onnx",
            version="v1.0.0",
            parameters=1000,
            artifact_size_mb=1.0,
            sample_rate=16000,
            channels=1,
            status="active",
            supported_languages=["hi-IN", "en-IN"]
        )

    def load(self): pass
    def health(self): return {"status": "SERVING"}
    def infer(self, input_data: AudioInferenceInput) -> LIDInferenceResult:
        return LIDInferenceResult(
            language="hi-IN",
            confidence=self._confidence,
            probabilities={"hi-IN": self._confidence, "en-IN": 1.0 - self._confidence},
            model_id="test-model",
            model_version="v1.0.0",
            inference_duration_ms=10.0
        )
    def unload(self): pass
    @property
    def metadata(self): return self._metadata
    @property
    def is_loaded(self): return True


class TestConfidenceHandler:
    def test_high_confidence_returns_success(self):
        settings = Settings.load()
        settings.model.confidence_threshold = 0.50

        runtime = DummyRuntime(confidence_to_return=0.85)
        service = LIDService(settings=settings, runtime=runtime)

        req = lid_pb2.IdentifyLanguageRequest(
            audio_content=b"\x00" * 32000,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            request_id="test_high"
        )
        context = MagicMock()
        resp = service.IdentifyLanguage(req, context)

        assert resp.status == lid_pb2.LID_STATUS_SUCCESS
        assert resp.language == "hi-IN"
        assert resp.confidence == pytest.approx(0.85, rel=1e-3)

    def test_low_confidence_returns_low_confidence_status(self):
        settings = Settings.load()
        settings.model.confidence_threshold = 0.50

        runtime = DummyRuntime(confidence_to_return=0.30)
        service = LIDService(settings=settings, runtime=runtime)

        req = lid_pb2.IdentifyLanguageRequest(
            audio_content=b"\x00" * 32000,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            request_id="test_low"
        )
        context = MagicMock()
        resp = service.IdentifyLanguage(req, context)

        assert resp.status == lid_pb2.LID_STATUS_LOW_CONFIDENCE
        assert resp.language == "hi-IN"
        assert resp.confidence == pytest.approx(0.30, rel=1e-3)

    def test_empty_audio_returns_invalid_audio_status(self):
        settings = Settings.load()
        runtime = DummyRuntime(confidence_to_return=0.90)
        service = LIDService(settings=settings, runtime=runtime)

        req = lid_pb2.IdentifyLanguageRequest(
            audio_content=b"",
            sample_rate=16000,
            request_id="test_empty"
        )
        context = MagicMock()
        resp = service.IdentifyLanguage(req, context)

        assert resp.status == lid_pb2.LID_STATUS_INVALID_AUDIO
