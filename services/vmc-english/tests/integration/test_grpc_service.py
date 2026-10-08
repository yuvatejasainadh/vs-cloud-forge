import os
import sys
from concurrent import futures
import numpy as np
import pytest
import grpc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen")))

from config.settings import Settings
from service.detector_service import EnglishDetectorService
from core.runtime import AASISTLRuntime, AASISTLPrediction
from vmc.detector.v1 import detector_pb2, detector_pb2_grpc


class MockTestRuntime(AASISTLRuntime):
    def __init__(self, spoof_score: float = 0.08):
        super().__init__(model_id="voiceshield-aasist-english", version="v1.0.0")
        self._spoof_score = spoof_score
        self._loaded = True

    def load(self): pass

    def predict(self, waveform: np.ndarray) -> AASISTLPrediction:
        return AASISTLPrediction(
            spoof_score=self._spoof_score,
            bonafide_score=1.0 - self._spoof_score,
            confidence=0.92,
            is_spoof=self._spoof_score >= 0.50,
            inference_duration_ms=12.5,
            model_id="voiceshield-aasist-english",
            model_version="v1.0.0",
            execution_provider="CPUExecutionProvider"
        )

    def health(self):
        return {"status": "SERVING", "model_id": self.model_id, "loaded": True}

    def is_ready(self): return True


class TestEnglishGrpcServiceIntegration:
    @pytest.fixture(scope="class")
    def grpc_channel(self):
        settings = Settings()
        runtime = MockTestRuntime(spoof_score=0.08)
        service = EnglishDetectorService(settings=settings, runtime=runtime)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        detector_pb2_grpc.add_DetectorServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()

        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        yield channel

        server.stop(0)
        channel.close()

    def test_detect_spoof_rpc(self, grpc_channel):
        stub = detector_pb2_grpc.DetectorServiceStub(grpc_channel)

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        int16_bytes = (tone * 32767).astype(np.int16).tobytes()

        req = detector_pb2.DetectSpoofRequest(
            audio_content=int16_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            language="en-IN",
            language_group="ENGLISH",
            request_id="grpc_vm7_test_1"
        )

        resp = stub.DetectSpoof(req)
        assert resp.request_id == "grpc_vm7_test_1"
        assert resp.target_vm == "VM7_ENGLISH"
        assert resp.language_group == "ENGLISH"
        assert resp.language == "en-IN"
        assert resp.status == detector_pb2.DETECTION_STATUS_GENUINE
        assert resp.spoof_score == pytest.approx(0.08, rel=1e-2)
        assert resp.bonafide_score == pytest.approx(0.92, rel=1e-2)
        assert resp.model_id == "voiceshield-aasist-english"

    def test_detect_spoof_language_gating_rpc(self, grpc_channel):
        stub = detector_pb2_grpc.DetectorServiceStub(grpc_channel)

        audio_bytes = b"\x00" * 32000
        req = detector_pb2.DetectSpoofRequest(
            audio_content=audio_bytes,
            sample_rate=16000,
            language="te-IN",
            language_group="SOUTH",
            request_id="grpc_vm7_gating_test"
        )

        resp = stub.DetectSpoof(req)
        assert resp.status == detector_pb2.DETECTION_STATUS_UNSUPPORTED_LANGUAGE

    def test_get_detector_info_rpc(self, grpc_channel):
        stub = detector_pb2_grpc.DetectorServiceStub(grpc_channel)
        req = detector_pb2.DetectorInfoRequest(model_id="voiceshield-aasist-english")
        resp = stub.GetDetectorInfo(req)

        assert resp.model_id == "voiceshield-aasist-english"
        assert resp.target_vm == "VM7_ENGLISH"
        assert resp.group == "ENGLISH"
        assert resp.model_type == "AASIST-L"
        assert "en-IN" in resp.supported_languages
        assert resp.provider == "cloudforge"
        assert resp.framework == "onnx"
        assert resp.status == "SERVING"
