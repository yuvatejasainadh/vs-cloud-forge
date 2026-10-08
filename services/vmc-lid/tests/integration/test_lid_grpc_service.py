import os, sys
import time
import numpy as np
import pytest
import grpc
from concurrent import futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen")))

from config.settings import Settings
from service.lid_service import LIDService
from vmc.lid.v1 import lid_pb2, lid_pb2_grpc


class TestLIDGrpcServiceIntegration:
    @pytest.fixture(scope="class")
    def grpc_channel(self):
        settings = Settings.load()
        service = LIDService(settings)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        lid_pb2_grpc.add_LIDServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()

        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        yield channel

        server.stop(0)
        channel.close()

    def test_identify_language_rpc(self, grpc_channel):
        stub = lid_pb2_grpc.LIDServiceStub(grpc_channel)

        sr = 16000
        num_samples = sr
        t = np.linspace(0, 1.0, num_samples, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t)
        int16_bytes = (audio * 32767).astype(np.int16).tobytes()

        req = lid_pb2.IdentifyLanguageRequest(
            audio_content=int16_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            request_id="grpc_test_req_1"
        )

        resp = stub.IdentifyLanguage(req)
        assert resp.request_id == "grpc_test_req_1"
        assert resp.model_id == "open-vakgyata"
        assert resp.language != ""
        assert resp.language_name != ""
        assert resp.language_group in ["SOUTH", "NORTH", "EAST", "NORTH_WEST_FRONTIER", "WEST", "ENGLISH"]
        assert resp.target_vm.startswith("VM")
        assert resp.status in [lid_pb2.LID_STATUS_SUCCESS, lid_pb2.LID_STATUS_LOW_CONFIDENCE]
        assert len(resp.probabilities) == 10

    def test_get_model_info_rpc(self, grpc_channel):
        stub = lid_pb2_grpc.LIDServiceStub(grpc_channel)
        req = lid_pb2.ModelInfoRequest(model_id="open-vakgyata")
        resp = stub.GetModelInfo(req)

        assert resp.model_id == "open-vakgyata"
        assert resp.framework == "onnx"
        assert resp.parameters == 58746794
        assert len(resp.supported_languages) == 10

    def test_get_supported_languages_rpc(self, grpc_channel):
        stub = lid_pb2_grpc.LIDServiceStub(grpc_channel)
        req = lid_pb2.GetSupportedLanguagesRequest()
        resp = stub.GetSupportedLanguages(req)

        assert resp.total_canonical == 19
        assert resp.total_active == 10
        assert len(resp.canonical_languages) == 19
        assert len(resp.active_model_languages) == 10

    def test_resolve_language_group_rpc(self, grpc_channel):
        stub = lid_pb2_grpc.LIDServiceStub(grpc_channel)
        req = lid_pb2.ResolveLanguageGroupRequest(language="te-IN")
        resp = stub.ResolveLanguageGroup(req)

        assert resp.language == "te-IN"
        assert resp.language_name == "Telugu"
        assert resp.group == "SOUTH"
        assert resp.target_vm == "VM2_SOUTH"
        assert resp.target_service == "vmc-south"
        assert resp.target_port == 50052
        assert resp.is_canonical is True
