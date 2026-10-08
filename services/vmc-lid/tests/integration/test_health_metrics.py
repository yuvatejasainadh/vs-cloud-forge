import os, sys
import urllib.request
import json
import pytest
import grpc
from concurrent import futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen")))

from config.settings import Settings
from service.lid_service import LIDService
from server.health_server import HealthServicer
from server.metrics_server import MetricsServer
from grpc_health.v1 import health_pb2, health_pb2_grpc


class TestHealthAndMetricsIntegration:
    @pytest.fixture(scope="class")
    def test_environment(self):
        settings = Settings.load()
        # Use an alternate port for test metrics server
        settings.server.metrics_port = 9099

        service = LIDService(settings)
        health_servicer = HealthServicer(settings, service.runtime)

        # Start HTTP metrics server
        metrics_server = MetricsServer(settings, health_servicer, registry=service.metrics.registry)
        metrics_server.start()

        # Start gRPC health server
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, grpc_server)
        grpc_port = grpc_server.add_insecure_port("127.0.0.1:0")
        grpc_server.start()

        channel = grpc.insecure_channel(f"127.0.0.1:{grpc_port}")

        yield {
            "channel": channel,
            "metrics_port": 9099,
            "health_servicer": health_servicer,
        }

        metrics_server.stop()
        grpc_server.stop(0)
        channel.close()

    def test_grpc_health_endpoint(self, test_environment):
        channel = test_environment["channel"]
        stub = health_pb2_grpc.HealthStub(channel)

        req = health_pb2.HealthCheckRequest(service="")
        resp = stub.Check(req)
        assert resp.status == health_pb2.HealthCheckResponse.SERVING

        req_service = health_pb2.HealthCheckRequest(service="vmc.lid.v1.LIDService")
        resp_service = stub.Check(req_service)
        assert resp_service.status == health_pb2.HealthCheckResponse.SERVING

    def test_http_health_readiness_endpoint(self, test_environment):
        port = test_environment["metrics_port"]
        url = f"http://127.0.0.1:{port}/healthz"
        with urllib.request.urlopen(url) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert data["ready"] is True
            assert data["serving_status"] == "SERVING"
            assert data["model_id"] == "open-vakgyata"
            assert data["artifact_exists"] is True
            assert data["runtime_initialized"] is True

    def test_http_prometheus_metrics_endpoint(self, test_environment):
        port = test_environment["metrics_port"]
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url) as response:
            assert response.status == 200
            content = response.read().decode("utf-8")
            assert "cloudforge_service_info" in content
            assert "cloudforge_service_health_status" in content
            assert "lid_inference_total" in content
            assert "cloudforge_ml_inference_duration_seconds" in content
