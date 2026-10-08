import os
import sys
import time
import urllib.request
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from server.metrics_server import MetricsServer
from telemetry.metrics import get_deepfake_metrics


class TestHealthAndMetricsIntegration:
    @pytest.fixture(scope="class")
    def servers(self):
        metrics = get_deepfake_metrics()
        metrics.record_inference(
            model_id="voiceshield-aasist-english",
            model_version="v1.0",
            vm="VM7_ENGLISH",
            language="en-IN",
            status="GENUINE",
            provider="test",
            duration_sec=0.05,
            is_spoof=False
        )

        metrics_srv = MetricsServer(port=0, readiness_check=lambda: True)
        metrics_srv.start()
        port = metrics_srv.server.server_port
        time.sleep(0.1)

        yield port

        metrics_srv.stop()

    def test_http_health_endpoint(self, servers):
        port = servers
        url = f"http://127.0.0.1:{port}/healthz"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "UP"
            assert data["vm"] == "VM7_ENGLISH"

    def test_http_readiness_endpoint(self, servers):
        port = servers
        url = f"http://127.0.0.1:{port}/ready"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "READY"

    def test_http_prometheus_metrics_endpoint(self, servers):
        port = servers
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "deepfake_inference_total" in content
            assert "voiceshield-aasist-english" in content
            assert "VM7_ENGLISH" in content
