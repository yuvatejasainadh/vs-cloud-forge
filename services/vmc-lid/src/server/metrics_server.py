import http.server
import json
import threading
from typing import Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

from config.settings import Settings
from server.health_server import HealthServicer
from telemetry.logging import get_logger

logger = get_logger("metrics_server")


class MetricsHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    health_servicer: Optional[HealthServicer] = None
    registry: Optional[CollectorRegistry] = None

    def do_GET(self):
        if self.path == "/metrics":
            reg = self.registry if self.registry is not None else REGISTRY
            output = generate_latest(reg)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
            return

        if self.path in ["/health", "/healthz", "/ready", "/status"]:
            if self.health_servicer is not None:
                res = self.health_servicer.check_readiness()
                status_code = 200 if res.get("ready") else 503
            else:
                res = {"status": "SERVING"}
                status_code = 200

            body = json.dumps(res, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy standard http access logs
        pass


class MetricsServer:
    """
    CloudForge Telemetry & Health HTTP Exporter running on port 9090.
    """

    def __init__(
        self,
        settings: Settings,
        health_servicer: Optional[HealthServicer] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        self.settings = settings
        self.port = settings.server.metrics_port
        self.health_servicer = health_servicer
        self.registry = registry
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler_cls = MetricsHTTPRequestHandler
        handler_cls.health_servicer = self.health_servicer
        handler_cls.registry = self.registry
        self._server = http.server.HTTPServer(("0.0.0.0", self.port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Prometheus metrics HTTP server listening on http://0.0.0.0:{self.port}/metrics")

    def stop(self) -> None:
        if self._server:
            logger.info("Stopping metrics HTTP server...")
            self._server.shutdown()
            self._server.server_close()
            self._server = None
