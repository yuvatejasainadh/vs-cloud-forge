import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from telemetry.logging import get_logger

logger = get_logger("metrics_server")


def create_metrics_and_health_handler(readiness_check_fn: Optional[Callable[[], bool]] = None):
    class MetricsAndHealthHTTPHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                try:
                    metrics_data = generate_latest()
                    self.send_response(200)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(metrics_data)
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"Error generating metrics: {str(e)}".encode("utf-8"))
                return

            if self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"UP","service":"vmc-english","vm":"VM7_ENGLISH"}')
                return

            if self.path in ["/ready", "/readiness"]:
                is_ready = True
                if readiness_check_fn is not None:
                    try:
                        is_ready = bool(readiness_check_fn())
                    except Exception:
                        is_ready = False

                if is_ready:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"READY","service":"vmc-english","vm":"VM7_ENGLISH"}')
                else:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"NOT_READY","service":"vmc-english","vm":"VM7_ENGLISH","reason":"Model or provider not ready"}')
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    return MetricsAndHealthHTTPHandler


class MetricsServer:
    def __init__(self, port: int = 9090, readiness_check: Optional[Callable[[], bool]] = None):
        self.port = port
        self.readiness_check = readiness_check
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler_cls = create_metrics_and_health_handler(self.readiness_check)
        self.server = HTTPServer(("0.0.0.0", self.port), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Prometheus Metrics & Health Server listening on port {self.port}")

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Metrics server stopped")
