import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from telemetry.logging import get_logger

logger = get_logger("health_server")


class HealthHTTPRequestHandler(BaseHTTPRequestHandler):
    readiness_check_func: Optional[Callable[[], bool]] = None

    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"UP","service":"vmc-english","vm":"VM7_ENGLISH"}')
            return

        if self.path in ["/ready", "/readiness"]:
            is_ready = True
            if self.readiness_check_func is not None:
                try:
                    is_ready = self.readiness_check_func()
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
        # Suppress noisy standard HTTP logs
        pass


class HTTPHealthServer:
    def __init__(self, port: int = 9090, readiness_check: Optional[Callable[[], bool]] = None):
        self.port = port
        self.readiness_check = readiness_check
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler_cls = HealthHTTPRequestHandler
        handler_cls.readiness_check_func = self.readiness_check

        self.server = HTTPServer(("0.0.0.0", self.port), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"HTTP Health Server listening on port {self.port}")

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("HTTP Health Server stopped")
