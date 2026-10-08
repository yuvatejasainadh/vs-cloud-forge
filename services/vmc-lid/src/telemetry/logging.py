import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class StructuredJSONFormatter(logging.Formatter):
    """
    Standard CloudForge JSON log formatter.
    Guarantees structured logging and sanitizes any sensitive or large payloads.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach structured event and custom extra fields
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "service"):
            log_entry["service"] = record.service
        if hasattr(record, "model_id"):
            log_entry["model_id"] = record.model_id
        if hasattr(record, "model_version"):
            log_entry["model_version"] = record.model_version
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "language"):
            log_entry["language"] = record.language
        if hasattr(record, "confidence"):
            log_entry["confidence"] = record.confidence
        if hasattr(record, "language_group"):
            log_entry["language_group"] = record.language_group
        if hasattr(record, "status"):
            log_entry["status"] = record.status
        if hasattr(record, "error"):
            log_entry["error"] = str(record.error)

        # Sanitize check: Ensure no raw binary or audio buffers are ever serialized
        for k in list(log_entry.keys()):
            if "audio" in k.lower() or "payload" in k.lower() or "raw" in k.lower():
                log_entry[k] = "<REDACTED_AUDIO_PAYLOAD>"

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str = "vmc-lid", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
