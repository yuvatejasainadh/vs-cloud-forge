import json
import logging
import os
import sys
import time
from typing import Any, Dict


class SafeJsonFormatter(logging.Formatter):
    """
    Standard CloudForge JSON Log Formatter.
    Strictly redacts raw audio bytes, passwords, and Azure credentials.
    """

    REDACTED_KEYS = {
        "audio", "audio_content", "audio_bytes", "raw_audio", "payload",
        "api_key", "azure_key", "password", "secret", "token", "authorization"
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "vmc-english",
            "vm": "VM7_ENGLISH"
        }

        # Merge extra attributes if provided
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in ["msg", "args", "asctime"]:
                key_lower = key.lower()
                if any(redacted in key_lower for redacted in self.REDACTED_KEYS):
                    log_entry[key] = "[REDACTED_AUDIO_OR_SECRET]"
                else:
                    log_entry[key] = val

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SafeJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
