import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from config.settings import Settings


class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.service.name == "vmc-english"
        assert settings.service.target_vm == "VM7_ENGLISH"
        assert settings.service.group == "ENGLISH"
        assert settings.service.language == "en-IN"
        assert settings.server.grpc_port == 50057
        assert settings.server.metrics_port == 9090
        assert settings.runtime.execution_provider == "CPUExecutionProvider"
        assert settings.detector.model_id == "voiceshield-aasist-english"
        assert settings.detector.threshold.value == 0.50

    def test_env_var_overrides(self):
        os.environ["CF_GRPC_PORT"] = "59057"
        os.environ["AASIST_THRESHOLD"] = "0.75"
        os.environ["AASIST_EXECUTION_PROVIDER"] = "CUDAExecutionProvider"
        try:
            settings = Settings.load()
            assert settings.server.grpc_port == 59057
            assert settings.detector.threshold.value == 0.75
            assert settings.runtime.execution_provider == "CUDAExecutionProvider"
        finally:
            os.environ.pop("CF_GRPC_PORT", None)
            os.environ.pop("AASIST_THRESHOLD", None)
            os.environ.pop("AASIST_EXECUTION_PROVIDER", None)
