import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from core.runtime import AASISTLRuntime, ModelUnavailableError
from core.runtime_factory import AASISTLRuntimeFactory
from core.artifact_resolver import ArtifactResolver, LocalFilesystemSource
from config.settings import Settings


class TestAASISTLRuntime:
    def test_runtime_unhydrated_artifact_state(self):
        source = LocalFilesystemSource(base_dirs=["nonexistent/dir"])
        resolver = ArtifactResolver(source)

        runtime = AASISTLRuntime(
            model_id="voiceshield-aasist-english",
            version="v1.0.0",
            resolver=resolver
        )
        runtime.load()

        assert runtime.is_ready() is False
        health = runtime.health()
        assert health["status"] == "AWAITING_ARTIFACT"
        assert health["loaded"] is False

        with pytest.raises(ModelUnavailableError):
            runtime.predict(np.zeros(16000, dtype=np.float32))

    def test_runtime_factory_creation(self):
        settings = Settings()
        settings.detector.model_id = "voiceshield-aasist-english"
        settings.runtime.execution_provider = "CPUExecutionProvider"

        runtime = AASISTLRuntimeFactory.create(settings)
        assert runtime.model_id == "voiceshield-aasist-english"
        assert runtime.execution_provider == "CPUExecutionProvider"
        assert runtime.is_ready() is False
