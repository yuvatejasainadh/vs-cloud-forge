import os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from core.runtime_factory import ModelRuntimeFactory
from core.model_runtime import ModelRuntime
from core.model_registry import ModelMetadata
from config.settings import Settings
import runtimes


class TestRuntimeFactory:
    def test_factory_creates_open_vakgyata_runtime(self):
        settings = Settings.load()
        meta = ModelMetadata(
            model_id="open-vakgyata",
            model_name="Open-Vakgyata",
            provider="onecxi",
            source="huggingface",
            repository="onecxi/open-vakgyata",
            task="speech-language-identification",
            framework="onnx",
            version="f2754058e485dfc65cc62589b8d0e21c1d328399",
            parameters=58746794,
            artifact_size_mb=224.19,
            sample_rate=16000,
            channels=1,
            status="active",
            supported_languages=["hi-IN", "en-IN"]
        )

        runtime = ModelRuntimeFactory.create(meta, settings)
        assert isinstance(runtime, ModelRuntime)
        assert runtime.metadata.model_id == "open-vakgyata"

    def test_unregistered_framework_raises_valueerror(self):
        settings = Settings.load()
        meta = ModelMetadata(
            model_id="custom-model",
            model_name="Custom Model",
            provider="test",
            source="test",
            repository="test",
            task="speech-language-identification",
            framework="unregistered-framework-123",
            version="v1",
            parameters=100,
            artifact_size_mb=1.0,
            sample_rate=16000,
            channels=1,
            status="active",
            supported_languages=["en-IN"]
        )

        with pytest.raises(ValueError, match="No runtime registered"):
            ModelRuntimeFactory.create(meta, settings)
