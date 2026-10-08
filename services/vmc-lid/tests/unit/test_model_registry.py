import os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from core.model_registry import ModelRegistry, ModelMetadata
from config.settings import Settings


class TestModelRegistry:
    def test_load_open_vakgyata_metadata(self):
        settings = Settings.load()
        registry = ModelRegistry(settings.model.registry_path, settings.model.models_dir)

        meta = registry.get_model("open-vakgyata")
        assert meta.model_id == "open-vakgyata"
        assert meta.framework == "onnx"
        assert meta.parameters == 58746794
        assert meta.sample_rate == 16000
        assert len(meta.supported_languages) == 10
        assert "hi-IN" in meta.supported_languages
        assert "te-IN" in meta.supported_languages
        assert meta.sha256 is not None

    def test_validate_artifact_existence(self):
        settings = Settings.load()
        registry = ModelRegistry(settings.model.registry_path, settings.model.models_dir)

        assert registry.validate_artifact("open-vakgyata") is True

    def test_unknown_model_raises_keyerror(self):
        settings = Settings.load()
        registry = ModelRegistry(settings.model.registry_path, settings.model.models_dir)

        with pytest.raises(KeyError):
            registry.get_model("nonexistent-model-xyz")
