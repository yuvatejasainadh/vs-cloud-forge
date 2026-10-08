import os
import pytest
from core.model_registry import ModelRegistry, ModelRole, ModelStatus, ModelMetadata


@pytest.fixture
def registry():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    reg_path = os.path.join(base_dir, "..", "..", "models", "registry.json")
    models_dir = os.path.join(base_dir, "..", "..", "models")
    return ModelRegistry(registry_path=reg_path, models_base_dir=models_dir)


class TestXLSRModelRegistry:

    def test_xlsr_registry_entry(self, registry):
        """Verifies that facebook-wav2vec2-xls-r-300m exists in the model registry."""
        meta = registry.get_model("facebook-wav2vec2-xls-r-300m")
        assert meta is not None
        assert meta.model_id == "facebook-wav2vec2-xls-r-300m"
        assert meta.architecture == "WAV2VEC2-XLS-R"
        assert meta.family == "XLS-R"
        assert meta.source == "huggingface"
        assert meta.repository == "facebook/wav2vec2-xls-r-300m"
        assert meta.revision == "1a640f32ac3e39899438a2931f9924c02f080a54"

    def test_xlsr_role_is_backbone(self, registry):
        """Verifies that XLS-R 300M is explicitly registered with role BACKBONE."""
        meta = registry.get_model("facebook-wav2vec2-xls-r-300m")
        assert meta.role == ModelRole.BACKBONE.value
        assert meta.is_backbone is True
        assert meta.task == "SPEECH_ENCODER"

    def test_xlsr_active_backbone_status(self, registry):
        """Verifies that XLS-R model is ACTIVE as speech encoder backbone without fabricated language labels."""
        meta = registry.get_model("facebook-wav2vec2-xls-r-300m")
        assert meta.status == "ACTIVE"
        assert meta.role == "BACKBONE"
        assert meta.task == "SPEECH_ENCODER"
        # Must have no 19-class language labels directly assigned to the backbone
        assert len(meta.language_labels) == 0
        assert len(meta.supported_languages) == 0

    def test_open_vakgyata_preserved(self, registry):
        """Verifies that Open-Vakgyata remains preserved as the benchmark model."""
        meta = registry.get_model("open-vakgyata")
        assert meta is not None
        assert meta.role == ModelRole.END_TO_END.value
        assert meta.status == ModelStatus.BENCHMARK.value
        assert len(meta.supported_languages) == 10
