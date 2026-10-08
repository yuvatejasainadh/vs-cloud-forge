import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from config.settings import Settings
from core.model_registry import ModelRegistry
from core.model_runtime import AudioInferenceInput
from core.runtime_factory import ModelRuntimeFactory
import runtimes


class TestOpenVakgyataRuntimeIntegration:
    @pytest.fixture(scope="class")
    def runtime(self):
        settings = Settings.load()
        registry = ModelRegistry(settings.model.registry_path, settings.model.models_dir)
        meta = registry.get_model("open-vakgyata")
        rt = ModelRuntimeFactory.create(meta, settings)
        rt.load()
        yield rt
        rt.unload()

    def test_runtime_health(self, runtime):
        h = runtime.health()
        assert h["status"] == "SERVING"
        assert h["model_id"] == "open-vakgyata"
        assert h["session_initialized"] is True
        assert h["artifact_exists"] is True

    def test_runtime_infer_valid_audio(self, runtime):
        # 1.5 seconds of 16kHz audio
        sr = 16000
        num_samples = int(sr * 1.5)
        t = np.linspace(0, 1.5, num_samples, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        int16_bytes = (audio * 32767).astype(np.int16).tobytes()

        inp = AudioInferenceInput(
            audio_data=int16_bytes,
            sample_rate=sr,
            channels=1,
            audio_format="pcm_s16le",
            request_id="int_test_1"
        )

        res = runtime.infer(inp)
        assert res.model_id == "open-vakgyata"
        assert res.language in runtime.metadata.supported_languages
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.probabilities) == 10
        assert res.inference_duration_ms > 0
        # Probabilities should sum to approximately 1.0
        assert np.isclose(sum(res.probabilities.values()), 1.0, atol=1e-3)
