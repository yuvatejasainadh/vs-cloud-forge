import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from config.settings import Settings
from core.model_registry import ModelRegistry
from core.model_runtime import AudioInferenceInput
from core.runtime_factory import ModelRuntimeFactory
from routing.canonical_registry import get_canonical_registry
import runtimes


class TestSmokeInference:
    """
    Smoke tests for Open-Vakgyata LID model running end-to-end inference across diverse speech audio patterns.
    """

    SUPPORTED_LANGUAGES = [
        "en-IN", "hi-IN", "or-IN", "bn-IN", "ta-IN",
        "te-IN", "kn-IN", "ml-IN", "mr-IN", "gu-IN"
    ]

    CANONICAL_GROUPS = {"SOUTH", "NORTH", "EAST", "NORTH_WEST_FRONTIER", "WEST", "ENGLISH"}

    @pytest.fixture(scope="class")
    def setup_pipeline(self):
        settings = Settings.load()
        registry = ModelRegistry(settings.model.registry_path, settings.model.models_dir)
        meta = registry.get_model("open-vakgyata")
        runtime = ModelRuntimeFactory.create(meta, settings)
        runtime.load()
        canonical_reg = get_canonical_registry(settings.routing.canonical_registry_file)
        yield runtime, canonical_reg
        runtime.unload()

    def test_smoke_end_to_end_predictions(self, setup_pipeline):
        runtime, canonical_reg = setup_pipeline

        # Test multi-frequency multi-duration audio inputs
        test_patterns = [
            (1.0, 300),   # 1s 300Hz tone
            (2.0, 500),   # 2s 500Hz tone
            (3.0, 1000),  # 3s 1000Hz tone
        ]

        for duration, freq in test_patterns:
            sr = 16000
            num_samples = int(sr * duration)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            tone = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(num_samples)
            audio_bytes = (tone * 32767).astype(np.int16).tobytes()

            inp = AudioInferenceInput(
                audio_data=audio_bytes,
                sample_rate=sr,
                channels=1,
                audio_format="pcm_s16le",
                request_id=f"smoke_d{duration}_f{freq}"
            )

            result = runtime.infer(inp)

            # Assertions
            assert result.language in self.SUPPORTED_LANGUAGES, f"Invalid predicted language: {result.language}"
            assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} out of range [0, 1]"
            assert len(result.probabilities) == 10, "Should have probabilities for all 10 Indic languages"
            
            # Canonical Group resolution verification
            group = canonical_reg.resolve_group(result.language)
            assert group in self.CANONICAL_GROUPS, f"Invalid language group: {group}"

            vm = canonical_reg.resolve_vm(result.language)
            assert vm.startswith("VM"), f"Invalid target VM: {vm}"

            lang_name = canonical_reg.get_language_name(result.language)
            assert lang_name != "Unknown", f"Unrecognized language name for: {result.language}"
