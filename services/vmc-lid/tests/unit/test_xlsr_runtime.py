import os
import pytest
import numpy as np
from core.model_registry import ModelMetadata, ModelRole, ModelStatus, ModelTask
from core.model_runtime import AudioInferenceInput
from core.classification_head import (
    NoClassificationHead,
    LinearClassificationHead,
    ClassificationHeadNotAvailableError,
)
from runtimes.xlsr_runtime import XLSRRuntime, MeanPooling
from config.settings import Settings


@pytest.fixture
def xlsr_metadata():
    return ModelMetadata(
        model_id="facebook-wav2vec2-xls-r-300m",
        model_name="Facebook Wav2Vec2 XLS-R 300M (Speech Encoder Backbone)",
        role=ModelRole.BACKBONE.value,
        task=ModelTask.SPEECH_ENCODER.value,
        architecture="WAV2VEC2-XLS-R",
        family="XLS-R",
        parameter_count=317374720,
        parameters=317374720,
        provider="meta-facebook",
        source="huggingface",
        repository="facebook/wav2vec2-xls-r-300m",
        revision="1a640f32ac3e39899438a2931f9924c02f080a54",
        runtime="transformers-pytorch",
        framework="transformers-pytorch",
        version="1a640f32ac3e39899438a2931f9924c02f080a54",
        sample_rate=16000,
        channels=1,
        status="ACTIVE",
        supported_languages=[],
        language_labels=[]
    )


class TestXLSRRuntime:

    def test_xlsr_runtime_metadata(self, xlsr_metadata):
        """Verifies metadata extraction and property accessors from XLSRRuntime instance."""
        settings = Settings()
        runtime = XLSRRuntime(metadata=xlsr_metadata, settings=settings)
        assert runtime.model_id == "facebook-wav2vec2-xls-r-300m"
        assert runtime.model_version == "1a640f32ac3e39899438a2931f9924c02f080a54"
        assert runtime.revision == "1a640f32ac3e39899438a2931f9924c02f080a54"
        assert runtime.runtime == "transformers-pytorch"
        assert runtime.framework == "transformers-pytorch"
        assert runtime.device in ("cpu", "cuda")
        assert runtime.sample_rate == 16000
        assert runtime.hidden_size == 1024
        assert runtime.parameter_count == 317374720
        assert runtime.metadata.is_backbone is True

    def test_xlsr_runtime_load(self, xlsr_metadata):
        """Verifies unhydrated load handling does not crash but reports status correctly."""
        settings = Settings()
        runtime = XLSRRuntime(metadata=xlsr_metadata, settings=settings)
        runtime.load()
        # When unhydrated, is_loaded is False
        assert runtime.is_loaded is False

    def test_xlsr_runtime_health(self, xlsr_metadata):
        """Verifies health diagnostics for backbone."""
        settings = Settings()
        runtime = XLSRRuntime(metadata=xlsr_metadata, settings=settings)
        h = runtime.health()
        assert "status" in h
        assert h["model_id"] == "facebook-wav2vec2-xls-r-300m"
        assert h["role"] == "BACKBONE"
        assert h["hidden_size"] == 1024
        assert h["parameter_count"] == 317374720
        assert h["classification_head_available"] is False
        assert h["pooling_strategy"] == "masked_mean"

    def test_xlsr_runtime_unload(self, xlsr_metadata):
        """Verifies unload resets session state."""
        settings = Settings()
        runtime = XLSRRuntime(metadata=xlsr_metadata, settings=settings)
        runtime.unload()
        assert runtime.is_loaded is False

    def test_xlsr_pooling(self):
        """Verifies MeanPooling with and without attention mask."""
        pooler = MeanPooling(strategy="masked_mean")
        # Hidden states: batch=1, time=4, hidden_dim=1024
        hidden_states = np.ones((1, 4, 1024), dtype=np.float32)
        hidden_states[0, 2:, :] = 3.0  # Frames 2 & 3 have value 3.0

        # Unmasked mean: (1+1+3+3)/4 = 2.0
        pooled_unmasked = pooler.forward(hidden_states)
        assert pooled_unmasked.shape == (1, 1024)
        np.testing.assert_allclose(pooled_unmasked[0, 0], 2.0)

        # Masked mean: only first 2 frames active -> (1+1)/2 = 1.0
        mask = np.array([[1, 1, 0, 0]], dtype=np.int32)
        pooled_masked = pooler.forward(hidden_states, attention_mask=mask)
        assert pooled_masked.shape == (1, 1024)
        np.testing.assert_allclose(pooled_masked[0, 0], 1.0)

    def test_xlsr_output_shape(self):
        """Verifies that pooled representations have expected hidden dimension."""
        pooler = MeanPooling()
        dummy_hidden = np.random.randn(2, 50, 1024).astype(np.float32)
        pooled = pooler.forward(dummy_hidden)
        assert pooled.shape == (2, 1024)

    def test_xlsr_infer_without_head_returns_model_head_not_available(self, xlsr_metadata):
        """Verifies that inferring on backbone without head returns explicit MODEL_HEAD_NOT_AVAILABLE."""
        settings = Settings()
        runtime = XLSRRuntime(metadata=xlsr_metadata, settings=settings)
        
        # 1-second 16kHz test audio
        pcm_bytes = (np.sin(np.linspace(0, 1, 16000)) * 10000).astype(np.int16).tobytes()
        inp = AudioInferenceInput(
            audio_data=pcm_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le"
        )
        res = runtime.infer(inp)
        assert res.language == "UNKNOWN"
        assert res.confidence == 0.0
        assert res.extra.get("status") == "MODEL_HEAD_NOT_AVAILABLE"

    def test_xlsr_with_mock_classification_head(self, xlsr_metadata):
        """Verifies pluggable head prediction integration."""
        settings = Settings()
        # Create a mock head with 19 canonical labels
        labels = [
            "te-IN", "ta-IN", "ml-IN", "kn-IN", "hi-IN", "mr-IN", "sa-IN",
            "bn-IN", "as-IN", "or-IN", "bpy-IN", "pa-IN", "ks-IN", "sd-IN",
            "ur-IN", "gu-IN", "raj-IN", "kut-IN", "en-IN"
        ]
        weights = np.zeros((1024, len(labels)), dtype=np.float32)
        # Give high weight to Telugu (te-IN, index 0)
        weights[0, 0] = 5.0
        bias = np.zeros(len(labels), dtype=np.float32)

        head = LinearClassificationHead(weights=weights, bias=bias)
        assert head.is_available is True
        assert len(head.supported_languages) == 19

        # Test predict on embedding
        emb = np.zeros((1, 1024), dtype=np.float32)
        emb[0, 0] = 1.0
        top_lang, conf, probs = head.predict(emb)
        assert top_lang == "te-IN"
        assert conf > 0.5
        assert len(probs) == 19
