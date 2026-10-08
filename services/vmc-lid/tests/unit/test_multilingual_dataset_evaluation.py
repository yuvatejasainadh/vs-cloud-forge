import os
import sys
import pytest
import numpy as np
import soundfile as sf

# Add src, service, and tools directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

for p in [SRC_DIR, BASE_DIR, TOOLS_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.model_registry import ModelMetadata, ModelRole, ModelStatus, ModelTask
from core.model_runtime import AudioInferenceInput
from core.classification_head import (
    NoClassificationHead,
    LinearClassificationHead,
    ClassificationHeadNotAvailableError,
)
from runtimes.xlsr_runtime import XLSRRuntime, MeanPooling
from config.settings import Settings
from routing.canonical_registry import get_canonical_registry
from routing.language_group_resolver import LanguageGroupResolver


DATASET_ROOT = r"C:\Users\yuvat\Music\Multilingual Dataset"


class TestMultilingualDatasetEvaluation:

    def test_dataset_discovery_and_metadata(self):
        """Verifies dataset directory existence, CSV metadata and audio directory presence."""
        if not os.path.exists(DATASET_ROOT):
            pytest.skip(f"Dataset root not found at {DATASET_ROOT}")

        meta_dir = os.path.join(DATASET_ROOT, "metadata")
        audio_dir = os.path.join(DATASET_ROOT, "audio")

        assert os.path.exists(meta_dir)
        assert os.path.exists(audio_dir)
        assert os.path.exists(os.path.join(meta_dir, "train.csv"))
        assert os.path.exists(os.path.join(meta_dir, "test.csv"))

    def test_canonical_label_mapping(self):
        """Verifies exact mapping of dataset languages to CloudForge 19 canonical labels."""
        from evaluate_multilingual_dataset import LANGUAGE_NAME_TO_CANONICAL
        reg = get_canonical_registry()
        canonical_labels = set(reg.lid_labels())

        for name, code in LANGUAGE_NAME_TO_CANONICAL.items():
            assert code in canonical_labels, f"Mapped code {code} for '{name}' must exist in canonical registry"

        # Check key languages
        assert LANGUAGE_NAME_TO_CANONICAL["Telugu"] == "te-IN"
        assert LANGUAGE_NAME_TO_CANONICAL["Tamil"] == "ta-IN"
        assert LANGUAGE_NAME_TO_CANONICAL["Hindi"] == "hi-IN"
        assert LANGUAGE_NAME_TO_CANONICAL["English"] == "en-IN"
        assert LANGUAGE_NAME_TO_CANONICAL["Gujarati"] == "gu-IN"
        assert LANGUAGE_NAME_TO_CANONICAL["Assamese"] == "as-IN"

    def test_unsupported_and_ambiguous_labels(self):
        """Verifies explicit identification of ambiguous and unsupported language labels."""
        from evaluate_multilingual_dataset import AMBIGUOUS_LANGUAGES, UNSUPPORTED_LANGUAGES

        # Manipuri must be flagged as ambiguous
        assert "Manipuri" in AMBIGUOUS_LANGUAGES

        # Bodo, Dogri, Nepali must be flagged as unsupported
        assert "Bodo" in UNSUPPORTED_LANGUAGES
        assert "Dogri" in UNSUPPORTED_LANGUAGES
        assert "Nepali" in UNSUPPORTED_LANGUAGES

    def test_audio_loading_and_specs(self):
        """Verifies loading real WAV audio files from the dataset with 16kHz mono PCM specifications."""
        if not os.path.exists(DATASET_ROOT):
            pytest.skip(f"Dataset root not found at {DATASET_ROOT}")

        audio_dir = os.path.join(DATASET_ROOT, "audio")
        files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
        if not files:
            pytest.skip("No audio files found")

        # Test first 5 files
        for fpath in files[:5]:
            info = sf.info(fpath)
            assert info.samplerate == 16000
            assert info.channels == 1
            assert info.subtype == "PCM_16"
            assert info.duration > 0.0

    def test_xlsr_representation_shape_and_pooling(self):
        """Verifies XLS-R MeanPooling aggregates 1024-dimensional utterance embeddings."""
        pooler = MeanPooling(strategy="masked_mean")
        hidden_states = np.random.randn(1, 150, 1024).astype(np.float32)
        mask = np.ones((1, 150), dtype=np.int32)

        pooled = pooler.forward(hidden_states, attention_mask=mask)
        assert pooled.shape == (1, 1024)
        assert pooled.dtype == np.float32

    def test_no_fabrication_guarantee(self):
        """Verifies that inferring on XLS-R backbone without a trained head returns MODEL_HEAD_NOT_AVAILABLE."""
        meta = ModelMetadata(
            model_id="facebook-wav2vec2-xls-r-300m",
            model_name="Facebook Wav2Vec2 XLS-R 300M",
            role=ModelRole.BACKBONE.value,
            task=ModelTask.SPEECH_ENCODER.value,
            parameters=317374720,
            status="ACTIVE"
        )
        settings = Settings()
        runtime = XLSRRuntime(metadata=meta, settings=settings)

        # 1-second 16kHz audio input
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
        assert res.probabilities == {}
        assert res.extra.get("status") == "MODEL_HEAD_NOT_AVAILABLE"

    def test_canonical_routing_contract(self):
        """Verifies that all 12 mapped dataset languages route to the correct canonical VMs."""
        from evaluate_multilingual_dataset import LANGUAGE_NAME_TO_CANONICAL
        reg = get_canonical_registry()
        resolver = LanguageGroupResolver(registry=reg)

        expected_vm_mappings = {
            "te-IN": "VM2_SOUTH",
            "ta-IN": "VM2_SOUTH",
            "ml-IN": "VM2_SOUTH",
            "kn-IN": "VM2_SOUTH",
            "hi-IN": "VM3_NORTH",
            "mr-IN": "VM3_NORTH",
            "sa-IN": "VM3_NORTH",
            "bn-IN": "VM4_EAST",
            "as-IN": "VM4_EAST",
            "or-IN": "VM4_EAST",
            "gu-IN": "VM6_WEST",
            "en-IN": "VM7_ENGLISH",
        }

        for lang_name, canon_code in LANGUAGE_NAME_TO_CANONICAL.items():
            routing = resolver.resolve(canon_code)
            expected_vm = expected_vm_mappings[canon_code]
            assert routing.vm == expected_vm, f"{lang_name} ({canon_code}) expected {expected_vm}, got {routing.vm}"
