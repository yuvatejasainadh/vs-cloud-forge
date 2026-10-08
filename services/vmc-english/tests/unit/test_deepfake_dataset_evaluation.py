import pytest
import os
import sys
import json
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen")))

from config.settings import Settings
from core.artifact_resolver import ArtifactResolver, ArtifactNotFoundError
from core.audio_processor import EnglishAudioProcessor
from core.runtime_factory import AASISTLRuntimeFactory
from core.runtime import AASISTLRuntime, ModelUnavailableError
from core.detector import EnglishDeepfakeDetector


class TestDeepfakeDatasetEvaluation:

    @pytest.fixture
    def settings(self):
        return Settings.load()

    @pytest.fixture
    def runtime(self, settings):
        rt = AASISTLRuntimeFactory.create(settings)
        rt.load()
        return rt

    @pytest.fixture
    def detector(self, settings, runtime):
        return EnglishDeepfakeDetector(settings, runtime)

    def test_vm7_routing_contract_and_language_gating(self, detector):
        """Verify VM7 contract: accepts en-IN, rejects other languages."""
        dummy_audio = (np.zeros(64600, dtype=np.float32) * 32767).astype(np.int16).tobytes()

        # Non-English rejection
        res_hi = detector.analyze(
            audio_content=dummy_audio,
            sample_rate=16000,
            language="hi-IN",
            language_group="NORTH",
            request_id="gate_hi"
        )
        assert res_hi.status == EnglishDeepfakeDetector.STATUS_UNSUPPORTED_LANGUAGE

        res_te = detector.analyze(
            audio_content=dummy_audio,
            sample_rate=16000,
            language="te-IN",
            language_group="SOUTH",
            request_id="gate_te"
        )
        assert res_te.status == EnglishDeepfakeDetector.STATUS_UNSUPPORTED_LANGUAGE

        # English accepted
        res_en = detector.analyze(
            audio_content=dummy_audio,
            sample_rate=16000,
            language="en-IN",
            language_group="ENGLISH",
            request_id="gate_en"
        )
        assert res_en.status in (
            EnglishDeepfakeDetector.STATUS_GENUINE,
            EnglishDeepfakeDetector.STATUS_SPOOF,
            EnglishDeepfakeDetector.STATUS_LOW_CONFIDENCE,
            EnglishDeepfakeDetector.STATUS_MODEL_UNAVAILABLE
        )
        assert res_en.target_vm == "VM7_ENGLISH"
        assert res_en.language_group == "ENGLISH"
        assert res_en.language == "en-IN"

    def test_hydrated_artifact_inference(self, runtime):
        """Verify that hydrated AASIST-L runtime generates valid predictions."""
        if not runtime.is_ready():
            pytest.skip("AASIST-L model artifact not yet hydrated in local cache.")

        assert runtime.is_ready() is True
        health = runtime.health()
        assert health["status"] == "SERVING"
        assert health["loaded"] is True
        assert health["architecture"] == "AASIST-L"
        assert health["runtime"] == "onnx"

        waveform = np.zeros(64600, dtype=np.float32)
        pred = runtime.predict(waveform)
        assert 0.0 <= pred.spoof_score <= 1.0
        assert 0.0 <= pred.bonafide_score <= 1.0
        assert abs(pred.spoof_score + pred.bonafide_score - 1.0) < 1e-4
        assert pred.inference_duration_ms > 0.0

    def test_unhydrated_artifact_anti_fabrication(self):
        """Verify that when model.onnx is missing, no predictions are fabricated."""
        unhydrated_rt = AASISTLRuntime(model_id="non-existent-model", version="v0.0.0")
        unhydrated_rt.load()

        assert unhydrated_rt.is_ready() is False
        health = unhydrated_rt.health()
        assert health["status"] == "AWAITING_ARTIFACT"
        assert health["loaded"] is False

        with pytest.raises(ModelUnavailableError):
            unhydrated_rt.predict(np.zeros(64600, dtype=np.float32))

    def test_audio_preprocessing_aasist_windowing(self):
        """Verify EnglishAudioProcessor produces exact 64,600 length AASIST float32 waveform."""
        # 1-second sine wave
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio_pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()

        proc = EnglishAudioProcessor.process_raw_audio(
            audio_bytes=audio_pcm,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            target_sample_rate=16000,
            do_normalize=True,
            fix_length=True
        )

        assert isinstance(proc, np.ndarray)
        assert proc.dtype == np.float32
        assert len(proc) == EnglishAudioProcessor.AASIST_STANDARD_SAMPLES  # 64600
        assert not np.isnan(proc).any()
        assert not np.isinf(proc).any()

    def test_deepfake_manifest_and_summary_integrity(self):
        """Verify manifest and summary JSON reports if generated."""
        summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "reports", "deepfake_dataset_evaluation_summary.json"))
        if not os.path.exists(summary_path):
            pytest.skip("Deepfake evaluation summary not yet generated.")

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["dataset"]["total_files"] == 12000
        assert data["model"]["architecture"] == "AASIST-L"
        assert data["model"]["target_vm"] == "VM7_ENGLISH"
        assert "evaluation_name" in data
