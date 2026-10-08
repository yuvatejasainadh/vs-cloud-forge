import io
import os
import sys
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from core.audio_processor import EnglishAudioProcessor, AudioProcessingError


class TestEnglishAudioProcessor:
    def test_process_valid_pcm_s16le(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        int16_bytes = (tone * 32767).astype(np.int16).tobytes()

        waveform = EnglishAudioProcessor.process_raw_audio(
            audio_bytes=int16_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le"
        )

        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32
        assert len(waveform) == sr
        assert not np.isnan(waveform).any()
        assert not np.isinf(waveform).any()

    def test_stereo_to_mono_conversion(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 300 * t)
        right = 0.5 * np.sin(2 * np.pi * 600 * t)
        stereo = np.column_stack((left, right))
        int16_bytes = (stereo * 32767).astype(np.int16).tobytes()

        waveform = EnglishAudioProcessor.process_raw_audio(
            audio_bytes=int16_bytes,
            sample_rate=16000,
            channels=2,
            audio_format="pcm_s16le"
        )

        assert waveform.ndim == 1
        assert len(waveform) == sr

    def test_polyphase_resampling(self):
        sr_orig = 44100
        t = np.linspace(0, 1.0, sr_orig, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        int16_bytes = (tone * 32767).astype(np.int16).tobytes()

        waveform = EnglishAudioProcessor.process_raw_audio(
            audio_bytes=int16_bytes,
            sample_rate=sr_orig,
            channels=1,
            target_sample_rate=16000
        )

        assert len(waveform) == 16000

    def test_aasist_fixed_length_padding(self):
        sr = 16000
        short_audio = (np.zeros(8000, dtype=np.float32) * 32767).astype(np.int16).tobytes()

        waveform = EnglishAudioProcessor.process_raw_audio(
            audio_bytes=short_audio,
            sample_rate=sr,
            channels=1,
            fix_length=True
        )

        assert len(waveform) == EnglishAudioProcessor.AASIST_STANDARD_SAMPLES

    def test_reject_empty_audio(self):
        with pytest.raises(AudioProcessingError):
            EnglishAudioProcessor.process_raw_audio(b"")

    def test_wav_container_decoding(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

        bio = io.BytesIO()
        sf.write(bio, tone, sr, format="WAV")
        wav_bytes = bio.getvalue()

        waveform = EnglishAudioProcessor.process_raw_audio(
            audio_bytes=wav_bytes,
            audio_format="wav"
        )
        assert len(waveform) == sr
