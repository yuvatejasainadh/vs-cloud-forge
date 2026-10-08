import io
import struct
import numpy as np
import pytest
import soundfile as sf

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from core.audio_processor import AudioProcessor, AudioProcessingError


class TestAudioProcessor:
    def test_process_valid_pcm_s16le(self):
        # 1 second of 16kHz sine tone
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        int16_bytes = (tone * 32767).astype(np.int16).tobytes()

        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=int16_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            target_sample_rate=16000,
            do_normalize=True
        )

        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert len(waveform) == sr
        assert waveform.dtype == np.float32
        # Check Wav2Vec2 zero-mean unit-variance
        assert np.isclose(np.mean(waveform), 0.0, atol=1e-3)
        assert np.isclose(np.var(waveform), 1.0, atol=1e-2)

    def test_stereo_to_mono_conversion(self):
        sr = 16000
        num_samples = 8000
        # 2 channels
        ch1 = (0.5 * np.ones(num_samples)).astype(np.float32)
        ch2 = (0.3 * np.ones(num_samples)).astype(np.float32)
        stereo = np.column_stack([ch1, ch2])
        int16_stereo = (stereo * 32767).astype(np.int16).tobytes()

        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=int16_stereo,
            sample_rate=sr,
            channels=2,
            audio_format="pcm_s16le",
            target_sample_rate=16000,
            do_normalize=False
        )

        assert waveform.ndim == 1
        assert len(waveform) == num_samples
        # Expected average is ~0.4
        assert np.isclose(np.mean(waveform), 0.4, atol=1e-2)

    def test_resampling(self):
        # 8kHz source audio -> resampled to 16kHz
        orig_sr = 8000
        target_sr = 16000
        t = np.linspace(0, 0.5, int(orig_sr * 0.5), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 200 * t)
        int16_bytes = (tone * 32767).astype(np.int16).tobytes()

        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=int16_bytes,
            sample_rate=orig_sr,
            channels=1,
            audio_format="pcm_s16le",
            target_sample_rate=target_sr,
            do_normalize=False
        )

        assert len(waveform) == int(target_sr * 0.5)

    def test_wav_container_decoding(self):
        sr = 16000
        num_samples = 16000
        t = np.linspace(0, 1.0, num_samples, endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        bio = io.BytesIO()
        sf.write(bio, audio_data, sr, format="WAV", subtype="PCM_16")
        wav_bytes = bio.getvalue()

        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=wav_bytes,
            sample_rate=sr,
            channels=1,
            audio_format="wav",
            target_sample_rate=16000,
            do_normalize=True
        )

        assert len(waveform) == num_samples
        assert waveform.dtype == np.float32

    def test_reject_empty_audio(self):
        with pytest.raises(AudioProcessingError, match="empty"):
            AudioProcessor.process_raw_audio(b"", 16000, 1, "pcm_s16le")

    def test_reject_corrupt_audio(self):
        # Odd single byte cannot form PCM s16le
        with pytest.raises(AudioProcessingError):
            AudioProcessor.process_raw_audio(b"\x01", 16000, 1, "pcm_s16le")
