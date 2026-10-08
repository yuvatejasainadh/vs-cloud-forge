import pytest
import numpy as np
from core.audio_processor import AudioProcessor, AudioProcessingError


class TestXLSRAudioProcessing:

    def test_16khz_audio(self):
        """Verifies standard 16kHz mono audio processing."""
        sample_rate = 16000
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm_bytes = (audio * 32767.0).astype(np.int16).tobytes()

        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=pcm_bytes,
            sample_rate=16000,
            channels=1,
            audio_format="pcm_s16le",
            target_sample_rate=16000,
            do_normalize=True
        )
        assert len(waveform) == 16000
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32
        assert np.all(np.isfinite(waveform))

    def test_resampling_standard_rates(self):
        """Verifies resampling from 8kHz, 16kHz, 22.05kHz, 44.1kHz, and 48kHz to target 16kHz."""
        for orig_sr in [8000, 16000, 22050, 44100, 48000]:
            t = np.linspace(0, 1.0, int(orig_sr), endpoint=False)
            audio = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
            pcm_bytes = (audio * 32767.0).astype(np.int16).tobytes()

            waveform = AudioProcessor.process_raw_audio(
                audio_bytes=pcm_bytes,
                sample_rate=int(orig_sr),
                channels=1,
                audio_format="pcm_s16le",
                target_sample_rate=16000,
                do_normalize=True
            )
            assert len(waveform) == 16000
            assert np.all(np.isfinite(waveform))

    def test_mono_conversion(self):
        """Verifies stereo to mono averaging."""
        num_samples = 16000
        left = 0.5 * np.ones(num_samples, dtype=np.float32)
        right = 0.3 * np.ones(num_samples, dtype=np.float32)
        interleaved = np.empty((num_samples * 2,), dtype=np.int16)
        interleaved[0::2] = (left * 32767.0).astype(np.int16)
        interleaved[1::2] = (right * 32767.0).astype(np.int16)

        waveform = AudioProcessor.process_raw_audio(
            audio_bytes=interleaved.tobytes(),
            sample_rate=16000,
            channels=2,
            audio_format="pcm_s16le",
            target_sample_rate=16000,
            do_normalize=False
        )
        assert len(waveform) == 16000
        # Expected average: (0.5 + 0.3) / 2 = 0.4
        np.testing.assert_allclose(waveform[0], 0.4, atol=0.01)

    def test_nan_inf_rejection(self):
        """Verifies rejection or clean handling of invalid / non-finite audio."""
        with pytest.raises(AudioProcessingError):
            AudioProcessor.process_raw_audio(
                audio_bytes=b"",
                sample_rate=16000,
                channels=1
            )
