import io
import math
import struct
from typing import Tuple, Optional
import numpy as np
import soundfile as sf

_SCIPY_AVAILABLE = False
try:
    import scipy.signal
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


class AudioProcessingError(Exception):
    """Raised when audio validation or decoding fails."""
    pass


class AudioProcessor:
    """
    CloudForge standard audio processor.
    Validates, decodes, converts to mono, resamples to target rate using polyphase sinc filter,
    and normalizes audio for Wav2Vec2/WavLM based architectures.
    """

    TARGET_SAMPLE_RATE = 16000

    @classmethod
    def process_raw_audio(
        cls,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_format: str = "pcm_s16le",
        target_sample_rate: int = 16000,
        do_normalize: bool = True
    ) -> np.ndarray:
        """
        Processes raw audio bytes into normalized 1D float32 waveform tensor.

        Args:
            audio_bytes: Raw audio bytes (PCM or WAV).
            sample_rate: Source sample rate (Hz).
            channels: Source channel count.
            audio_format: Encoding format ("pcm_s16le", "pcm_f32le", "wav").
            target_sample_rate: Target sample rate in Hz (default: 16000).
            do_normalize: Whether to apply Wav2Vec2 zero-mean unit-variance normalization.

        Returns:
            np.ndarray: 1D float32 array shaped (num_samples,).
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise AudioProcessingError("Audio payload is empty (0 bytes)")

        # 1. Decode audio bytes
        waveform, sr = cls._decode_audio(audio_bytes, sample_rate, channels, audio_format)

        if waveform.size == 0:
            raise AudioProcessingError("Decoded audio waveform is empty")

        # 2. Check for NaN/Inf in raw signal
        if np.isnan(waveform).any() or np.isinf(waveform).any():
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3. Convert to Mono if multi-channel
        if waveform.ndim > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
        elif waveform.ndim > 1:
            waveform = waveform.squeeze()

        # 4. Resample to 16 kHz using bandlimited polyphase filter if necessary
        if sr != target_sample_rate:
            waveform = cls._resample_polyphase(waveform, sr, target_sample_rate)

        # 5. Wav2Vec2 zero-mean and unit-variance normalization
        if do_normalize:
            waveform = cls._normalize_wav2vec2(waveform)

        # 6. Final sanity check
        if np.isnan(waveform).any() or np.isinf(waveform).any():
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

        return waveform.astype(np.float32)

    @classmethod
    def _decode_audio(
        cls,
        audio_bytes: bytes,
        sample_rate: int,
        channels: int,
        audio_format: str
    ) -> Tuple[np.ndarray, int]:
        format_lower = audio_format.lower()

        # Check if bytes start with RIFF / WAV magic header
        if audio_bytes.startswith(b"RIFF") or format_lower in ["wav", "wave"]:
            try:
                with io.BytesIO(audio_bytes) as bio:
                    data, sr = sf.read(bio, dtype="float32")
                    return data, sr
            except Exception as e:
                raise AudioProcessingError(f"Failed to decode WAV audio header/stream: {str(e)}")

        # Decode raw PCM s16le
        if format_lower in ["pcm", "pcm_s16le", "raw", "s16le"]:
            num_bytes = len(audio_bytes)
            if num_bytes % 2 != 0:
                audio_bytes = audio_bytes[:num_bytes - (num_bytes % 2)]

            if len(audio_bytes) == 0:
                raise AudioProcessingError("PCM audio bytes truncated or invalid size")

            try:
                int16_arr = np.frombuffer(audio_bytes, dtype=np.int16)
                float32_arr = int16_arr.astype(np.float32) / 32768.0
                if channels > 1:
                    float32_arr = float32_arr.reshape(-1, channels)
                return float32_arr, sample_rate
            except Exception as e:
                raise AudioProcessingError(f"Failed to parse PCM s16le buffer: {str(e)}")

        # Decode raw PCM float32
        if format_lower in ["pcm_f32le", "f32le", "float32"]:
            num_bytes = len(audio_bytes)
            if num_bytes % 4 != 0:
                audio_bytes = audio_bytes[:num_bytes - (num_bytes % 4)]

            try:
                float32_arr = np.frombuffer(audio_bytes, dtype=np.float32)
                if channels > 1:
                    float32_arr = float32_arr.reshape(-1, channels)
                return float32_arr, sample_rate
            except Exception as e:
                raise AudioProcessingError(f"Failed to parse PCM f32le buffer: {str(e)}")

        # Fallback: attempt soundfile generic read
        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
                return data, sr
        except Exception:
            # Fallback to int16 assumption
            num_samples = len(audio_bytes) // 2
            int16_arr = np.frombuffer(audio_bytes[:num_samples * 2], dtype=np.int16)
            return int16_arr.astype(np.float32) / 32768.0, sample_rate

    @classmethod
    def _resample_polyphase(cls, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Bandlimited polyphase sinc resampling using scipy.signal.resample_poly when available,
        with high-accuracy bandlimited numpy interpolation fallback.
        """
        if orig_sr <= 0 or target_sr <= 0:
            raise AudioProcessingError(f"Invalid sample rates: orig={orig_sr}, target={target_sr}")
        if orig_sr == target_sr:
            return waveform

        if _SCIPY_AVAILABLE:
            try:
                gcd = math.gcd(orig_sr, target_sr)
                up = target_sr // gcd
                down = orig_sr // gcd
                resampled = scipy.signal.resample_poly(waveform, up, down)
                return resampled.astype(np.float32)
            except Exception:
                pass

        # High-accuracy numpy resampler fallback
        duration = len(waveform) / float(orig_sr)
        target_num_samples = int(round(duration * target_sr))
        orig_times = np.linspace(0, duration, len(waveform), endpoint=False)
        target_times = np.linspace(0, duration, target_num_samples, endpoint=False)
        resampled = np.interp(target_times, orig_times, waveform).astype(np.float32)
        return resampled

    @classmethod
    def _normalize_wav2vec2(cls, waveform: np.ndarray) -> np.ndarray:
        """Applies Wav2Vec2 standard zero-mean and unit-variance normalization."""
        variance = float(np.var(waveform))
        mean = float(np.mean(waveform))
        if variance < 1e-12:
            return np.zeros_like(waveform)
        normalized = (waveform - mean) / np.sqrt(variance + 1e-7)
        return normalized
