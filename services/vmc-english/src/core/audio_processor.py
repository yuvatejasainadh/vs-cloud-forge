import io
import math
import struct
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import scipy.signal


class AudioProcessingError(Exception):
    """Raised when audio validation or decoding fails."""
    pass


class EnglishAudioProcessor:
    """
    Standard CloudForge Audio Processor tailored for English AASIST-L deepfake detection.
    Performs polyphase sinc resampling to 16 kHz mono, sanitizes NaN/Infs,
    applies zero-mean unit-variance normalization, and applies AASIST standard windowing.
    """

    TARGET_SAMPLE_RATE = 16000
    AASIST_STANDARD_SAMPLES = 64600  # ~4.0375 seconds at 16 kHz

    @classmethod
    def process_raw_audio(
        cls,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_format: str = "pcm_s16le",
        target_sample_rate: int = 16000,
        do_normalize: bool = True,
        fix_length: bool = False
    ) -> np.ndarray:
        """
        Validates, decodes, resamples, and normalizes audio for AASIST-L input.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise AudioProcessingError("Audio payload is empty (0 bytes)")

        # 1. Decode audio bytes
        waveform, sr = cls._decode_audio(audio_bytes, sample_rate, channels, audio_format)

        if waveform.size == 0:
            raise AudioProcessingError("Decoded audio waveform is empty")

        # 2. Sanitize NaN / Inf
        if np.isnan(waveform).any() or np.isinf(waveform).any():
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3. Convert multi-channel to mono
        if waveform.ndim > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
        elif waveform.ndim > 1:
            waveform = waveform.squeeze()

        # 4. Resample to 16 kHz using bandlimited polyphase sinc filter
        if sr != target_sample_rate:
            waveform = cls._resample_polyphase(waveform, sr, target_sample_rate)

        # 5. Zero-mean unit-variance normalization
        if do_normalize:
            waveform = cls._normalize_waveform(waveform)

        # 6. Optional AASIST-L fixed sample length padding/truncation
        if fix_length:
            waveform = cls._pad_or_truncate(waveform, cls.AASIST_STANDARD_SAMPLES)

        # Final sanitization
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

        if audio_bytes.startswith(b"RIFF") or format_lower in ["wav", "wave"]:
            try:
                with io.BytesIO(audio_bytes) as bio:
                    data, sr = sf.read(bio, dtype="float32")
                    return data, sr
            except Exception as e:
                raise AudioProcessingError(f"Failed to decode WAV audio header/stream: {str(e)}")

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

        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
                return data, sr
        except Exception:
            num_samples = len(audio_bytes) // 2
            int16_arr = np.frombuffer(audio_bytes[:num_samples * 2], dtype=np.int16)
            return int16_arr.astype(np.float32) / 32768.0, sample_rate

    @classmethod
    def _resample_polyphase(cls, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr <= 0 or target_sr <= 0:
            raise AudioProcessingError(f"Invalid sample rates: orig={orig_sr}, target={target_sr}")
        if orig_sr == target_sr:
            return waveform

        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        try:
            return scipy.signal.resample_poly(waveform, up, down)
        except Exception as e:
            raise AudioProcessingError(f"Polyphase resampling failed: {str(e)}")

    @classmethod
    def _normalize_waveform(cls, waveform: np.ndarray) -> np.ndarray:
        variance = float(np.var(waveform))
        mean = float(np.mean(waveform))
        if variance < 1e-12:
            return np.zeros_like(waveform)
        return (waveform - mean) / np.sqrt(variance + 1e-7)

    @classmethod
    def _pad_or_truncate(cls, waveform: np.ndarray, target_len: int) -> np.ndarray:
        if len(waveform) >= target_len:
            return waveform[:target_len]
        # Repeat or zero pad
        repeat_factor = int(np.ceil(target_len / len(waveform)))
        repeated = np.tile(waveform, repeat_factor)
        return repeated[:target_len]
