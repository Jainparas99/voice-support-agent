"""Microphone recording, playback, and WAV preparation helpers."""

from pathlib import Path
import tempfile
import time
from typing import Any, Union

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
import sounddevice as sd


SAMPLE_RATE = 16_000
CHANNELS = 1
DEFAULT_SILENCE_THRESHOLD = 500
DEFAULT_SILENCE_SECONDS = 1.0
DEFAULT_CHUNK_SECONDS = 0.25

AudioInput = Union[bytes, str, Path]


def _temp_wav_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        return tmp.name


def _to_mono(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.ndim == 1:
        return audio_data
    return audio_data.mean(axis=1)


def _to_int16(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.dtype == np.int16:
        return audio_data

    if np.issubdtype(audio_data.dtype, np.unsignedinteger):
        midpoint = np.iinfo(audio_data.dtype).max / 2
        centered = (audio_data.astype(np.float32) - midpoint) / midpoint
        return (np.clip(centered, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)

    if np.issubdtype(audio_data.dtype, np.floating):
        clipped = np.clip(audio_data, -1.0, 1.0)
        return (clipped * np.iinfo(np.int16).max).astype(np.int16)

    max_abs = np.max(np.abs(audio_data)) if audio_data.size else 0
    if max_abs == 0:
        return audio_data.astype(np.int16)

    normalized = audio_data.astype(np.float32) / max_abs
    return (normalized * np.iinfo(np.int16).max).astype(np.int16)


def _resample(audio_data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio_data

    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(audio_data, up, down)


def _peak_amplitude(audio_data: np.ndarray) -> float:
    if audio_data.size == 0:
        return 0.0
    return float(np.max(np.abs(audio_data.astype(np.float32))))


def record_audio(duration_seconds: int = 5, output_path: str | None = None) -> str:
    """Record microphone audio to a 16 kHz mono WAV file and return its path."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0.")

    print(f"Recording for {duration_seconds} seconds... Speak now.")
    audio_data = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16,
    )
    sd.wait()
    print("Recording complete.")

    if output_path is None:
        output_path = _temp_wav_path()

    wav.write(output_path, SAMPLE_RATE, audio_data)
    return output_path


def record_until_silence(
    output_path: str | None = None,
    max_duration_seconds: int = 30,
    silence_seconds: float = DEFAULT_SILENCE_SECONDS,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> str:
    """Record until sustained silence or max duration is reached."""
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be greater than 0.")
    if silence_seconds <= 0:
        raise ValueError("silence_seconds must be greater than 0.")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than 0.")

    print("Recording... speak now. Recording stops after silence.")
    chunks: list[np.ndarray] = []
    silent_for = 0.0
    voice_started = False
    started_at = time.monotonic()
    chunk_frames = max(1, int(chunk_seconds * SAMPLE_RATE))

    while time.monotonic() - started_at < max_duration_seconds:
        chunk = sd.rec(
            chunk_frames,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
        )
        sd.wait()
        chunks.append(chunk)

        if _peak_amplitude(chunk) >= silence_threshold:
            voice_started = True
            silent_for = 0.0
        elif voice_started:
            silent_for += chunk_seconds
            if silent_for >= silence_seconds:
                break

    print("Recording complete.")
    audio_data = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, CHANNELS), dtype=np.int16)
    )

    if output_path is None:
        output_path = _temp_wav_path()

    wav.write(output_path, SAMPLE_RATE, audio_data)
    return output_path


def play_audio(audio: AudioInput) -> None:
    """Play WAV audio from bytes or from a file path."""
    tmp_path: str | None = None

    try:
        if isinstance(audio, bytes):
            tmp_path = save_audio_bytes(audio)
            audio_path = tmp_path
        else:
            audio_path = str(audio)

        rate, data = wav.read(audio_path)
        sd.play(data, rate)
        sd.wait()
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def get_audio_metadata(audio_path: str | Path) -> dict[str, Any]:
    """Return basic metadata for a readable WAV file."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sample_rate, data = wav.read(path)
    frames = int(data.shape[0]) if data.ndim > 0 else 0
    channels = 1 if data.ndim == 1 else int(data.shape[1])
    duration_seconds = frames / sample_rate if sample_rate else 0.0

    return {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": duration_seconds,
        "frames": frames,
        "dtype": str(data.dtype),
        "file_size_bytes": path.stat().st_size,
    }


def validate_audio_file(
    audio_path: str | Path,
    required_sample_rate: int = SAMPLE_RATE,
    required_channels: int = CHANNELS,
) -> None:
    """Raise a helpful error if a WAV file is not ready for Sarvam STT."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {path.suffix or 'no extension'}")

    metadata = get_audio_metadata(path)
    if metadata["frames"] == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if metadata["sample_rate"] != required_sample_rate:
        raise ValueError(
            f"Expected {required_sample_rate} Hz audio, got {metadata['sample_rate']} Hz: {path}"
        )
    if metadata["channels"] != required_channels:
        raise ValueError(
            f"Expected {required_channels} channel audio, got {metadata['channels']}: {path}"
        )


def convert_to_sarvam_wav(input_path: str | Path, output_path: str | None = None) -> str:
    """Convert a readable WAV file to Sarvam-ready 16 kHz mono int16 WAV."""
    input_file = Path(input_path)
    if input_file.suffix.lower() != ".wav":
        raise ValueError("Only WAV conversion is supported without ffmpeg. Upload or record a WAV file.")
    if output_path is None:
        output_path = _temp_wav_path()

    source_rate, audio_data = wav.read(input_file)
    mono_data = _to_mono(audio_data)
    resampled = _resample(mono_data, int(source_rate), SAMPLE_RATE)
    int16_data = _to_int16(resampled)

    wav.write(output_path, SAMPLE_RATE, int16_data)
    return output_path


def trim_silence(
    audio_path: str | Path,
    output_path: str | None = None,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    padding_seconds: float = 0.1,
) -> str:
    """Remove leading and trailing silence from a WAV file."""
    if padding_seconds < 0:
        raise ValueError("padding_seconds cannot be negative.")
    if output_path is None:
        output_path = _temp_wav_path()

    sample_rate, audio_data = wav.read(audio_path)
    mono_for_detection = np.abs(_to_mono(audio_data).astype(np.float32))
    voiced_indices = np.where(mono_for_detection > silence_threshold)[0]

    if voiced_indices.size == 0:
        wav.write(output_path, sample_rate, audio_data)
        return output_path

    padding_frames = int(padding_seconds * sample_rate)
    start = max(int(voiced_indices[0]) - padding_frames, 0)
    end = min(int(voiced_indices[-1]) + padding_frames + 1, audio_data.shape[0])
    wav.write(output_path, sample_rate, audio_data[start:end])
    return output_path


def normalize_volume(
    audio_path: str | Path,
    output_path: str | None = None,
    target_peak: float = 0.9,
) -> str:
    """Normalize a WAV file to a target peak amplitude and return the output path."""
    if not 0 < target_peak <= 1.0:
        raise ValueError("target_peak must be between 0 and 1.")
    if output_path is None:
        output_path = _temp_wav_path()

    sample_rate, audio_data = wav.read(audio_path)
    float_data = audio_data.astype(np.float32)
    peak = _peak_amplitude(float_data)

    if peak == 0:
        wav.write(output_path, sample_rate, audio_data)
        return output_path

    target = target_peak * np.iinfo(np.int16).max
    normalized = np.clip(float_data * (target / peak), -target, target)
    wav.write(output_path, sample_rate, normalized.astype(np.int16))
    return output_path


def save_audio_bytes(audio_bytes: bytes, output_path: str | None = None) -> str:
    """Save audio bytes to a WAV file and return its path."""
    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty.")
    if output_path is None:
        output_path = _temp_wav_path()

    Path(output_path).write_bytes(audio_bytes)
    return output_path


def load_audio_bytes(audio_path: str | Path) -> bytes:
    """Load an audio file as bytes for API upload."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path.read_bytes()


def cleanup_temp_file(audio_path: str | Path | None) -> None:
    """Delete a temporary audio file if it exists."""
    if audio_path is None:
        return
    Path(audio_path).unlink(missing_ok=True)


def list_audio_devices() -> list[dict[str, Any]]:
    """Return available audio input/output devices."""
    devices = sd.query_devices()
    return [
        {
            "index": index,
            "name": str(device["name"]),
            "max_input_channels": int(device["max_input_channels"]),
            "max_output_channels": int(device["max_output_channels"]),
            "default_sample_rate": float(device["default_samplerate"]),
        }
        for index, device in enumerate(devices)
    ]
