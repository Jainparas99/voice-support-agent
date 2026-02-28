"""Microphone recording and audio playback helpers."""

from pathlib import Path
import tempfile
from typing import Union

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd


SAMPLE_RATE = 16_000
CHANNELS = 1


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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

    wav.write(output_path, SAMPLE_RATE, audio_data)
    return output_path


def play_audio(audio: Union[bytes, str, Path]) -> None:
    """Play WAV audio from bytes or from a file path."""
    tmp_path: str | None = None

    try:
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            audio_path = tmp_path
        else:
            audio_path = str(audio)

        rate, data = wav.read(audio_path)
        sd.play(data, rate)
        sd.wait()
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
