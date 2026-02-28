"""Text-to-speech module using Sarvam Bulbul."""

from pathlib import Path


def synthesize_speech(text: str, output_path: str = "response.wav") -> Path:
    """Synthesize a text response into speech."""
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    raise NotImplementedError("Connect Sarvam Bulbul TTS API here.")
