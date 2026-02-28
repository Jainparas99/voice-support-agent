"""Microphone recording and audio playback helpers."""


def record_audio(output_path: str = "input.wav", duration_seconds: int = 5) -> str:
    """Record microphone audio to a WAV file."""
    raise NotImplementedError("Add microphone recording implementation here.")


def play_audio(audio_path: str) -> None:
    """Play an audio file."""
    raise NotImplementedError("Add audio playback implementation here.")
