"""Speech-to-text module with Sarvam and Whisper placeholders."""


def transcribe_audio(audio_path: str, provider: str = "sarvam") -> str:
    """Transcribe an audio file using the selected STT provider."""
    if provider == "sarvam":
        return transcribe_with_sarvam(audio_path)
    if provider == "whisper":
        return transcribe_with_whisper(audio_path)
    raise ValueError(f"Unsupported STT provider: {provider}")


def transcribe_with_sarvam(audio_path: str) -> str:
    """Transcribe audio with Sarvam STT."""
    raise NotImplementedError("Connect Sarvam STT API here.")


def transcribe_with_whisper(audio_path: str) -> str:
    """Transcribe audio with Whisper."""
    raise NotImplementedError("Connect Whisper STT API here.")
