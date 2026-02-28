"""CLI entry point for the voice support agent."""

from pathlib import Path

from pipeline.llm import generate_response
from pipeline.stt import transcribe_audio
from pipeline.tts import synthesize_speech


def run(audio_path: str) -> Path:
    """Run the full speech-to-speech pipeline for one audio file."""
    transcript = transcribe_audio(audio_path)
    response_text = generate_response(transcript)
    return synthesize_speech(response_text)


def main() -> None:
    audio_path = input("Path to input audio file: ").strip()
    output_path = run(audio_path)
    print(f"Response audio saved to: {output_path}")


if __name__ == "__main__":
    main()
