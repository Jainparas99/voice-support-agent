"""CLI entry point for the voice support agent."""

from pathlib import Path

from dotenv import load_dotenv

from pipeline.audio import play_audio, record_audio
from pipeline.llm import get_support_response
from pipeline.stt import transcribe_sarvam
from pipeline.tts import synthesize_speech


load_dotenv()


def run(audio_path: str) -> Path:
    """Run the full speech-to-speech pipeline for one audio file."""
    stt_result = transcribe_sarvam(audio_path)
    transcript = str(stt_result.get("transcript", ""))
    detected_lang = str(stt_result.get("language_code") or "unknown")

    if not transcript.strip():
        raise ValueError("No transcript was detected from the input audio.")

    response_text, _ = get_support_response(transcript)
    return synthesize_speech(
        response_text,
        target_language_code=_tts_language_from_stt(detected_lang),
    )


def run_voice_agent() -> None:
    """Run the interactive microphone-based voice support loop."""
    print("=== Multilingual Voice Support Agent ===")
    print("Supports: English, Hindi, Hinglish (code-switched)")
    print("Type Ctrl+C to exit\n")

    conversation_history = []

    while True:
        try:
            input("Press Enter to start speaking (5 seconds)...")

            audio_path = record_audio(duration_seconds=5)
            stt_result = transcribe_sarvam(audio_path)
            transcript = str(stt_result.get("transcript", ""))
            detected_lang = str(stt_result.get("language_code") or "unknown")
            print(f"\nYou said [{detected_lang}]: {transcript}")

            if not transcript.strip():
                print("Didn't catch that. Try again.\n")
                continue

            response_text, conversation_history = get_support_response(
                transcript,
                conversation_history,
            )
            print(f"Agent: {response_text}")

            output_path = synthesize_speech(
                response_text,
                target_language_code=_tts_language_from_stt(detected_lang),
            )
            play_audio(output_path)
            print()

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break


def _tts_language_from_stt(language_code: str) -> str:
    """Map STT language detection to a supported TTS language code."""
    if language_code and language_code != "unknown":
        return language_code
    return "hi-IN"


def main() -> None:
    run_voice_agent()


if __name__ == "__main__":
    main()
