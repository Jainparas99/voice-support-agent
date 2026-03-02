"""CLI entry point for the voice support agent."""

import argparse
import json
from pathlib import Path
from time import monotonic
from datetime import datetime

from dotenv import load_dotenv

from pipeline.audio import cleanup_temp_file, play_audio, record_audio
from pipeline.llm import get_support_response_with_metadata
from pipeline.stt import transcribe_sarvam
from pipeline.tts import SUPPORTED_TTS_LANGUAGE_CODES, detect_tts_language, synthesize_speech


load_dotenv()


def run(
    audio_path: str,
    tts_language: str | None = None,
    speaker: str | None = None,
    stt_language_code: str = "unknown",
    log_dir: str | None = None,
) -> Path:
    """Run the full speech-to-speech pipeline for one audio file."""
    result = process_turn(
        audio_path=audio_path,
        conversation_history=[],
        tts_language=tts_language,
        speaker=speaker,
        playback=False,
        cleanup_input_audio=False,
        log_dir=log_dir,
        stt_language_code=stt_language_code,
    )
    return Path(str(result["output_path"]))


def run_voice_agent(
    duration_seconds: int = 5,
    playback: bool = True,
    tts_language: str | None = None,
    speaker: str | None = None,
    log_dir: str | None = None,
    stt_language_code: str = "unknown",
) -> None:
    """Run the interactive microphone-based voice support loop."""
    print("=== Multilingual Voice Support Agent ===")
    print("Supports: English, Hindi, Hinglish (code-switched)")
    print("Type Ctrl+C to exit\n")

    conversation_history = []

    while True:
        try:
            user_input = input(
                f"Press Enter to start speaking ({duration_seconds} seconds), or type 'q' to quit: "
            ).strip()
            if user_input.lower() in {"q", "quit", "exit", "stop"}:
                print("Exiting. Goodbye!")
                break

            audio_path = record_audio(duration_seconds=duration_seconds)
            result = process_turn(
                audio_path=audio_path,
                conversation_history=conversation_history,
                tts_language=tts_language,
                speaker=speaker,
                playback=playback,
                cleanup_input_audio=True,
                log_dir=log_dir,
                stt_language_code=stt_language_code,
            )
            conversation_history = list(result["conversation_history"])
            _print_turn_result(result)

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as exc:
            print(f"Error: {exc}\n")


def process_turn(
    audio_path: str,
    conversation_history: list[dict[str, str]] | None = None,
    tts_language: str | None = None,
    speaker: str | None = None,
    playback: bool = True,
    cleanup_input_audio: bool = False,
    log_dir: str | None = None,
    stt_language_code: str = "unknown",
) -> dict[str, object]:
    """Process one user turn through STT, LLM, and TTS."""
    started_at = monotonic()
    audio_path_str = str(audio_path)

    try:
        stt_started_at = monotonic()
        stt_result = transcribe_sarvam(audio_path_str, language_code=stt_language_code)
        stt_duration = monotonic() - stt_started_at
        transcript = str(stt_result.get("transcript", "")).strip()
        detected_lang = str(stt_result.get("language_code") or "unknown")
        if not transcript:
            raise ValueError("Didn't catch any speech. Try again.")

        llm_started_at = monotonic()
        llm_result = get_support_response_with_metadata(
            transcript,
            conversation_history=conversation_history,
        )
        llm_duration = monotonic() - llm_started_at
        response_text = str(llm_result["text"])
        updated_history = list(llm_result["conversation_history"])

        resolved_tts_language = _tts_language_from_stt(
            detected_lang,
            response_text=response_text,
            override=tts_language,
        )
        tts_started_at = monotonic()
        tts_kwargs = {"target_language_code": resolved_tts_language}
        if speaker:
            tts_kwargs["speaker"] = speaker
        output_path = synthesize_speech(response_text, output_path=_output_path(log_dir), **tts_kwargs)
        tts_duration = monotonic() - tts_started_at

        playback_duration = 0.0
        if playback:
            playback_started_at = monotonic()
            play_audio(output_path)
            playback_duration = monotonic() - playback_started_at

        result: dict[str, object] = {
            "source_audio_path": audio_path_str,
            "transcript": transcript,
            "detected_language": detected_lang,
            "stt_result": stt_result,
            "response_text": response_text,
            "conversation_history": updated_history,
            "llm_result": llm_result,
            "tts_language": resolved_tts_language,
            "speaker": speaker,
            "output_path": str(output_path),
            "timings": {
                "stt_seconds": round(stt_duration, 3),
                "llm_seconds": round(llm_duration, 3),
                "tts_seconds": round(tts_duration, 3),
                "playback_seconds": round(playback_duration, 3),
                "total_seconds": round(monotonic() - started_at, 3),
            },
        }
        _log_turn(result, log_dir)
        return result
    finally:
        if cleanup_input_audio:
            cleanup_temp_file(audio_path_str)


def _tts_language_from_stt(
    language_code: str,
    response_text: str,
    override: str | None = None,
) -> str:
    """Map STT language detection to a supported TTS language code."""
    if override:
        return override
    if language_code in SUPPORTED_TTS_LANGUAGE_CODES:
        return language_code
    inferred = detect_tts_language(response_text, fallback="hi-IN")
    if inferred in SUPPORTED_TTS_LANGUAGE_CODES:
        return inferred
    return "hi-IN"


def _output_path(log_dir: str | None) -> str:
    """Return a per-turn output path, optionally inside a log directory."""
    if not log_dir:
        return "response.wav"
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(log_path / f"response_{timestamp}.wav")


def _log_turn(result: dict[str, object], log_dir: str | None) -> None:
    """Append turn metadata to a JSONL session log."""
    if not log_dir:
        return
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    session_log = log_path / "session.jsonl"
    with session_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def _print_turn_result(result: dict[str, object]) -> None:
    """Print a readable CLI summary for one completed turn."""
    timings = dict(result["timings"])
    print(f"\nYou said [{result['detected_language']}]: {result['transcript']}")
    print(f"Agent: {result['response_text']}")
    print(
        "Timings: "
        f"STT {timings['stt_seconds']}s | "
        f"LLM {timings['llm_seconds']}s | "
        f"TTS {timings['tts_seconds']}s | "
        f"Total {timings['total_seconds']}s"
    )
    print(f"Response audio: {result['output_path']}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for mic and file demo modes."""
    parser = argparse.ArgumentParser(description="Run the multilingual voice support agent.")
    parser.add_argument("--mode", choices=["mic", "file"], default="mic", help="Input mode.")
    parser.add_argument("--audio-path", help="Path to an existing audio file for file mode.")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds.")
    parser.add_argument("--tts-language", help="Override TTS language code, for example en-IN.")
    parser.add_argument("--speaker", help="Override the Bulbul speaker name.")
    parser.add_argument(
        "--stt-language-code",
        default="unknown",
        help="Sarvam STT language code. Use 'unknown' for auto-detect.",
    )
    parser.add_argument("--no-playback", action="store_true", help="Skip audio playback.")
    parser.add_argument("--log-dir", help="Directory to save response audio and session logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "file":
        if not args.audio_path:
            raise SystemExit("--audio-path is required when --mode file is used.")
        result = process_turn(
            audio_path=args.audio_path,
            conversation_history=[],
            tts_language=args.tts_language,
            speaker=args.speaker,
            playback=not args.no_playback,
            cleanup_input_audio=False,
            log_dir=args.log_dir,
            stt_language_code=args.stt_language_code,
        )
        _print_turn_result(result)
        return

    run_voice_agent(
        duration_seconds=args.duration,
        playback=not args.no_playback,
        tts_language=args.tts_language,
        speaker=args.speaker,
        log_dir=args.log_dir,
        stt_language_code=args.stt_language_code,
    )


if __name__ == "__main__":
    main()
