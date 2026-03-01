"""Quick smoke test for audio recording and Sarvam STT."""

import argparse
import json

from pipeline.audio import record_audio
from pipeline.stt import transcribe_sarvam


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or load audio and transcribe it with Sarvam.")
    parser.add_argument("--audio-path", help="Existing WAV file to transcribe.")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds.")
    parser.add_argument("--language-code", default="unknown", help="Sarvam language code.")
    parser.add_argument(
        "--mode",
        default="transcribe",
        choices=["transcribe", "translate", "verbatim", "translit", "codemix"],
        help="Saaras v3 output mode.",
    )
    args = parser.parse_args()

    audio_path = args.audio_path or record_audio(duration_seconds=args.duration)
    result = transcribe_sarvam(audio_path, language_code=args.language_code, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
