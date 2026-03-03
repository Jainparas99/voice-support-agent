"""Text-to-speech helpers using Sarvam Bulbul."""

import base64
import re
from pathlib import Path
import time
from typing import Any

import requests
from sarvamai import SarvamAI

from config import SARVAM_API_KEY, SARVAM_TTS_MODEL, SARVAM_TTS_URL
from pipeline.audio import get_audio_metadata, play_audio, save_audio_bytes


MAX_TTS_CHARS = 2_500
DEFAULT_SPEAKER = "priya"
DEFAULT_LANGUAGE_CODE = "en-IN"
DEFAULT_SPEECH_RATE = 1.0
DEFAULT_PITCH = 0.0
DEFAULT_LOUDNESS = 1.5
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MAX_RETRIES = 2

SUPPORTED_TTS_LANGUAGE_CODES = {
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
}
SUPPORTED_BULBUL_SPEAKERS = {
    "aditya",
    "ritu",
    "priya",
    "neha",
    "rahul",
    "pooja",
    "rohan",
    "simran",
    "kavya",
    "amit",
    "dev",
    "ishita",
    "shreya",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "roopa",
    "kabir",
    "aayan",
    "shubh",
    "ashutosh",
    "advait",
    "anand",
    "tanya",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "shruti",
    "suhani",
    "mohit",
    "kavitha",
    "rehan",
    "soham",
    "rupali",
    "niharika",
}
LANGUAGE_TO_DEFAULT_SPEAKER = {
    "en-IN": "priya",
    "hi-IN": "priya",
    "bn-IN": "priya",
    "gu-IN": "priya",
    "kn-IN": "shruti",
    "ml-IN": "priya",
    "mr-IN": "priya",
    "od-IN": "priya",
    "pa-IN": "priya",
    "ta-IN": "kavitha",
    "te-IN": "priya",
}
LANGUAGE_HINTS = {
    "hi-IN": ("ह", "नहीं", "नमस्ते", "कृपया", "धन्यवाद", "क्या", "कैसे"),
    "bn-IN": ("আমি", "আপনি", "ধন্যবাদ"),
    "gu-IN": ("તમે", "આભાર", "કૃપા"),
    "kn-IN": ("ನಾನು", "ಧನ್ಯವಾದ"),
    "ml-IN": ("ഞാൻ", "നന്ദി"),
    "mr-IN": ("आहे", "धन्यवाद"),
    "pa-IN": ("ਤੁਸੀਂ", "ਧੰਨਵਾਦ"),
    "ta-IN": ("நான்", "நன்றி"),
    "te-IN": ("నేను", "ధన్యవాదాలు"),
}


class TTSError(RuntimeError):
    """Raised when text-to-speech processing fails."""


def synthesize_speech(
    text: str,
    output_path: str = "response.wav",
    target_language_code: str = DEFAULT_LANGUAGE_CODE,
    speaker: str = DEFAULT_SPEAKER,
    speech_rate: float = DEFAULT_SPEECH_RATE,
) -> Path:
    """Synthesize text to a WAV file and return its path."""
    result = synthesize_speech_result(
        text=text,
        output_path=output_path,
        target_language_code=target_language_code,
        speaker=speaker,
        speech_rate=speech_rate,
    )
    return Path(str(result["audio_path"]))


def synthesize_speech_bytes(
    text: str,
    target_language_code: str = DEFAULT_LANGUAGE_CODE,
    speaker: str = DEFAULT_SPEAKER,
    speech_rate: float = DEFAULT_SPEECH_RATE,
    pitch: float = DEFAULT_PITCH,
    loudness: float = DEFAULT_LOUDNESS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    enable_preprocessing: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes:
    """Convert text to WAV audio bytes using Sarvam Bulbul v3."""
    validate_tts_input(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        speech_rate=speech_rate,
    )
    if not SARVAM_API_KEY:
        raise TTSError("SARVAM_API_KEY is missing. Add it to your .env file.")

    response = _convert_with_retries(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        speech_rate=speech_rate,
        pitch=pitch,
        loudness=loudness,
        sample_rate=sample_rate,
        enable_preprocessing=enable_preprocessing,
        max_retries=max_retries,
    )
    return _decode_tts_response(response)


def synthesize_speech_result(
    text: str,
    output_path: str = "response.wav",
    target_language_code: str | None = None,
    speaker: str | None = None,
    speech_rate: float = DEFAULT_SPEECH_RATE,
    pitch: float = DEFAULT_PITCH,
    loudness: float = DEFAULT_LOUDNESS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    enable_preprocessing: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Synthesize text and return audio path, bytes, settings, and metadata."""
    cleaned_text = clean_text_for_tts(text)
    resolved_language = target_language_code or detect_tts_language(cleaned_text)
    resolved_speaker = speaker or select_speaker(resolved_language)
    chunks = chunk_text_for_tts(cleaned_text)

    if len(chunks) > 1:
        raise TTSError("Long text chunking is available, but audio concatenation is not implemented yet.")

    audio_bytes = synthesize_speech_bytes(
        text=chunks[0],
        target_language_code=resolved_language,
        speaker=resolved_speaker,
        speech_rate=speech_rate,
        pitch=pitch,
        loudness=loudness,
        sample_rate=sample_rate,
        enable_preprocessing=enable_preprocessing,
        max_retries=max_retries,
    )
    audio_path = save_speech(audio_bytes, output_path)
    metadata = get_tts_metadata(
        audio_path=audio_path,
        text=cleaned_text,
        target_language_code=resolved_language,
        speaker=resolved_speaker,
        speech_rate=speech_rate,
    )

    return {
        "audio_path": str(audio_path),
        "audio_bytes": audio_bytes,
        "text": cleaned_text,
        "chunks": chunks,
        "target_language_code": resolved_language,
        "speaker": resolved_speaker,
        "speech_rate": speech_rate,
        "model": SARVAM_TTS_MODEL,
        "metadata": metadata,
    }


def clean_text_for_tts(text: str) -> str:
    """Clean text so it sounds natural when spoken aloud."""
    cleaned = text.strip()
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", "link", cleaned)
    cleaned = re.sub(r"[\*_~#>]+", "", cleaned)
    cleaned = re.sub(r"[-•]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def detect_tts_language(text: str, fallback: str = DEFAULT_LANGUAGE_CODE) -> str:
    """Infer a supported TTS language code from text characters/keywords."""
    if not text.strip():
        return fallback

    for language_code, hints in LANGUAGE_HINTS.items():
        if any(hint in text for hint in hints):
            return language_code

    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn-IN"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gu-IN"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn-IN"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "ml-IN"
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "pa-IN"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta-IN"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te-IN"

    return fallback


def select_speaker(
    target_language_code: str = DEFAULT_LANGUAGE_CODE,
    preferred_speaker: str | None = None,
) -> str:
    """Select a Bulbul v3 speaker for the target language."""
    if preferred_speaker:
        if preferred_speaker not in SUPPORTED_BULBUL_SPEAKERS:
            raise ValueError(f"Unsupported Bulbul speaker: {preferred_speaker}")
        return preferred_speaker

    return LANGUAGE_TO_DEFAULT_SPEAKER.get(target_language_code, DEFAULT_SPEAKER)


def chunk_text_for_tts(text: str, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    """Split text into chunks that fit Bulbul's per-request character limit."""
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        raise ValueError("Text cannot be empty.")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0.")
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?।])\s+", cleaned)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                candidate = f"{current} {word}".strip()
                if len(candidate) > max_chars and current:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def get_tts_metadata(
    audio_path: str | Path,
    text: str,
    target_language_code: str,
    speaker: str,
    speech_rate: float,
) -> dict[str, Any]:
    """Return metadata for a generated TTS file and synthesis settings."""
    audio_metadata = get_audio_metadata(audio_path)
    return {
        "audio_path": str(audio_path),
        "text_length": len(text),
        "model": SARVAM_TTS_MODEL,
        "target_language_code": target_language_code,
        "speaker": speaker,
        "speech_rate": speech_rate,
        **audio_metadata,
    }


def play_speech(
    text: str,
    target_language_code: str | None = None,
    speaker: str | None = None,
    speech_rate: float = DEFAULT_SPEECH_RATE,
) -> dict[str, Any]:
    """Synthesize text, play the generated audio, and return the TTS result."""
    result = synthesize_speech_result(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        speech_rate=speech_rate,
    )
    play_audio(str(result["audio_path"]))
    return result


def synthesize_speech_rest(
    text: str,
    target_language_code: str = DEFAULT_LANGUAGE_CODE,
    speaker: str = DEFAULT_SPEAKER,
    speech_rate: float = DEFAULT_SPEECH_RATE,
    pitch: float = DEFAULT_PITCH,
    loudness: float = DEFAULT_LOUDNESS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    enable_preprocessing: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes:
    """REST fallback matching Sarvam's base64 audio response shape."""
    validate_tts_input(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        speech_rate=speech_rate,
    )
    if not SARVAM_API_KEY:
        raise TTSError("SARVAM_API_KEY is missing. Add it to your .env file.")
    if not SARVAM_TTS_URL:
        raise TTSError("SARVAM_TTS_URL is missing. Add the Sarvam TTS endpoint to your .env file.")

    payload = {
        "inputs": [text.strip()],
        "target_language_code": target_language_code,
        "speaker": speaker,
        "pace": speech_rate,
        "speech_sample_rate": sample_rate,
        "enable_preprocessing": enable_preprocessing,
        "model": SARVAM_TTS_MODEL,
        "output_audio_codec": "wav",
    }
    if SARVAM_TTS_MODEL != "bulbul:v3":
        payload["pitch"] = pitch
        payload["loudness"] = loudness
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    response = _post_json_with_retries(
        SARVAM_TTS_URL,
        payload=payload,
        headers=headers,
        timeout=60,
        max_retries=max_retries,
    )
    result = response.json()
    return _decode_audio_b64(result["audios"][0])


def save_speech(audio_bytes: bytes, output_path: str = "response.wav") -> Path:
    """Save WAV audio bytes to disk and return the path."""
    return Path(save_audio_bytes(audio_bytes, output_path))


def validate_tts_input(
    text: str,
    target_language_code: str,
    speaker: str,
    speech_rate: float,
) -> None:
    """Validate text and Bulbul settings before making an API call."""
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"Text is too long for Bulbul TTS. Max characters: {MAX_TTS_CHARS}.")
    if target_language_code not in SUPPORTED_TTS_LANGUAGE_CODES:
        raise ValueError(f"Unsupported TTS language code: {target_language_code}")
    if speaker not in SUPPORTED_BULBUL_SPEAKERS:
        raise ValueError(f"Unsupported Bulbul speaker: {speaker}")
    if speech_rate <= 0:
        raise ValueError("speech_rate must be greater than 0.")


def _convert_with_retries(
    text: str,
    target_language_code: str,
    speaker: str,
    speech_rate: float,
    pitch: float,
    loudness: float,
    sample_rate: int,
    enable_preprocessing: bool,
    max_retries: int,
) -> Any:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
            kwargs: dict[str, Any] = {
                "text": text.strip(),
                "target_language_code": target_language_code,
                "speaker": speaker,
                "pace": speech_rate,
                "speech_sample_rate": sample_rate,
                "enable_preprocessing": enable_preprocessing,
                "model": SARVAM_TTS_MODEL,
                "output_audio_codec": "wav",
            }
            if SARVAM_TTS_MODEL != "bulbul:v3":
                kwargs["pitch"] = pitch
                kwargs["loudness"] = loudness

            return client.text_to_speech.convert(
                **kwargs,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)

    raise TTSError(f"Sarvam TTS failed: {last_error}") from last_error


def _decode_tts_response(response: Any) -> bytes:
    raw_response = response.model_dump() if hasattr(response, "model_dump") else response
    audios = raw_response.get("audios", []) if isinstance(raw_response, dict) else []
    if not audios:
        raise TTSError("Sarvam TTS response did not include audio.")
    return _decode_audio_b64(audios[0])


def _decode_audio_b64(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64)
    except Exception as exc:
        raise TTSError(f"Failed to decode Sarvam TTS audio: {exc}") from exc


def _post_json_with_retries(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    max_retries: int,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)

    raise TTSError(f"Sarvam TTS REST request failed: {last_error}") from last_error
