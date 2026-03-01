"""Text-to-speech helpers using Sarvam Bulbul."""

import base64
from pathlib import Path
import time
from typing import Any

import requests
from sarvamai import SarvamAI

from config import SARVAM_API_KEY, SARVAM_TTS_MODEL, SARVAM_TTS_URL
from pipeline.audio import save_audio_bytes


MAX_TTS_CHARS = 2_500
DEFAULT_SPEAKER = "anushka"
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
    "anushka",
    "abhilash",
    "manisha",
    "vidya",
    "arya",
    "karun",
    "hitesh",
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
    "amelia",
    "sophia",
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
    audio_bytes = synthesize_speech_bytes(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        speech_rate=speech_rate,
    )
    saved_path = save_audio_bytes(audio_bytes, output_path)
    return Path(saved_path)


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
        "pitch": pitch,
        "pace": speech_rate,
        "loudness": loudness,
        "speech_sample_rate": sample_rate,
        "enable_preprocessing": enable_preprocessing,
        "model": SARVAM_TTS_MODEL,
        "output_audio_codec": "wav",
    }
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
            return client.text_to_speech.convert(
                text=text.strip(),
                target_language_code=target_language_code,
                speaker=speaker,
                pitch=pitch,
                pace=speech_rate,
                loudness=loudness,
                speech_sample_rate=sample_rate,
                enable_preprocessing=enable_preprocessing,
                model=SARVAM_TTS_MODEL,
                output_audio_codec="wav",
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
