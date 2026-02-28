"""Speech-to-text helpers for Sarvam Saaras/Saarika and local Whisper."""

from functools import lru_cache
import base64
import json
from pathlib import Path
import re
import time
from typing import Any

import requests
from sarvamai import SarvamAI

from config import (
    SARVAM_API_KEY,
    SARVAM_STT_MODE,
    SARVAM_STT_MODEL,
    SARVAM_STT_URL,
    WHISPER_MODEL,
)
from pipeline.audio import (
    cleanup_temp_file,
    convert_to_sarvam_wav,
    get_audio_metadata,
    load_audio_bytes,
    validate_audio_file,
)


STTResult = dict[str, Any]
DEFAULT_RESULTS_DIR = Path("evaluation/results")
SUPPORTED_PROVIDERS = {"sarvam", "whisper"}
SUPPORTED_SARVAM_MODES = {"transcribe", "translate", "verbatim", "translit", "codemix"}
SUPPORTED_LANGUAGE_CODES = {
    "unknown",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "en-IN",
    "gu-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
}
DEFAULT_MODE_COMPARISON_ORDER = ["transcribe", "verbatim", "codemix", "translit", "translate"]


class STTError(RuntimeError):
    """Raised when speech-to-text processing fails."""


def transcribe_audio(
    audio_path: str,
    provider: str = "sarvam",
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
) -> str:
    """Transcribe an audio file using the selected STT provider."""
    result = transcribe(audio_path, provider=provider, language_code=language_code, mode=mode)
    return str(result.get("transcript", ""))


def transcribe(
    audio_path: str,
    provider: str = "sarvam",
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
) -> STTResult:
    """Transcribe an audio file and return the full STT result."""
    validate_provider(provider)
    validate_language_code(language_code)

    if provider == "sarvam":
        return transcribe_sarvam(audio_path, language_code=language_code, mode=mode)
    if provider == "whisper":
        return transcribe_whisper(audio_path)

    raise STTError(f"Unsupported STT provider: {provider}")


def transcribe_sarvam(
    audio_file_path: str,
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
) -> STTResult:
    """
    Transcribe audio using Sarvam Saaras v3 through the official SDK.

    Supported modes: transcribe, translate, verbatim, translit, codemix.
    """
    if not SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY is missing. Add it to your .env file.")
    validate_sarvam_mode(mode)
    validate_language_code(language_code)

    sarvam_ready_path: str | None = None
    try:
        sarvam_ready_path = convert_to_sarvam_wav(audio_file_path)
        validate_audio_file(sarvam_ready_path)

        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        with open(sarvam_ready_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model=SARVAM_STT_MODEL,
                mode=mode,
                language_code=language_code,
                input_audio_codec="wav",
            )
    except Exception as exc:
        raise STTError(f"Sarvam STT failed: {exc}") from exc
    finally:
        cleanup_temp_file(sarvam_ready_path)

    result = _normalize_sarvam_response(response, mode=mode)
    return _enrich_stt_result(result, provider="sarvam", audio_path=audio_file_path)


def transcribe_sarvam_rest(
    audio_file_path: str,
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
    with_timestamps: bool = False,
    max_retries: int = 2,
) -> STTResult:
    """
    Transcribe audio using the Sarvam REST payload shape.

    Keep this as a fallback while the project moves to the SDK-based Saaras v3 path.
    """
    if not SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY is missing. Add it to your .env file.")
    if not SARVAM_STT_URL:
        raise STTError("SARVAM_STT_URL is missing. Add the Sarvam STT endpoint to your .env file.")
    validate_sarvam_mode(mode)
    validate_language_code(language_code)

    sarvam_ready_path: str | None = None
    try:
        sarvam_ready_path = convert_to_sarvam_wav(audio_file_path)
        validate_audio_file(sarvam_ready_path)
        audio_b64 = base64.b64encode(load_audio_bytes(sarvam_ready_path)).decode("utf-8")
    except Exception as exc:
        raise STTError(f"Sarvam REST audio preparation failed: {exc}") from exc
    finally:
        cleanup_temp_file(sarvam_ready_path)

    payload = {
        "audio": audio_b64,
        "language_code": language_code,
        "model": SARVAM_STT_MODEL,
        "mode": mode,
        "with_timestamps": with_timestamps,
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    response = _post_json_with_retries(
        SARVAM_STT_URL,
        payload=payload,
        headers=headers,
        timeout=60,
        max_retries=max_retries,
    )
    result = response.json()

    normalized = {
        "request_id": result.get("request_id"),
        "transcript": result.get("transcript", ""),
        "language_code": result.get("language_code", language_code),
        "language_probability": result.get("language_probability"),
        "confidence": result.get("language_probability"),
        "timestamps": result.get("timestamps"),
        "diarized_transcript": result.get("diarized_transcript"),
        "mode": mode,
        "model": SARVAM_STT_MODEL,
        "raw_response": result,
    }
    return _enrich_stt_result(normalized, provider="sarvam-rest", audio_path=audio_file_path)


def transcribe_whisper(audio_file_path: str, model_name: str = WHISPER_MODEL) -> STTResult:
    """
    Fallback: transcribe using a local Whisper model.

    Useful for WER comparison benchmarking.
    """
    model = _load_whisper_model(model_name)
    result = model.transcribe(audio_file_path)
    normalized = {
        "request_id": None,
        "transcript": result["text"].strip(),
        "language_code": result.get("language", "en"),
        "language_probability": None,
        "confidence": None,
        "timestamps": None,
        "diarized_transcript": None,
        "mode": "transcribe",
        "model": model_name,
        "raw_response": result,
    }
    return _enrich_stt_result(normalized, provider="whisper", audio_path=audio_file_path)


def transcribe_with_fallback(
    audio_path: str,
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
    fallback_provider: str = "whisper",
) -> STTResult:
    """Try Sarvam first, then fall back to another provider if Sarvam fails."""
    validate_provider(fallback_provider)
    if fallback_provider == "sarvam":
        raise ValueError("fallback_provider must be different from the primary Sarvam provider.")

    try:
        result = transcribe_sarvam(audio_path, language_code=language_code, mode=mode)
        result["fallback_used"] = False
        result["primary_error"] = None
        return result
    except Exception as primary_error:
        fallback_result = transcribe(audio_path, provider=fallback_provider, language_code=language_code)
        fallback_result["fallback_used"] = True
        fallback_result["primary_error"] = str(primary_error)
        return fallback_result


def batch_transcribe(
    audio_paths: list[str],
    provider: str = "sarvam",
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
    continue_on_error: bool = True,
) -> list[STTResult]:
    """Transcribe multiple audio files for evaluation or benchmarking."""
    results: list[STTResult] = []

    for audio_path in audio_paths:
        try:
            results.append(
                transcribe(audio_path, provider=provider, language_code=language_code, mode=mode)
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            results.append(_error_result(audio_path, provider=provider, error=exc, mode=mode))

    return results


def compare_providers(
    audio_path: str,
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
) -> dict[str, STTResult]:
    """Run Sarvam and Whisper on the same audio file."""
    return {
        "sarvam": _safe_transcribe(audio_path, provider="sarvam", language_code=language_code, mode=mode),
        "whisper": _safe_transcribe(audio_path, provider="whisper", language_code=language_code, mode=mode),
    }


def compare_sarvam_modes(
    audio_path: str,
    language_code: str = "unknown",
    modes: list[str] | None = None,
) -> dict[str, STTResult]:
    """Run Saaras v3 across multiple output modes for the same audio file."""
    selected_modes = modes or DEFAULT_MODE_COMPARISON_ORDER
    return {
        mode: _safe_transcribe(audio_path, provider="sarvam", language_code=language_code, mode=mode)
        for mode in selected_modes
    }


def save_transcription_result(result: STTResult, output_path: str | None = None) -> str:
    """Save a normalized STT result as JSON and return the path."""
    if output_path is None:
        output_dir = DEFAULT_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / _default_result_filename(result))
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return output_path


def load_transcription_result(result_path: str) -> STTResult:
    """Load a previously saved STT result JSON file."""
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_transcript(
    transcript: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
) -> str:
    """Normalize transcript text for display or WER comparison."""
    normalized = " ".join(transcript.strip().split())
    if lowercase:
        normalized = normalized.lower()
    if remove_punctuation:
        normalized = re.sub(r"[^\w\s]", "", normalized, flags=re.UNICODE)
        normalized = " ".join(normalized.split())
    return normalized


def validate_provider(provider: str) -> None:
    """Validate an STT provider name."""
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported STT provider: {provider}")


def validate_sarvam_mode(mode: str) -> None:
    """Validate a Saaras v3 output mode."""
    if mode not in SUPPORTED_SARVAM_MODES:
        raise ValueError(f"Unsupported Sarvam mode: {mode}")


def validate_language_code(language_code: str) -> None:
    """Validate language codes supported by Sarvam STT."""
    if language_code not in SUPPORTED_LANGUAGE_CODES:
        raise ValueError(f"Unsupported language code: {language_code}")


def estimate_audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds for a WAV file."""
    return float(get_audio_metadata(audio_path)["duration_seconds"])


def transcribe_with_sarvam(
    audio_path: str,
    language_code: str = "unknown",
    mode: str = SARVAM_STT_MODE,
) -> str:
    """Backward-compatible helper that returns only the Sarvam transcript."""
    return str(
        transcribe_sarvam(audio_path, language_code=language_code, mode=mode).get("transcript", "")
    )


def transcribe_with_whisper(audio_path: str) -> str:
    """Backward-compatible helper that returns only the Whisper transcript."""
    return str(transcribe_whisper(audio_path).get("transcript", ""))


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    import whisper

    return whisper.load_model(model_name)


def _normalize_sarvam_response(response: Any, mode: str) -> STTResult:
    if hasattr(response, "model_dump"):
        raw_response = response.model_dump()
    elif hasattr(response, "dict"):
        raw_response = response.dict()
    elif isinstance(response, dict):
        raw_response = response
    else:
        raw_response = {
            "request_id": getattr(response, "request_id", None),
            "transcript": getattr(response, "transcript", ""),
            "timestamps": getattr(response, "timestamps", None),
            "diarized_transcript": getattr(response, "diarized_transcript", None),
            "language_code": getattr(response, "language_code", None),
            "language_probability": getattr(response, "language_probability", None),
        }

    language_probability = raw_response.get("language_probability")
    return {
        "request_id": raw_response.get("request_id"),
        "transcript": raw_response.get("transcript", ""),
        "language_code": raw_response.get("language_code"),
        "language_probability": language_probability,
        "confidence": language_probability,
        "timestamps": raw_response.get("timestamps"),
        "diarized_transcript": raw_response.get("diarized_transcript"),
        "mode": mode,
        "model": SARVAM_STT_MODEL,
        "raw_response": raw_response,
    }


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

    raise STTError(f"Sarvam REST request failed: {last_error}") from last_error


def _safe_transcribe(
    audio_path: str,
    provider: str,
    language_code: str,
    mode: str,
) -> STTResult:
    try:
        return transcribe(audio_path, provider=provider, language_code=language_code, mode=mode)
    except Exception as exc:
        return _error_result(audio_path, provider=provider, error=exc, mode=mode)


def _error_result(audio_path: str, provider: str, error: Exception, mode: str) -> STTResult:
    return {
        "request_id": None,
        "provider": provider,
        "source_audio_path": audio_path,
        "audio_duration_seconds": _safe_audio_duration(audio_path),
        "transcript": "",
        "normalized_transcript": "",
        "language_code": None,
        "language_probability": None,
        "confidence": None,
        "timestamps": None,
        "diarized_transcript": None,
        "mode": mode,
        "model": _model_for_provider(provider),
        "fallback_used": False,
        "error": str(error),
        "raw_response": None,
    }


def _enrich_stt_result(result: STTResult, provider: str, audio_path: str) -> STTResult:
    enriched = dict(result)
    transcript = str(enriched.get("transcript", ""))
    enriched["provider"] = provider
    enriched["source_audio_path"] = audio_path
    enriched["audio_duration_seconds"] = _safe_audio_duration(audio_path)
    enriched["normalized_transcript"] = normalize_transcript(transcript)
    enriched.setdefault("fallback_used", False)
    enriched.setdefault("error", None)
    return enriched


def _safe_audio_duration(audio_path: str) -> float | None:
    try:
        return estimate_audio_duration(audio_path)
    except Exception:
        return None


def _model_for_provider(provider: str) -> str | None:
    if provider.startswith("sarvam"):
        return SARVAM_STT_MODEL
    if provider == "whisper":
        return WHISPER_MODEL
    return None


def _default_result_filename(result: STTResult) -> str:
    source_path = Path(str(result.get("source_audio_path") or "audio"))
    provider = str(result.get("provider") or "stt")
    mode = str(result.get("mode") or "transcribe")
    request_id = str(result.get("request_id") or "local")
    filename = f"{source_path.stem}_{provider}_{mode}_{request_id}.json"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
