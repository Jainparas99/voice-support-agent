"""Run WER/CER evaluation for Sarvam and Whisper STT on benchmark samples."""

import json
from pathlib import Path
from typing import Any

from jiwer import cer, wer

from pipeline.stt import normalize_transcript, transcribe_sarvam, transcribe_whisper


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "test_samples"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"
RESULTS_PATH = ROOT / "results.json"


def run_evaluation() -> list[dict[str, Any]]:
    """Benchmark Sarvam and Whisper against the configured ground-truth samples."""
    samples = load_samples()
    results: list[dict[str, Any]] = []

    for sample in samples:
        audio_path = SAMPLES_DIR / sample["file"]
        ground_truth = normalize_transcript(sample["ground_truth"], lowercase=True)

        sarvam_result = transcribe_sarvam(str(audio_path))
        sarvam_transcript = normalize_transcript(
            str(sarvam_result.get("transcript", "")),
            lowercase=True,
        )
        sarvam_wer = wer(ground_truth, sarvam_transcript)
        sarvam_cer = cer(ground_truth, sarvam_transcript)

        whisper_result = transcribe_whisper(str(audio_path))
        whisper_transcript = normalize_transcript(
            str(whisper_result.get("transcript", "")),
            lowercase=True,
        )
        whisper_wer = wer(ground_truth, whisper_transcript)
        whisper_cer = cer(ground_truth, whisper_transcript)

        improvement = _compute_improvement(sarvam_wer, whisper_wer)
        result = {
            "file": sample["file"],
            "type": sample["type"],
            "ground_truth": ground_truth,
            "sarvam_transcript": sarvam_transcript,
            "whisper_transcript": whisper_transcript,
            "sarvam_wer": round(sarvam_wer, 4),
            "whisper_wer": round(whisper_wer, 4),
            "sarvam_cer": round(sarvam_cer, 4),
            "whisper_cer": round(whisper_cer, 4),
            "improvement": improvement,
            "sarvam_language_code": sarvam_result.get("language_code"),
            "whisper_language_code": whisper_result.get("language_code"),
        }
        results.append(result)
        _print_result(result)

    save_results(results)
    return results


def load_samples() -> list[dict[str, str]]:
    """Load benchmark sample definitions from JSON."""
    raw_samples = json.loads(GROUND_TRUTH_PATH.read_text())

    if isinstance(raw_samples, list):
        return [validate_sample(sample) for sample in raw_samples]

    if isinstance(raw_samples, dict):
        return [
            validate_sample(
                {
                    "file": filename,
                    "ground_truth": transcript,
                    "type": Path(filename).stem,
                }
            )
            for filename, transcript in raw_samples.items()
        ]

    raise ValueError("ground_truth.json must contain either a list of sample objects or a filename map.")


def validate_sample(sample: dict[str, Any]) -> dict[str, str]:
    """Validate one evaluation sample definition."""
    required_keys = {"file", "ground_truth", "type"}
    missing = required_keys - set(sample)
    if missing:
        raise ValueError(f"Sample is missing required keys: {sorted(missing)}")

    file_name = str(sample["file"])
    audio_path = SAMPLES_DIR / file_name
    if not audio_path.exists():
        raise FileNotFoundError(f"Evaluation sample not found: {audio_path}")

    return {
        "file": file_name,
        "ground_truth": str(sample["ground_truth"]),
        "type": str(sample["type"]),
    }


def save_results(results: list[dict[str, Any]], output_path: Path = RESULTS_PATH) -> Path:
    """Save evaluation results to JSON."""
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return output_path


def _compute_improvement(sarvam_wer: float, whisper_wer: float) -> float:
    if whisper_wer == 0:
        return 0.0 if sarvam_wer == 0 else -100.0
    return round((whisper_wer - sarvam_wer) / whisper_wer * 100, 1)


def _print_result(result: dict[str, Any]) -> None:
    print(f"\n[{str(result['type']).upper()}]")
    print(f"  Ground truth:       {result['ground_truth']}")
    print(f"  Sarvam transcript:  {result['sarvam_transcript']}")
    print(f"  Whisper transcript: {result['whisper_transcript']}")
    print(
        "  WER: "
        f"Sarvam {result['sarvam_wer']:.2%} | "
        f"Whisper {result['whisper_wer']:.2%}"
    )
    print(
        "  CER: "
        f"Sarvam {result['sarvam_cer']:.2%} | "
        f"Whisper {result['whisper_cer']:.2%}"
    )
    print(f"  Sarvam advantage: {result['improvement']}%")


if __name__ == "__main__":
    run_evaluation()
